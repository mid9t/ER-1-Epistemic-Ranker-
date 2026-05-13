from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from .config import DAEDLConfig
from .data import (
    EpicRankerQuadDataset,
    OodFlatDataset,
    collate_flat,
    collate_quads,
    ensure_phase2_inputs,
)
from .density import GDADensityModel
from .losses import (
    binary_kl_penalty,
    brier_score_loss,
    get_expected_probs,
    get_kl_weight,
    get_smoothed_one_hot,
    s_ordering_loss,
    s_regularisation_loss,
    vacuity_loss_fn,
)
from .metrics import (
    base_accuracy,
    ece_from_probs,
    eru_mutual_information,
    mc_dropout_epistemic_variance,
    profile_uncertainty,
)
from .model import BertWithEvidentialHead, unfreeze_last_bert_layers


@dataclass
class TrainArtifacts:
    model: BertWithEvidentialHead
    gda: GDADensityModel
    train_loader: DataLoader
    test_loader: DataLoader
    ood_eval_loader: DataLoader
    density_active: bool
    train_losses: list[float]
    val_accs: list[float]
    epoch_times: list[float]


def setup_environment(config: DAEDLConfig) -> torch.device:
    set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    config.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = config.device
    print("Environment setup complete.")
    print(f"Seed: {config.seed}")
    print(f"Device: {device}")
    print(f"Processed dir: {config.active_processed_dir}")
    return device


def build_dataloaders(config: DAEDLConfig, tokenizer):
    train_path, test_path, ood_path = ensure_phase2_inputs(config)
    train_ds = EpicRankerQuadDataset(train_path, tokenizer, config.max_length)
    test_ds = EpicRankerQuadDataset(test_path, tokenizer, config.max_length)
    ood_train_ds = OodFlatDataset(ood_path, tokenizer, config.max_length)
    ood_eval_ds = OodFlatDataset(test_path, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_num_workers,
        persistent_workers=config.train_persistent_workers and config.train_num_workers > 0,
        collate_fn=collate_quads,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.eval_num_workers,
        collate_fn=collate_quads,
    )
    ood_train_loader = DataLoader(
        ood_train_ds,
        batch_size=config.batch_size * 4,
        shuffle=True,
        num_workers=config.eval_num_workers,
        collate_fn=collate_flat,
    )
    ood_eval_loader = DataLoader(
        ood_eval_ds,
        batch_size=config.batch_size * 4,
        shuffle=False,
        num_workers=config.eval_num_workers,
        collate_fn=collate_flat,
    )
    return train_loader, test_loader, ood_train_loader, ood_eval_loader


@torch.no_grad()
def extract_embeddings_for_gda(model, loader, gda_device):
    model.eval()
    all_embs, all_pair_codes = [], []
    pair_to_code = {"positive": 0, "hard_negative": 1, "easy_negative": 2}
    for batch in loader:
        indist_mask = torch.tensor([pt != "ood" for pt in batch["pair_types"]], dtype=torch.bool)
        if not indist_mask.any():
            continue
        model(
            input_ids=batch["input_ids"][indist_mask].to(gda_device),
            attention_mask=batch["attention_mask"][indist_mask].to(gda_device),
            token_type_ids=batch["token_type_ids"][indist_mask].to(gda_device),
        )
        all_embs.append(model.feature.cpu().float())
        codes = torch.tensor(
            [pair_to_code[pt] for pt, keep in zip(batch["pair_types"], indist_mask) if keep.item()]
        )
        all_pair_codes.append(codes)
    model.train()
    return torch.cat(all_embs, dim=0), torch.cat(all_pair_codes, dim=0)


def get_ood_batch(ood_iter, ood_loader):
    try:
        return next(ood_iter), ood_iter
    except StopIteration:
        ood_iter = iter(ood_loader)
        return next(ood_iter), ood_iter


def quick_gradient_check(model, device, num_classes: int, max_length: int):
    model.train()
    ids = torch.randint(0, 30522, (4, max_length), device=device)
    mask = torch.ones_like(ids)
    tids = torch.zeros_like(ids)
    target = torch.randint(0, num_classes, (4,), device=device)
    _, alpha = model(input_ids=ids, attention_mask=mask, token_type_ids=tids)
    loss = brier_score_loss(alpha, target, num_classes=num_classes)
    assert torch.isfinite(loss), "Loss NaN/Inf"
    model.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.head.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), "No head gradients"
    assert all(torch.isfinite(g).all() for g in grads), "NaN/Inf in head gradients"
    print("Gradient check passed.")


def forward_with_density(model, gda, batch_ids, batch_mask, batch_tids, density_active):
    if density_active:
        outputs = model.bert(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            token_type_ids=batch_tids,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        model.feature = cls
        density_score = gda.score(cls)
        _, alpha = model.head(cls, density_score=density_score)
    else:
        _, alpha = model(input_ids=batch_ids, attention_mask=batch_mask, token_type_ids=batch_tids)
    return alpha


def train(config: DAEDLConfig) -> TrainArtifacts:
    device = setup_environment(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_loader, test_loader, ood_train_loader, ood_eval_loader = build_dataloaders(config, tokenizer)
    ood_iter = iter(ood_train_loader)

    model = BertWithEvidentialHead(
        model_name=config.model_name,
        num_classes=config.num_classes,
        dropout_p=config.mc_dropout_p,
    ).to(device)
    unfreeze_last_bert_layers(model)
    model.head.set_temperature(1.0)
    quick_gradient_check(model, device, config.num_classes, config.max_length)

    optimizer = optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": config.lr_head},
            {"params": model.bert.encoder.layer[11].parameters(), "lr": config.lr_layer11},
            {"params": model.bert.encoder.layer[10].parameters(), "lr": config.lr_layer10},
            {"params": model.bert.encoder.layer[9].parameters(), "lr": config.lr_layer9},
        ],
        weight_decay=config.weight_decay,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    gda = GDADensityModel(
        n_components=config.gda_pca_components,
        temperature=config.gda_density_temp,
    )
    train_losses, val_accs, epoch_times = [], [], []

    print("\n--- Phase 2 Training (DAEDL) ---")
    print(f"  Epochs: {config.epochs} ({config.warmup_epochs} warm-up)")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Train workers: {config.train_num_workers}")
    print(f"  GDA PCA: 768->{config.gda_pca_components} (QDA, whiten=True)")

    for epoch in range(1, config.epochs + 1):
        density_active = gda.fitted
        if epoch == config.warmup_epochs + 1 and not gda.fitted:
            print(f"\n[Epoch {epoch}] Warm-up done. Fitting GDA...")
            train_embs, train_labs = extract_embeddings_for_gda(model, train_loader, device)
            gda.fit(train_embs, train_labs)
            gda.to(device)

            all_diag_embs, all_diag_types = [], []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device),
                    )
                    all_diag_embs.append(model.feature.cpu().float())
                    all_diag_types.extend(batch["pair_types"])
                for batch in ood_eval_loader:
                    model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch["token_type_ids"].to(device),
                    )
                    all_diag_embs.append(model.feature.cpu().float())
                    all_diag_types.extend(batch["pair_types"])
                    break
            gda.density_histogram(torch.cat(all_diag_embs), all_diag_types, title=f"Pre-Phase-B (epoch {epoch})")
            density_active = True

        kl_weight = get_kl_weight(
            epoch,
            config.epochs,
            config.kl_max_weight,
            config.kl_start_epoch,
        )
        use_hard = epoch >= config.hard_neg_start
        model.train()
        for layer_idx in (9, 10, 11):
            model.bert.encoder.layer[layer_idx].train()

        total_loss, total_count, batch_idx = 0.0, 0, 0
        epoch_start = time.time()

        for batch in train_loader:
            batch_idx += 1
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            token_tids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            pair_types = batch["pair_types"]

            sample_weights = torch.tensor(
                [
                    config.pair_weights.get(pt, 1.0)
                    if (pt != "hard_negative" or use_hard)
                    else 0.0
                    for pt in pair_types
                ],
                dtype=torch.float32,
                device=device,
            )
            one_hot = get_smoothed_one_hot(labels, config.num_classes, config.label_smooth_eps)

            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_tids,
            )
            cls = outputs.last_hidden_state[:, 0, :]
            model.feature = cls

            density_score = gda.score(cls.detach()) if density_active else None
            _, alpha = model.head(cls, density_score=density_score)

            loss = (
                brier_score_loss(alpha, one_hot, config.num_classes, sample_weights)
                + kl_weight * binary_kl_penalty(alpha, labels, config.num_classes)
                + config.s_reg_weight * s_regularisation_loss(alpha, pair_types, config.s_target)
                + config.ordering_weight * s_ordering_loss(alpha, pair_types, config.ordering_margin)
            )

            if batch_idx % config.ood_expose_every == 0:
                ood_batch, ood_iter = get_ood_batch(ood_iter, ood_train_loader)
                ood_out = model.bert(
                    input_ids=ood_batch["input_ids"].to(device),
                    attention_mask=ood_batch["attention_mask"].to(device),
                    token_type_ids=ood_batch["token_type_ids"].to(device),
                )
                ood_cls = ood_out.last_hidden_state[:, 0, :]
                ood_ds = gda.score(ood_cls.detach()) if density_active else None
                _, ood_alpha = model.head(ood_cls, density_score=ood_ds)
                loss = loss + config.ood_loss_weight * vacuity_loss_fn(ood_alpha, config.num_classes)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

        epoch_time = time.time() - epoch_start
        model.eval()
        va_list, vl_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                alpha = forward_with_density(
                    model,
                    gda,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device),
                    density_active,
                )
                va_list.append(alpha)
                vl_list.append(batch["labels"].to(device))

        val_acc = base_accuracy(torch.cat(va_list), torch.cat(vl_list)).item()
        train_losses.append(total_loss / total_count)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)
        print(
            f"Epoch {epoch}/{config.epochs} | {'Phase B' if density_active else 'Phase A'} | "
            f"KL={kl_weight:.3f} | Loss={train_losses[-1]:.4f} | "
            f"Val={val_acc:.4f} | {epoch_time:.1f}s"
        )

    return TrainArtifacts(
        model=model,
        gda=gda,
        train_loader=train_loader,
        test_loader=test_loader,
        ood_eval_loader=ood_eval_loader,
        density_active=density_active,
        train_losses=train_losses,
        val_accs=val_accs,
        epoch_times=epoch_times,
    )


def evaluate(artifacts: TrainArtifacts, config: DAEDLConfig) -> dict:
    device = config.device
    model = artifacts.model
    gda = artifacts.gda
    density_active = artifacts.density_active

    print("\n--- Full Phase 2 Metric Suite ---")
    model.eval()

    all_alphas_indist, all_labels_indist, all_pair_types_indist = [], [], []
    with torch.no_grad():
        for batch in artifacts.test_loader:
            alpha = forward_with_density(
                model,
                gda,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                density_active,
            )
            all_alphas_indist.append(alpha)
            all_labels_indist.append(batch["labels"].to(device))
            all_pair_types_indist.extend(batch["pair_types"])

    alpha_indist = torch.cat(all_alphas_indist)
    labels_indist = torch.cat(all_labels_indist)

    all_alphas_ood, all_pair_types_ood = [], []
    with torch.no_grad():
        for batch in artifacts.ood_eval_loader:
            alpha = forward_with_density(
                model,
                gda,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                density_active,
            )
            all_alphas_ood.append(alpha)
            all_pair_types_ood.extend(batch["pair_types"])
    alpha_ood = torch.cat(all_alphas_ood)

    print("\nRunning MC Dropout epistemic variance diagnostic...")
    mc_var = mc_dropout_epistemic_variance(
        model,
        artifacts.test_loader,
        device,
        config.mc_dropout_samples,
    )

    clean_acc = base_accuracy(alpha_indist, labels_indist).item()
    clean_brier = brier_score_loss(alpha_indist, labels_indist, config.num_classes).item()
    clean_ece = ece_from_probs(get_expected_probs(alpha_indist), labels_indist).item()
    clean_eru = eru_mutual_information(alpha_indist).mean().item()
    mean_s_clean = alpha_indist.sum(dim=1).mean().item()
    mean_s_ood = alpha_ood.sum(dim=1).mean().item()

    print("\n1. Predictive Performance")
    print(f"   Accuracy : {clean_acc:.4f}")
    print(f"   Brier    : {clean_brier:.4f}")
    print(f"   ECE      : {clean_ece:.4f}")
    print("\n2. Reducible Uncertainty")
    print(f"   ERU/MI (clean):   {clean_eru:.4f}")
    print(f"   MC Dropout S-var: {mc_var:.4f}")
    print(f"   Mean S (clean):   {mean_s_clean:.2f}")
    print(f"   Mean S (OOD):     {mean_s_ood:.2f} (n={len(all_pair_types_ood)})")

    all_alphas_comb = torch.cat([alpha_indist, alpha_ood])
    all_pair_types_comb = all_pair_types_indist + all_pair_types_ood
    uncertainty_profiles = profile_uncertainty(all_alphas_comb, all_pair_types_comb)

    pos_s = (uncertainty_profiles.get("positive") or {}).get("mean_S", 0.0)
    hard_s = (uncertainty_profiles.get("hard_negative") or {}).get("mean_S", 0.0)
    ood_s = (uncertainty_profiles.get("ood") or {}).get("mean_S", 0.0)
    routing_margin = pos_s - hard_s
    s_ratio = ood_s / hard_s if hard_s > 0 else float("nan")

    results = {
        "daedl_active": density_active,
        "gda_pca_components": config.gda_pca_components if density_active else None,
        "accuracy": round(clean_acc, 4),
        "brier": round(clean_brier, 4),
        "ece": round(clean_ece, 4),
        "eru_mi": round(clean_eru, 4),
        "mc_dropout_S_variance": round(mc_var, 4),
        "mean_S_clean": round(mean_s_clean, 4),
        "mean_S_ood": round(mean_s_ood, 4),
        "ood_eval_n": len(all_pair_types_ood),
        "routing_margin": round(routing_margin, 4),
        "s_ratio": round(s_ratio, 4) if not np.isnan(s_ratio) else None,
        "uncertainty_profiles": uncertainty_profiles,
        "train_losses": artifacts.train_losses,
        "val_accs": artifacts.val_accs,
        "epoch_times": artifacts.epoch_times,
    }

    out_path = config.reports_dir / "daedl_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved -> {out_path}")
    return results


def orchestrate(config: DAEDLConfig | None = None, *, run_train: bool = True, run_eval: bool = True):
    config = config or DAEDLConfig()
    artifacts = train(config) if run_train else None
    results = evaluate(artifacts, config) if run_eval and artifacts is not None else None
    return {"config": config, "artifacts": artifacts, "results": results}
