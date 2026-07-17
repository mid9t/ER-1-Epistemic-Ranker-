"""Training loop shared by CE / EDL / DAEDL arms."""

from __future__ import annotations

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from bert_daedl.data.clinc150 import labels_to_one_hot
from bert_daedl.losses.edl import edl_loss, logits_to_alpha
from bert_daedl.uncertainty.scores import scores_from_alpha, scores_from_logits


def build_optimizer(model, cfg):
    enc, head = [], []
    for n, p in model.named_parameters():
        (enc if n.startswith("encoder") else head).append(p)
    return AdamW(
        [{"params": enc, "lr": cfg.lr_encoder}, {"params": head, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )


@torch.no_grad()
def quick_val(model, val_loader, oos_val_loader, cfg):
    """ID-val accuracy + uncertainty gap (oos − id). Negative gap ⇒ broken signal."""
    model.eval()
    dev = cfg.device
    correct, total = 0, 0
    id_unc, oos_unc = [], []

    def _unc_from_logits(logits):
        if cfg.arm == "ce":
            return scores_from_logits(logits)["entropy"]
        alpha = logits_to_alpha(logits, cfg.logit_clamp, cfg.alpha_eps)
        return scores_from_alpha(alpha)["vacuity"]

    for b in val_loader:
        ids = b["input_ids"].to(dev)
        am = b["attention_mask"].to(dev)
        tt = b.get("token_type_ids")
        tt = tt.to(dev) if tt is not None else None
        y = b["labels"].to(dev)
        logits = model(ids, am, tt)
        pred = logits.argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        id_unc.append(_unc_from_logits(logits).cpu())

    for b in oos_val_loader:
        ids = b["input_ids"].to(dev)
        am = b["attention_mask"].to(dev)
        tt = b.get("token_type_ids")
        tt = tt.to(dev) if tt is not None else None
        logits = model(ids, am, tt)
        oos_unc.append(_unc_from_logits(logits).cpu())

    acc = correct / max(total, 1)
    gap = torch.cat(oos_unc).mean().item() - torch.cat(id_unc).mean().item()
    return acc, gap


def train_model(model, train_loader, val_loader, oos_val_loader, cfg, num_classes):
    dev = cfg.device
    model.to(dev)
    opt = build_optimizer(model, cfg)
    total = len(train_loader) * cfg.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(cfg.warmup_frac * total), max(total, 1)
    )
    ce = torch.nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for step, b in enumerate(train_loader):
            ids = b["input_ids"].to(dev)
            am = b["attention_mask"].to(dev)
            tt = b.get("token_type_ids")
            tt = tt.to(dev) if tt is not None else None
            y = b["labels"].to(dev)
            logits = model(ids, am, tt)
            if cfg.arm == "ce":
                loss = ce(logits, y)
            else:
                loss = edl_loss(
                    logits, labels_to_one_hot(y, num_classes), epoch, cfg
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += loss.item()
            n_batches += 1
            if (step + 1) % 50 == 0:
                print(
                    f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"loss={running / n_batches:.4f}",
                    flush=True,
                )

        acc, vac_gap = quick_val(model, val_loader, oos_val_loader, cfg)
        print(
            f"epoch {epoch}: val_acc={acc:.4f} vacuity_gap(oos-id)={vac_gap:+.3f}",
            flush=True,
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc
