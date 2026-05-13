from __future__ import annotations

import torch

from .losses import get_expected_probs


def base_accuracy(alpha, target):
    return (torch.argmax(get_expected_probs(alpha), dim=1) == target).float().mean()


def eru_mutual_information(alpha, eps=1e-12):
    total_strength = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    p_mean = alpha / total_strength
    pred_ent = -torch.sum(p_mean * torch.log(p_mean.clamp_min(eps)), dim=1)
    exp_ent = -torch.sum(
        (alpha / total_strength)
        * (torch.digamma(alpha + 1) - torch.digamma(total_strength + 1)),
        dim=1,
    )
    return pred_ent - exp_ent


def dirichlet_dissonance(alpha, eps=1e-12):
    total_strength = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    belief = ((alpha - 1.0) / total_strength).clamp_min(0.0)
    num_classes = alpha.shape[1]
    diss = torch.zeros(alpha.shape[0], device=alpha.device)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            denom = (belief[:, i] + belief[:, j]).clamp_min(eps)
            balance = 1.0 - (belief[:, i] - belief[:, j]).abs() / denom
            diss += belief[:, i] * belief[:, j] * balance
    return diss


def prediction_margin(alpha):
    probs = get_expected_probs(alpha)
    top2 = probs.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def ece_from_probs(probs, target, n_bins=15):
    conf, preds = probs.max(dim=1)
    acc = preds.eq(target).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (conf >= lo) & (conf <= hi if idx == n_bins - 1 else conf < hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - acc[mask].mean()).abs()
    return ece.squeeze(0)


@torch.no_grad()
def mc_dropout_epistemic_variance(model, loader, device, n_samples=20):
    model.train()
    all_vars = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tids = batch["token_type_ids"].to(device)
        samples = []
        for _ in range(n_samples):
            _, alpha = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=tids,
                mc_dropout_active=True,
            )
            samples.append(alpha.sum(dim=1))
        all_vars.append(torch.stack(samples, dim=0).var(dim=0))
    model.eval()
    return torch.cat(all_vars).mean().item()


def profile_uncertainty(all_alphas, all_pair_types):
    keys = ("positive", "hard_negative", "easy_negative", "ood")
    profiles = {key: {"S": [], "dissonance": [], "margin": []} for key in keys}
    total_strength = all_alphas.sum(dim=1)
    diss_all = dirichlet_dissonance(all_alphas)
    marg_all = prediction_margin(all_alphas)

    for idx, pair_type in enumerate(all_pair_types):
        if pair_type in profiles:
            profiles[pair_type]["S"].append(total_strength[idx].item())
            profiles[pair_type]["dissonance"].append(diss_all[idx].item())
            profiles[pair_type]["margin"].append(marg_all[idx].item())

    print("\n--- Phase 2 Uncertainty Profiles ---")
    print(f"  {'Slice':>15}  {'Mean S':>8}  {'Dissonance':>10}  {'Margin':>8}  {'n':>6}")
    print(f"  {'-' * 15}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 6}")

    results = {}
    for pair_type, vals in profiles.items():
        if vals["S"]:
            n = len(vals["S"])
            mean_s = sum(vals["S"]) / n
            mean_d = sum(vals["dissonance"]) / n
            mean_m = sum(vals["margin"]) / n
            results[pair_type] = {
                "mean_S": round(mean_s, 4),
                "mean_dissonance": round(mean_d, 4),
                "mean_margin": round(mean_m, 4),
                "n": n,
            }
            print(
                f"  {pair_type.capitalize():>15}  {mean_s:>8.4f}  "
                f"{mean_d:>10.4f}  {mean_m:>8.4f}  {n:>6,}"
            )
        else:
            results[pair_type] = None
            print(f"  {pair_type.capitalize():>15}  {'no samples':>28}")
    return results


def evaluate_phase3_training_criteria(
    results: dict,
    routing_margin_target: float = 3.0,
    s_ratio_target: float = 0.40,
) -> dict:
    profiles = results.get("uncertainty_profiles") or {}
    pos_s = (profiles.get("positive") or {}).get("mean_S")
    hard_s = (profiles.get("hard_negative") or {}).get("mean_S")
    ood_s = (profiles.get("ood") or {}).get("mean_S")

    routing_margin = results.get("routing_margin")
    if routing_margin is None and pos_s is not None and hard_s is not None:
        routing_margin = float(pos_s - hard_s)

    s_ratio = results.get("s_ratio")
    if s_ratio is None and ood_s is not None and hard_s not in (None, 0):
        s_ratio = float(ood_s / hard_s)

    checks = {
        "density_active": bool(results.get("daedl_active")),
        "routing_margin": routing_margin is not None and routing_margin > routing_margin_target,
        "s_ratio": s_ratio is not None and s_ratio < s_ratio_target,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "routing_margin": routing_margin,
        "routing_margin_target": routing_margin_target,
        "s_ratio": s_ratio,
        "s_ratio_target": s_ratio_target,
    }
