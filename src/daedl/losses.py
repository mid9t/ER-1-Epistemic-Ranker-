from __future__ import annotations

import torch
import torch.nn.functional as F


def get_expected_probs(alpha):
    return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)


def get_smoothed_one_hot(labels, num_classes=2, epsilon=0.02):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return (1 - epsilon) * one_hot + (epsilon / num_classes)


def brier_score_loss(alpha, target_or_onehot, num_classes=2, sample_weights=None):
    p_hat = get_expected_probs(alpha)
    if target_or_onehot.dim() == 1:
        one_hot = torch.zeros(target_or_onehot.size(0), num_classes, device=target_or_onehot.device)
        one_hot.scatter_(1, target_or_onehot.unsqueeze(1), 1.0)
    else:
        one_hot = target_or_onehot
    per_sample = torch.sum((p_hat - one_hot) ** 2, dim=1)
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return per_sample.mean()


def binary_kl_penalty(alpha, labels, num_classes=2, eps=1e-12):
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    wrong_mask = 1.0 - one_hot
    beta = one_hot + wrong_mask * alpha
    s_beta = beta.sum(dim=1, keepdim=True).clamp_min(eps)
    term1 = torch.lgamma(s_beta) - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
    term2 = -torch.sum(torch.lgamma(beta.clamp_min(eps)), dim=1, keepdim=True)
    term3 = torch.sum(
        (beta - 1.0) * (torch.digamma(beta.clamp_min(eps)) - torch.digamma(s_beta)),
        dim=1,
        keepdim=True,
    )
    return (term1 + term2 + term3).mean()


def s_regularisation_loss(alpha, pair_types, s_target=10.0):
    easy_mask = torch.tensor(
        [pt == "easy_negative" for pt in pair_types],
        dtype=torch.bool,
        device=alpha.device,
    )
    if not easy_mask.any():
        return torch.tensor(0.0, device=alpha.device)
    return F.relu(alpha[easy_mask].sum(dim=1) - s_target).pow(2).mean()


def s_ordering_loss(alpha, pair_types, margin=2.0):
    total_strength = alpha.sum(dim=1)
    pos_mask = torch.tensor(
        [pt == "positive" for pt in pair_types],
        dtype=torch.bool,
        device=alpha.device,
    )
    hard_mask = torch.tensor(
        [pt == "hard_negative" for pt in pair_types],
        dtype=torch.bool,
        device=alpha.device,
    )
    if not pos_mask.any() or not hard_mask.any():
        return torch.tensor(0.0, device=alpha.device)
    return F.relu(total_strength[hard_mask].mean() + margin - total_strength[pos_mask].mean())


def vacuity_loss_fn(ood_alpha, num_classes=2):
    return (ood_alpha.sum(dim=1) - num_classes).pow(2).mean()


def get_kl_weight(epoch, total_epochs, max_weight=0.4, start_epoch=3):
    if epoch < start_epoch:
        return 0.0
    return max_weight * (epoch - start_epoch + 1) / max(total_epochs - start_epoch + 1, 1)

