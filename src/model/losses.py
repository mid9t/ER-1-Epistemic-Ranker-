def base_accuracy(alpha, target):
    p_mean = get_expected_probs(alpha) #p_mean shape: (128, 4)
    preds = torch.argmax(p_mean, dim=1) # preds shape: (128,) with values in [0, 1, 2, 3]
    return (preds == target).float().mean()

def get_smoothed_one_hot(labels, num_classes=4, epsilon=0.05):
    """
    Create smoothed one-hot targets for label smoothing.
    (1-ε) * hard_onehot + ε/num_classes
    Introduces controlled aleatoric uncertainty → prevents Dirichlet collapse,
    improves calibration, and gives P-Disp proxy signal.
    """
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    smoothed = (1 - epsilon) * one_hot + (epsilon / num_classes)
    return smoothed

def mixup_embeddings(cls_features, one_hot, alpha=0.2):
    """Proper Mixup for frozen BERT: mix continuous CLS embeddings + labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = cls_features.size(0)
    index = torch.randperm(batch_size).to(cls_features.device)
    
    mixed_cls = lam * cls_features + (1 - lam) * cls_features[index]
    mixed_onehot = lam * one_hot + (1 - lam) * one_hot[index]
    return mixed_cls, mixed_onehot, lam

def get_ood_batch(ood_iter, ood_loader):
    """Safe iterator with auto-reset that properly returns the fresh iterator."""
    try:
        batch = next(ood_iter)
    except StopIteration:
        ood_iter = iter(ood_loader) 
        batch = next(ood_iter)
    return batch, ood_iter

def brier_score_loss(alpha, target_or_onehot, num_classes=4):
    """
    Brier Score on Dirichlet mean.
    Now accepts either:
      - integer labels (dim=1) → creates hard one-hot internally
      - pre-smoothed one-hot tensor (dim=2)
    """
    p_hat = get_expected_probs(alpha)
    if target_or_onehot.dim() == 1:  # legacy integer labels
        one_hot = torch.zeros(target_or_onehot.size(0), num_classes, device=target_or_onehot.device)
        one_hot.scatter_(1, target_or_onehot.unsqueeze(1), 1.0)
    else:
        one_hot = target_or_onehot  # already smoothed
    return torch.mean(torch.sum((p_hat - one_hot) ** 2, dim=1))

def _ece_from_probs(probs, target, n_bins=15):
    conf, preds = probs.max(dim=1)
    acc = preds.eq(target).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == n_bins - 1 else (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - acc[mask].mean()).abs()
    return ece.squeeze(0)

def eru_mutual_information(alpha, eps=1e-12):
    S = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    p_mean = alpha / S
    predictive_entropy = -torch.sum(p_mean * torch.log(p_mean.clamp_min(eps)), dim=1)
    expected_entropy = -torch.sum((alpha / S) * (torch.digamma(alpha + 1) - torch.digamma(S + 1)), dim=1)
    return predictive_entropy - expected_entropy

def p_disc_auroc(alpha_clean, alpha_ood):
    try:
        from torchmetrics.classification import BinaryAUROC
    except ImportError:
        raise ImportError("Please install torchmetrics for AUROC calculation")
        
    # Replace the marginal KL calculation with your ERU Mutual Information
    scores_clean = eru_mutual_information(alpha_clean)
    scores_ood = eru_mutual_information(alpha_ood)
    
    scores = torch.cat([scores_clean, scores_ood], dim=0)
    labels = torch.cat([torch.zeros_like(scores_clean, dtype=torch.long), 
                        torch.ones_like(scores_ood, dtype=torch.long)], dim=0)
    
    metric = BinaryAUROC().to(scores.device)
    return metric(scores, labels)

def p_disp_cohens_d(alpha, is_flipped):
    """Fixed: uses full test set, simplified signature"""
    ent = -torch.sum(get_expected_probs(alpha) * torch.log(get_expected_probs(alpha).clamp_min(1e-12)), dim=1)
    
    flipped_mask = is_flipped.bool()
    ent_flipped = ent[flipped_mask] if flipped_mask.any() else ent
    ent_clean_only = ent[~flipped_mask] if (~flipped_mask).any() else ent
    
    if len(ent_flipped) < 2 or len(ent_clean_only) < 2:
        return 0.0
    
    v1, v2 = ent_flipped.var(unbiased=False), ent_clean_only.var(unbiased=False)
    n1, n2 = ent_flipped.numel(), ent_clean_only.numel()
    pooled_std = torch.sqrt(((n1 * v1) + (n2 * v2)) / (n1 + n2) + 1e-12)
    return (ent_flipped.mean() - ent_clean_only.mean()) / pooled_std