import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed
from datasets import load_dataset
import time

# ==========================================
# 1. Environment Setup
# ==========================================
def setup_environment():
    set_seed(42) # Sets seed for transformers, torch, numpy, and random
    os.makedirs("data/", exist_ok=True)
    os.makedirs("reports/phase1/", exist_ok=True)
    print("Environment setup complete. Seed=42.")

# M2/MPS detection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")   # ← add this so you can verify

# ==========================================
# 2. Data Pipeline
# ==========================================
def prepare_datasets():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # Load AG News
    ag_news = load_dataset("ag_news", cache_dir="data/")
    
    # Split train into train (114k) and val (6k)
    train_val_split = ag_news["train"].train_test_split(test_size=6000, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    clean_test = ag_news["test"]

    # Create Noisy Test (20% flipped)
    def make_noisy_test(clean_test, noise_rate = 0.2, seed=42):
        rng = random.Random(seed)
        n = len(clean_test)
        k = int(n*noise_rate)
        flip_index = rng.sample(range(n), k)

        def inject_noise(example, idx):
            # Deterministic flip for 20% of the data
            is_flipped = idx in flip_index
            if is_flipped:
                original_label = example["label"]
                possible_flips = [l for l in range(4) if l != original_label]
                example["label"] = random.choice(possible_flips)
            example["is_flipped"] = is_flipped
            return example
        return clean_test.map(inject_noise, with_indices=True)

    noisy_test = make_noisy_test(clean_test, noise_rate=0.2, seed=42)

    # Create OOD Test (IMDB)
    imdb = load_dataset("imdb", split="test", cache_dir="data/")
    ood_test = imdb.select(range(3000))
    # Add dummy labels to IMDB to avoid DataLoader key errors
    ood_test = ood_test.map(lambda x: {"label": 0}) 

    # Tokenize all
    encoded_train = train_data.map(tokenize_func, batched=True).with_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_val = val_data.map(tokenize_func, batched=True).with_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_clean_test = clean_test.map(tokenize_func, batched=True).with_format("torch", columns=["input_ids", "attention_mask", "label"])
    encoded_noisy_test = noisy_test.map(tokenize_func, batched=True).with_format("torch", columns=["input_ids", "attention_mask", "label", "is_flipped"])
    encoded_ood_test = ood_test.map(tokenize_func, batched=True).with_format("torch", columns=["input_ids", "attention_mask", "label"])

    return encoded_train, encoded_val, encoded_clean_test, encoded_noisy_test, encoded_ood_test

# ==========================================
# 3. Model & Loss
# ==========================================
class EvidentialHead(nn.Module):
    def __init__(self, input_dim=768, num_classes=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

        # Initialize temperature at 1.0 (no scaling)
        self.register_buffer('temperature', torch.ones(1))

    def set_temperature(self, temp: float):
        if temp <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature.fill_(float(temp))

    def forward(self, features, temp=None):
        # Use provided temp for calibration, otherwise use stored value
        t = temp if temp is not None else self.temperature
        raw_logits = self.linear(features)
        # Apply temperature scaling at the logit level
        scaled_logits = raw_logits / t
        # Generate evidence and alpha parameters
        evidence = F.softplus(scaled_logits)           # e >= 0
        alpha = evidence + 1.0                         # alpha >= 1
        return evidence, alpha

class BertWithEvidentialHead(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()
        
        self.head = EvidentialHead(768, num_classes)

    def forward(self, input_ids=None, attention_mask=None, temp=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        
        # Pass the temperature to the head
        return self.head(cls, temp=temp)

# v3.1 - Add label smoothing to Brier loss to prevent overconfidence on noisy labels, which can destabilize training and hurt P-Disp.
def get_expected_probs(alpha):
    S = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return alpha / S

# V3.0 Replace your current kl_divergence_penalty with this refined version
def kl_divergence_penalty(alpha, one_hot, num_classes=4, eps=1e-12):
    """
    Refined KL: ONLY penalize evidence on the 3 incorrect classes.
    Correct-class evidence is fixed at 1 (no penalty).
    This keeps high confidence on clean data while forcing vacuity on OOD.
    """
    wrong_mask = 1.0 - one_hot  # 1 on incorrect classes, 0 on correct
    beta = one_hot + wrong_mask * alpha   # correct class β=1, wrong classes get α penalized
    S_beta = torch.sum(beta, dim=1, keepdim=True).clamp_min(eps)

    term1 = torch.lgamma(S_beta) - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
    term2 = -torch.sum(torch.lgamma(beta), dim=1, keepdim=True)
    term3 = torch.sum((beta - 1.0) * (torch.digamma(beta) - torch.digamma(S_beta)), dim=1, keepdim=True)

    kl_loss = term1 + term2 + term3
    return kl_loss.mean()

def quick_gradient_check(model, num_classes=4):
    model.train()
    input_ids = torch.randint(0, 30522, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)
    target = torch.randint(0, num_classes, (2,), device=device)

    _, alpha = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = brier_score_loss(alpha, target, num_classes=num_classes)
    assert torch.isfinite(loss).all(), "Loss has NaN/Inf"
    
    model.zero_grad()
    loss.backward()
    head_grads = [p.grad for p in model.head.parameters() if p.requires_grad]
    assert all(g is not None for g in head_grads), "No grads on head"
    assert all(torch.isfinite(g).all() for g in head_grads), "NaN/Inf in head grads"
    print("Gradient check passed.")

# ==========================================
# 4. Metrics Implementation
# ==========================================
def base_accuracy(alpha, target):
    p_mean = get_expected_probs(alpha) #p_mean shape: (128, 4)
    preds = torch.argmax(p_mean, dim=1) # preds shape: (128,)
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

# ==========================================
# 5. Training Loop & Evaluation
# ==========================================
def extract_tensors(batch):
    return batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

def get_all_alphas_and_targets(model, dataloader):
    """

    """
    model.eval()
    all_alphas, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attn_mask, labels = extract_tensors(batch)
            _, alpha = model(input_ids=input_ids, attention_mask=attn_mask)
            all_alphas.append(alpha)
            all_targets.append(labels)
    return torch.cat(all_alphas), torch.cat(all_targets)

def train_and_evaluate():
    setup_environment()
    train_data, val_data, clean_test, noisy_test, ood_test = prepare_datasets()
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=256, shuffle=False, num_workers=0)
    ood_loader   = DataLoader(ood_test,   batch_size=128, shuffle=True, num_workers=0)
    
    ood_iter = iter(ood_loader)

    model = BertWithEvidentialHead(num_classes=4).to(device)
    # Unfreeze only the final transformer block for light backbone adaptation.
    for p in model.bert.encoder.layer[11].parameters():
        p.requires_grad = True

    shared_temp = 1.0
    model.head.set_temperature(shared_temp)
    quick_gradient_check(model)

    layer11_params = list(model.bert.encoder.layer[11].parameters())
    optimizer = optim.AdamW(
        [
            {"params": model.head.parameters(), "lr": 5e-4},
            {"params": layer11_params, "lr": 1e-5},
        ],
        weight_decay=1e-3,
    )
    label_smoothing_eps = 0.02
    epochs = 5
    ood_exposure_every = 3
    train_losses, val_accs, epoch_times = [], [], []

    print("\n--- Starting Training (Clean Ablation: Base Brier + OOD Exposure) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        batch_idx = 0

        epoch_start_time = time.time()
        
        for batch in train_loader:
            batch_idx += 1
            input_ids, attn_mask, labels = extract_tensors(batch)  # shape: input (128, 128), mask (128, 128), labels (128)
            one_hot = get_smoothed_one_hot(labels, num_classes=4, epsilon=label_smoothing_eps) # shape (128, 4)
            
            # === Forward BERT ONCE (Layer 11 is trainable) ===
            outputs = model.bert(input_ids=input_ids, attention_mask=attn_mask) # shape (128, 128, 768)
            cls = outputs.last_hidden_state[:, 0, :] # shape (128, 768)
            
            # === Head only (cheap) - NO MIXUP ===
            # Uses shared model.head.temperature across train/OOD/eval.
            _, alpha = model.head(cls) # shape alpha (128, 4)
            
            # === Base Decision-Theoretic Loss Only ===
            loss = brier_score_loss(alpha, one_hot) 
            
            # === OOD exposure (Fixed control-flow) ===
            if batch_idx % ood_exposure_every == 0:
                ood_batch, ood_iter = get_ood_batch(ood_iter, ood_loader)
                ood_ids, ood_mask, _ = extract_tensors(ood_batch)
                _, ood_alpha = model(input_ids=ood_ids, attention_mask=ood_mask)
                vacuity_loss = 0.1 * ood_alpha.mean()
                loss = loss + vacuity_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)
       
        # Validation
        val_alphas, val_targets = get_all_alphas_and_targets(model, val_loader)
        val_acc = base_accuracy(val_alphas, val_targets).item()
        epoch_time = time.time() - epoch_start_time

        train_losses.append(total_loss / total_count)
        val_accs.append(val_acc)
        epoch_times.append(epoch_time)

    print("\n--- Running Full Metric Suite ---")
    test_loader = lambda ds: DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    alpha_clean, targets_clean = get_all_alphas_and_targets(model, test_loader(clean_test))
    alpha_noisy, _ = get_all_alphas_and_targets(model, test_loader(noisy_test))
    alpha_ood, _   = get_all_alphas_and_targets(model, test_loader(ood_test))

    # Metrics computation
    clean_acc = base_accuracy(alpha_clean, targets_clean)
    clean_brier = brier_score_loss(alpha_clean, targets_clean, num_classes=4)
    clean_ece = _ece_from_probs(get_expected_probs(alpha_clean), targets_clean, n_bins=15)
    clean_eru = eru_mutual_information(alpha_clean).mean()
    auroc = p_disc_auroc(alpha_clean, alpha_ood)
    
    # Safely gather is_flipped flags
    test_loader_noisy = test_loader(noisy_test)
    all_is_flipped = []
    with torch.no_grad():
        for batch in test_loader_noisy:
            if "is_flipped" in batch:
                all_is_flipped.append(batch["is_flipped"].to(device))
    
    is_flipped_tensor = torch.cat(all_is_flipped) if all_is_flipped else None
    
    if is_flipped_tensor is not None and is_flipped_tensor.shape[0] == alpha_noisy.shape[0]:
        cohens_d = p_disp_cohens_d(alpha_noisy, is_flipped_tensor)
    else:
        cohens_d = 0.0
        print("Warning: is_flipped collection failed or shape mismatch → P-Disp set to 0")

    # Document-recommended structured outputs
    print("\n1. Predictive Performance / Calibration")
    print(f"   Accuracy: {clean_acc:.4f}")
    print(f"   Brier:    {clean_brier:.4f}")
    print(f"   ECE:      {clean_ece:.4f}")

    print("\n2. OOD / Epistemic Proxy")
    print(f"   P-Disc (AUROC): {auroc:.4f}")
    print(f"   ERU / MI:       {clean_eru:.4f}")

    print("\n3. Corruption Sensitivity")
    print(f"   P-Disp (d): {cohens_d:.4f}")
    
    print("\n--- Diagnostics ---")
    print(f"Mean Strength S (clean): {alpha_clean.sum(dim=1).mean():.1f}")
    print(f"Mean Strength S (OOD):   {alpha_ood.sum(dim=1).mean():.1f}")

if __name__ == "__main__":
    train_and_evaluate()