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

    def forward(self, features):
        raw = self.linear(features)
        evidence = F.softplus(raw)           # e >= 0
        alpha = evidence + 1.0               # alpha >= 1
        return evidence, alpha

class BertWithEvidentialHead(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        
        # Freeze backbone
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval() # Ensure dropout is disabled
        
        self.head = EvidentialHead(768, num_classes)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Force BERT to eval mode even if model.train() is called on the wrapper
        self.bert.eval() 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.head(cls) 

def get_expected_probs(alpha):
    S = alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return alpha / S

def brier_score_loss(alpha, target, num_classes=4):
    p_hat = get_expected_probs(alpha)
    one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    return torch.mean(torch.sum((p_hat - one_hot) ** 2, dim=1))

# v-2.0: KL divergence penalty to encourage sharper distributions
def kl_divergence_penalty(alpha, labels, num_classes=4, eps=1e-12):
    """
    KL[Dir(beta) || Dir(1)] where beta = y + (1 - y) * alpha
    labels should be one-hot encoded (batch_size, num_classes)
    """
    labels = labels.float()
    beta = labels + (1 - labels) * alpha
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
    p_mean = get_expected_probs(alpha)
    preds = torch.argmax(p_mean, dim=1)
    return (preds == target).float().mean()

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

def _kl_p_mean_expected(p_mean, expected_p, eps=1e-12):
    if expected_p.dim() == 1:
        expected_p = expected_p.unsqueeze(0)
    expected_p = expected_p / expected_p.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.sum(p_mean.clamp_min(eps) * (torch.log(p_mean.clamp_min(eps)) - torch.log(expected_p.clamp_min(eps))), dim=1)

def p_disc_auroc(alpha_clean, alpha_ood):
    try:
        from torchmetrics.classification import BinaryAUROC
    except ImportError:
        raise ImportError("Please install torchmetrics for AUROC calculation")
        
    p_mean_clean = get_expected_probs(alpha_clean)
    p_mean_ood = get_expected_probs(alpha_ood)
    expected_p = p_mean_clean.mean(dim=0) # Aggregate expected probability
    
    scores_clean = _kl_p_mean_expected(p_mean_clean, expected_p)
    scores_ood = _kl_p_mean_expected(p_mean_ood, expected_p)
    
    scores = torch.cat([scores_clean, scores_ood], dim=0)
    labels = torch.cat([torch.zeros_like(scores_clean, dtype=torch.long), torch.ones_like(scores_ood, dtype=torch.long)], dim=0)
    
    metric = BinaryAUROC().to(scores.device)
    return metric(scores, labels)

def p_disp_cohens_d(alpha_clean, alpha_noisy, eps=1e-12):
    p_mean_clean = get_expected_probs(alpha_clean)
    p_mean_noisy = get_expected_probs(alpha_noisy)
    
    ent_clean = -torch.sum(p_mean_clean * torch.log(p_mean_clean.clamp_min(eps)), dim=1)
    ent_noisy = -torch.sum(p_mean_noisy * torch.log(p_mean_noisy.clamp_min(eps)), dim=1)
    
    v1, v2 = ent_clean.var(unbiased=False), ent_noisy.var(unbiased=False)
    n1, n2 = ent_clean.numel(), ent_noisy.numel()
    pooled_std = torch.sqrt(((n1 * v1) + (n2 * v2)) / (n1 + n2) + eps)
    return (ent_noisy.mean() - ent_clean.mean()) / pooled_std # Noisy should have higher entropy

# ==========================================
# 5. Training Loop & Evaluation
# ==========================================
def extract_tensors(batch):
    return batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

def get_all_alphas_and_targets(model, dataloader):
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
    # Model initialization and environment setup
    model = BertWithEvidentialHead(num_classes=4).to(device)
    setup_environment()
    train_data, val_data, clean_test, noisy_test, ood_test = prepare_datasets()
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    quick_gradient_check(model)
    
    optimizer = optim.AdamW(model.head.parameters(), lr=5e-4)
    epochs = 5
    kl_weight = 1.0
    kl_anneal_epochs = epochs
    
    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        anneal = min(1.0, epoch / max(1, kl_anneal_epochs))
        total_loss, total_count = 0.0, 0
        
        for batch in train_loader:
            input_ids, attn_mask, labels = extract_tensors(batch)
            _, alpha = model(input_ids=input_ids, attention_mask=attn_mask)
            
            #v-1.0 loss = brier_score_loss(alpha, labels, num_classes=4)
            #v-2.0: Add KL penalty with annealing
            num_classes = alpha.size(1)
            brier = brier_score_loss(alpha, labels, num_classes=num_classes)
            one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            kl = kl_divergence_penalty(alpha, one_hot, num_classes=num_classes)
            loss = brier + (anneal * kl_weight * kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {total_loss/total_count:.4f} | AvgTime per Epoch: {total_loss/total_count:.4f} seconds")

    print("\n--- Running Full Metric Suite ---")
    
    # Generate predictions for all test sets
    alpha_clean, targets_clean = get_all_alphas_and_targets(model, DataLoader(clean_test, batch_size=64))
    alpha_noisy, targets_noisy = get_all_alphas_and_targets(model, DataLoader(noisy_test, batch_size=64))
    alpha_ood, _ = get_all_alphas_and_targets(model, DataLoader(ood_test, batch_size=64))

    # Calculate Core Metrics
    clean_acc = base_accuracy(alpha_clean, targets_clean)
    clean_brier = brier_score_loss(alpha_clean, targets_clean, num_classes=4)
    clean_ece = _ece_from_probs(get_expected_probs(alpha_clean), targets_clean, n_bins=15)
    clean_eru = eru_mutual_information(alpha_clean).mean()
    
    auroc = p_disc_auroc(alpha_clean, alpha_ood)
    cohens_d = p_disp_cohens_d(alpha_clean, alpha_noisy)

    print(f"Clean Accuracy:  {clean_acc:.4f} (Target > 0.90)")
    print(f"Brier Score:     {clean_brier:.4f}")
    print(f"ECE (15 bins):   {clean_ece:.4f} (Target < 0.05)")
    print(f"ERU (Mean):      {clean_eru:.4f}")
    print(f"P-Disc (AUROC):  {auroc:.4f} (Target > 0.90)")
    print(f"P-Disp (Cohen's d): {cohens_d:.4f} (Target > 1.5)")

if __name__ == "__main__":
    train_and_evaluate()

    