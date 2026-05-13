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