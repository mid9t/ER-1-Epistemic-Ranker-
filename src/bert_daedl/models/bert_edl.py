"""BERT encoder + spectral-norm bottleneck + evidential classifier head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from transformers import AutoModel

_ACT = {"none": nn.Identity, "relu": nn.ReLU, "tanh": nn.Tanh}


class BertEvidentialClassifier(nn.Module):
    """Encoder -> pooled -> (SN) bottleneck feature -> logits.

    GDA fits on `features`. `alpha = exp(logits)` (EXP parameterization,
    DAEDL) is computed in the loss / scores, NOT here — forward stays
    loss-agnostic so all three arms share this class.

    Milestone 2 note: pass `output_hidden_states=True` to the encoder and
    attach `exit_heads` per layer; the forward signature already reserves
    the flag so no rework is needed.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        feat_dim: int = 256,
        sn_bottleneck: bool = True,
        bottleneck_activation: str = "none",
        pooling: str = "cls",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        bottleneck = nn.Linear(hidden, feat_dim)
        self.bottleneck = spectral_norm(bottleneck) if sn_bottleneck else bottleneck
        self.act = _ACT[bottleneck_activation]()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def _pool(self, last_hidden, attention_mask):
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        return_features: bool = False,
        return_hidden_states: bool = False,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=return_hidden_states,
        )
        pooled = self._pool(out.last_hidden_state, attention_mask)
        feats = self.act(self.bottleneck(self.dropout(pooled)))
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits
