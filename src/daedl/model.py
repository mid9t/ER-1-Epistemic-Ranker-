from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class EvidentialHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        dropout_p: float = 0.10,
        use_dropout_in_training: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(input_dim, num_classes)
        self.register_buffer("temperature", torch.ones(1))
        self.use_dropout_in_training = use_dropout_in_training

    def set_temperature(self, temp):
        if temp <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature.fill_(float(temp))

    def forward(self, features, temp=None, density_score=None, mc_dropout_active: bool = False):
        t = temp if temp is not None else self.temperature
        dropped = (
            features
            if self.training and not self.use_dropout_in_training and not mc_dropout_active
            else self.dropout(features)
        )
        logits = self.linear(dropped) / t

        if density_score is not None:
            s = density_score.unsqueeze(1).clamp(min=1e-6)
            logits = logits * s

        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        return evidence, alpha


class BertWithEvidentialHead(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_p: float = 0.10,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()
        self.head = EvidentialHead(768, num_classes, dropout_p=dropout_p)
        self.feature = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        temp=None,
        density_score=None,
        mc_dropout_active: bool = False,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        self.feature = cls
        return self.head(
            cls,
            temp=temp,
            density_score=density_score,
            mc_dropout_active=mc_dropout_active,
        )


def unfreeze_last_bert_layers(model: BertWithEvidentialHead, layer_indices=(9, 10, 11)) -> None:
    for layer_idx in layer_indices:
        for param in model.bert.encoder.layer[layer_idx].parameters():
            param.requires_grad = True
        model.bert.encoder.layer[layer_idx].train()
