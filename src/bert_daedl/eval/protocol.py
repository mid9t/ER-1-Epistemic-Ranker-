"""Single evaluation entrypoint: one forward pass per split → results dict."""

from __future__ import annotations

import numpy as np
import torch

from bert_daedl.density.gda import scale_alpha
from bert_daedl.eval.metrics import ece, misclf_metrics, ood_metrics
from bert_daedl.losses.edl import logits_to_alpha
from bert_daedl.uncertainty.scores import (
    scores_from_alpha,
    scores_from_density,
    scores_from_logits,
)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    L, F, Y = [], [], []
    for b in loader:
        ids = b["input_ids"].to(device)
        am = b["attention_mask"].to(device)
        tt = b.get("token_type_ids")
        tt = tt.to(device) if tt is not None else None
        logits, feats = model(ids, am, tt, return_features=True)
        L.append(logits.cpu())
        F.append(feats.cpu())
        Y.append(b["labels"])
    return torch.cat(L), torch.cat(F), torch.cat(Y)


def evaluate_arm(model, loaders, cfg, gda=None, normalizer=None) -> dict:
    """Evaluate on val/test/oos_val/oos_test. Train loader is ignored here."""
    dev = cfg.device
    _, va, te, ova, ote = loaders[:5]
    Lv, Fv, Yv = collect(model, va, dev)
    Lt, Ft, Yt = collect(model, te, dev)
    Lov, Fov, _ = collect(model, ova, dev)
    Lot, Fot, _ = collect(model, ote, dev)

    def score_split(logits, feats):
        s = scores_from_logits(logits)
        if cfg.arm in ("edl", "daedl"):
            alpha = logits_to_alpha(logits, cfg.logit_clamp, cfg.alpha_eps)
            s |= scores_from_alpha(alpha)
        if gda is not None:
            lp = gda.marginal_log_prob(feats.numpy())
            s |= scores_from_density(lp)
            if cfg.arm == "daedl" and normalizer is not None:
                phat = torch.from_numpy(np.asarray(normalizer(lp), dtype=np.float64))
                a_s = scale_alpha(
                    logits, phat, cfg.combine_mode, cfg.lam, cfg.logit_clamp
                )
                s |= scores_from_alpha(a_s, prefix="daedl_")
        return {
            k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in s.items()
        }

    S_val = score_split(Lv, Fv)
    S_test = score_split(Lt, Ft)
    S_otest = score_split(Lot, Fot)

    pred = Lt.argmax(-1)
    correct = (pred == Yt).numpy()
    conf = Lt.softmax(-1).max(-1).values.numpy()

    if getattr(cfg, "primary_score", None):
        primary = cfg.primary_score
    elif cfg.arm == "daedl":
        primary = "daedl_vacuity"
    elif cfg.arm == "edl":
        primary = "vacuity"
    else:
        primary = "msp"

    res = {
        "in_scope_acc": float(correct.mean()),
        "ece": ece(conf, correct),
        "per_score": {},
    }
    for name in S_test:
        res["per_score"][name] = {
            **ood_metrics(S_test[name], S_otest[name]),
            **misclf_metrics(S_test[name], correct),
        }

    # Larson-style thresholded operating point; threshold from VAL only.
    thr = float(np.quantile(S_val[primary], 0.95))  # accept 95% of ID val
    res["thresholded"] = {
        "score": primary,
        "threshold": thr,
        "oos_recall": float((S_otest[primary] > thr).mean()),
        "id_reject_rate": float((S_test[primary] > thr).mean()),
    }
    return res
