from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


def detect_repo_root(start: Optional[Path] = None) -> Path:
    here = (start or Path.cwd()).resolve()
    candidates = [here, *here.parents]
    for candidate in candidates:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    return here


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DAEDLConfig:
    project_root: Path = field(default_factory=detect_repo_root)
    batch_size: int = 128
    max_length: int = 128
    epochs: int = 5
    warmup_epochs: int = 2
    hard_neg_start: int = 2
    kl_start_epoch: int = 3

    num_classes: int = 2
    lr_head: float = 5e-4
    lr_layer11: float = 1e-5
    lr_layer10: float = 5e-6
    lr_layer9: float = 1e-6
    weight_decay: float = 1e-3

    label_smooth_eps: float = 0.02
    ood_expose_every: int = 3
    ood_loss_weight: float = 0.5
    kl_max_weight: float = 0.4
    s_reg_weight: float = 0.01
    s_target: float = 10.0
    ordering_weight: float = 0.1
    ordering_margin: float = 2.0

    gda_pca_components: int = 64
    gda_density_temp: float = 0.1
    mc_dropout_samples: int = 20
    mc_dropout_p: float = 0.10

    train_num_workers: int = 0
    train_persistent_workers: bool = False
    eval_num_workers: int = 0

    use_local_processed: bool = False
    local_processed_dir: Path = Path("/tmp/epicranker")
    model_name: str = "bert-base-uncased"
    seed: int = 42
    phase3_routing_margin_target: float = 3.0
    phase3_s_ratio_target: float = 0.40

    pair_weights: dict[str, float] = field(default_factory=lambda: {
        "positive": 4.0,
        "hard_negative": 2.0,
        "easy_negative": 0.5,
    })

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def active_processed_dir(self) -> Path:
        return self.local_processed_dir if self.use_local_processed else self.processed_dir

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports" / "phase2"

    @property
    def ckpt_dir(self) -> Path:
        return self.project_root / "checkpoints"

    @property
    def device(self) -> torch.device:
        return default_device()

