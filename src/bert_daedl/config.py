from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GDAConfig:
    cov_mode: str = "tied"  # "tied" | "per_class"
    pca_dim: Optional[int] = 128  # None = no reduction
    whiten: bool = True
    jitter_grid: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
    min_var: float = 1e-8


@dataclass
class ExperimentConfig:
    # data
    model_name: str = "bert-base-uncased"
    max_length: int = 64
    batch_size: int = 32
    eval_batch_size: int = 128
    num_workers: int = 4  # set 0 on MPS
    # model
    feat_dim: int = 256
    sn_bottleneck: bool = True
    bottleneck_activation: str = "none"  # "none" | "relu" | "tanh"
    pooling: str = "cls"  # "cls" | "mean"
    dropout: float = 0.1
    # training
    arm: str = "edl"  # "ce" | "edl" | "daedl"
    epochs: int = 5
    lr_encoder: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    # EDL — mirror validated DAEDL/ CIFAR run (EXP param + annealed KL)
    reg_param: float = 1e-3
    kl_anneal_epochs: int = 5  # = epochs
    alpha_eps: float = 1e-6
    logit_clamp: float = 15.0
    # density
    gda: GDAConfig = field(default_factory=GDAConfig)
    normalizer: str = "qsigmoid"  # "minmax" | "qsigmoid" | "ecdf"
    combine_mode: str = "mul"  # "mul" | "add_log"
    lam: float = 5.0  # add_log strength
    # bookkeeping
    seed: int = 0
    device: str = "cuda"
    output_dir: str = "results"
    primary_score: Optional[str] = None
