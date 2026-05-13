from .config import DAEDLConfig, default_device, detect_repo_root
from .data import (
    EpicRankerQuadDataset,
    OodFlatDataset,
    collate_flat,
    collate_quads,
    ensure_phase2_inputs,
    materialize_local_processed,
)
from .density import GDADensityModel
from .losses import (
    binary_kl_penalty,
    brier_score_loss,
    get_expected_probs,
    get_kl_weight,
    get_smoothed_one_hot,
    s_ordering_loss,
    s_regularisation_loss,
    vacuity_loss_fn,
)
from .metrics import (
    base_accuracy,
    dirichlet_dissonance,
    ece_from_probs,
    eru_mutual_information,
    evaluate_phase3_training_criteria,
    mc_dropout_epistemic_variance,
    prediction_margin,
    profile_uncertainty,
)
from .model import BertWithEvidentialHead, EvidentialHead
from .training import TrainArtifacts, evaluate, orchestrate, train

__all__ = [
    "DAEDLConfig",
    "TrainArtifacts",
    "BertWithEvidentialHead",
    "EpicRankerQuadDataset",
    "EvidentialHead",
    "GDADensityModel",
    "OodFlatDataset",
    "base_accuracy",
    "binary_kl_penalty",
    "brier_score_loss",
    "collate_flat",
    "collate_quads",
    "default_device",
    "detect_repo_root",
    "dirichlet_dissonance",
    "ece_from_probs",
    "ensure_phase2_inputs",
    "eru_mutual_information",
    "evaluate",
    "evaluate_phase3_training_criteria",
    "get_expected_probs",
    "get_kl_weight",
    "get_smoothed_one_hot",
    "materialize_local_processed",
    "mc_dropout_epistemic_variance",
    "orchestrate",
    "prediction_margin",
    "profile_uncertainty",
    "s_ordering_loss",
    "s_regularisation_loss",
    "train",
    "vacuity_loss_fn",
]

