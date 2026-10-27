"""Dataset dispatch for the training entry point."""

from __future__ import annotations

from bert_daedl.data.banking77_oos import Banking77OOSConfig, build_banking77_dataloaders
from bert_daedl.data.clinc150 import CLINC150Config, build_clinc150_dataloaders

DATASETS = ("clinc150", "banking77_oos")


def build_dataloaders(name: str, cfg, data_dir: str | None = None):
    """Build (train, val, test, oos_val, oos_test, meta) for ``name``.

    ``cfg`` is an ``ExperimentConfig``. ``data_dir`` overrides the default
    on-disk location when it is not None.
    """
    if data_dir is None:
        data_dir = getattr(cfg, "data_dir", None)
    if name == "clinc150":
        data_cfg = CLINC150Config(
            model_name=cfg.model_name,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            eval_batch_size=cfg.eval_batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
        if data_dir:
            data_cfg.local_data_dir = data_dir
        return build_clinc150_dataloaders(data_cfg)
    if name == "banking77_oos":
        data_cfg = Banking77OOSConfig(
            model_name=cfg.model_name,
            max_length=cfg.max_length,
            batch_size=cfg.batch_size,
            eval_batch_size=cfg.eval_batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            data_dir=data_dir or "data/BANKING77-OOS",
        )
        return build_banking77_dataloaders(data_cfg)
    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
