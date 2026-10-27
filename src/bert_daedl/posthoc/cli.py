"""Arguments shared by the post-hoc scripts."""

from __future__ import annotations


def add_common_args(parser, *, density: bool = True) -> None:
    """Add ``--oos_subset`` and ``--pca_dim``.

    ``density=True`` also adds ``--feature``. Pass ``density=False`` when the
    script already defines ``--feature`` (the coupling grid, complementarity,
    the GDA ablation, and the bootstrap).
    """
    parser.add_argument(
        "--oos_subset",
        default="all",
        choices=("all", "id", "ood"),
        help="OOS utterances to score: all (pooled), id (ID-OOS), or ood (OOD-OOS).",
    )
    parser.add_argument(
        "--pca_dim",
        default=None,
        help="Override the density PCA dimension. 'none' fits the full vector. "
        "Omit to keep --variant or the saved run config.",
    )
    if density:
        parser.add_argument("--feature", default="feats", choices=("feats", "pooled"))


def density_kwargs(args) -> dict:
    """Keyword arguments for ``logp_from_run`` taken from a parsed CLI."""
    from bert_daedl.posthoc.scoring import PCA_UNSET, pca_from_arg

    kw = {}
    feature = getattr(args, "feature", None)
    if feature:
        kw["feature"] = feature
    variant = getattr(args, "variant", None)
    if variant:
        kw["variant"] = variant
    if hasattr(args, "pca_dim"):
        dim = pca_from_arg(args.pca_dim)
        if dim is not PCA_UNSET:
            kw["pca_dim"] = dim
    return kw
