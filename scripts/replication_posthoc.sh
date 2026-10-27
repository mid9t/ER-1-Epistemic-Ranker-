#!/bin/bash
# Post-hoc pipeline for one replication root. Bash 3.2 (macOS) compatible.
# Usage: scripts/replication_posthoc.sh ROOT [all|id|ood ...]
# The pooled subset is always run before id or ood, because selection files
# are written only for --oos_subset all.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

if [ "$#" -lt 1 ]; then
  echo "usage: scripts/replication_posthoc.sh ROOT [all|id|ood ...]" >&2
  exit 2
fi
ROOT="$1"
shift
if [ "$#" -eq 0 ]; then
  set -- all
fi

HAS_ALL=0
HAS_SPLIT=0
REST=""
for subset in "$@"; do
  if [ "$subset" = "all" ]; then
    HAS_ALL=1
  else
    REST="$REST $subset"
    if [ "$subset" = "id" ] || [ "$subset" = "ood" ]; then
      HAS_SPLIT=1
    fi
  fi
done
if [ "$HAS_ALL" -eq 1 ]; then
  ORDERED="all$REST"
else
  ORDERED="$REST"
fi

# Pre-registered λ grid. coupling_grid.py keeps the shorter CLINC150 default.
LAMS="1 2 5 10 20 40 80 160"

run_density() {
  subset="$1"
  density="$2"
  if [ "$density" = "port" ]; then
    FEATURE=feats
    VARIANT=config
    PCA=128
  else
    FEATURE=pooled
    VARIANT=tied
    PCA=none
  fi
  OUT="$ROOT/posthoc/$subset"
  "$PY" scripts/score_table.py --root "$ROOT" --oos_subset "$subset" \
    --feature "$FEATURE" --pca_dim "$PCA" --verify \
    --out "$OUT/scores_$density"
  # shellcheck disable=SC2086
  "$PY" scripts/coupling_grid.py --root "$ROOT" --oos_subset "$subset" \
    --feature "$FEATURE" --variant "$VARIANT" --pca_dim "$PCA" \
    --lams $LAMS --extend-edge --out "$OUT/grid_$density"
  "$PY" scripts/complementarity.py --root "$ROOT" --oos_subset "$subset" \
    --feature "$FEATURE" --variant "$VARIANT" --pca_dim "$PCA" \
    --lam_max 160 --out "$OUT/complementarity_$density"
  SEL="$ROOT/posthoc/all/grid_${density}/selected.json"
  if [ ! -f "$SEL" ]; then
    echo "missing $SEL (run the pooled subset before $subset)" >&2
    exit 1
  fi
  "$PY" scripts/paired_bootstrap.py --preset replication --root "$ROOT" \
    --oos_subset "$subset" --feature "$FEATURE" --variant "$VARIANT" \
    --pca_dim "$PCA" --selected "$SEL" \
    --out "$OUT/bootstrap_${density}.csv" --B 5000
}

for subset in $ORDERED; do
  run_density "$subset" port
  run_density "$subset" full
done

if [ "$HAS_SPLIT" -eq 1 ]; then
  "$PY" scripts/paired_bootstrap.py --did --root "$ROOT" \
    --feature pooled --variant tied --pca_dim none \
    --selected-secondary "$ROOT/posthoc/all/grid_full/selected_secondary.json" \
    --out "$ROOT/posthoc/h3_did.csv" --B 5000
fi

"$PY" scripts/decide_hypotheses.py --root "$ROOT" --out "$ROOT/posthoc/verdicts"
