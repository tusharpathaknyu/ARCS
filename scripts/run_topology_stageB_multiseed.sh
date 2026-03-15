#!/usr/bin/env bash
# Stage B multiseed: runs family-MoE warm-start from pilot3b topo-head
# checkpoints across 5 seeds and aggregates results.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

SEEDS="${SEEDS:-41 42 43 44 45}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
N_SAMPLES="${N_SAMPLES:-48}"
TOPO_ALPHA="${TOPO_ALPHA:-0.1}"
FAMILY_ALPHA="${FAMILY_ALPHA:-0.1}"
RUN_PREFIX="${RUN_PREFIX:-stageB}"
STAGE_A_PREFIX="${STAGE_A_PREFIX:-pilot3b}"
OUT_JSON="${OUT_JSON:-results/topology_training_sweep_stageB_multiseed.json}"

mkdir -p logs results

echo "Running Stage B multiseed sweep (family-MoE warm-start from ${STAGE_A_PREFIX})"
echo "  seeds: $SEEDS"
echo "  epochs: $EPOCHS, lr: $LR, topo_alpha: $TOPO_ALPHA, family_alpha: $FAMILY_ALPHA"

for seed in $SEEDS; do
  run_tag="${RUN_PREFIX}_seed${seed}"
  echo ""
  echo "=== Seed $seed (${run_tag}) ==="
  RUN_TAG="$run_tag" \
  EPOCHS="$EPOCHS" \
  LR="$LR" \
  N_SAMPLES="$N_SAMPLES" \
  SEED="$seed" \
  TOPO_ALPHA="$TOPO_ALPHA" \
  FAMILY_ALPHA="$FAMILY_ALPHA" \
  STAGE_A_PREFIX="$STAGE_A_PREFIX" \
  bash scripts/run_topology_stageB_family_moe.sh

done

echo ""
echo "Aggregating results..."
PYTHONPATH=src "$PY" scripts/summarize_topology_training_multiseed.py \
  --run-prefix "$RUN_PREFIX" \
  --seeds $SEEDS \
  --output "$OUT_JSON"

echo "Done. Summary: $OUT_JSON"
