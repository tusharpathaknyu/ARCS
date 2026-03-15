#!/usr/bin/env bash
# Pilot3 multiseed wrapper: runs stability-first topology-head-only sweep
# across 5 seeds and aggregates results.
#
# Recipe: LR=1e-5, 1 epoch, topo-head alpha=0.1, no MoE, no augmentation.
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
RUN_PREFIX="${RUN_PREFIX:-pilot3}"
OUT_JSON="${OUT_JSON:-results/topology_training_sweep_pilot3_multiseed.json}"

mkdir -p logs results

echo "Running pilot3 multiseed sweep (stability-first, topology-head only)"
echo "  seeds: $SEEDS"
echo "  epochs: $EPOCHS, lr: $LR, alpha: $TOPO_ALPHA, n_samples: $N_SAMPLES"

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
  bash scripts/run_topology_training_sweep_pilot3.sh

done

echo ""
echo "Aggregating results..."
PYTHONPATH=src "$PY" scripts/summarize_topology_training_pilot3_multiseed.py \
  --run-prefix "$RUN_PREFIX" \
  --seeds $SEEDS \
  --output "$OUT_JSON"

echo "Done. Summary: $OUT_JSON"
