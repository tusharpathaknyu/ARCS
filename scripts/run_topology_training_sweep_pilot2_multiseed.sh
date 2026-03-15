#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

SEEDS="${SEEDS:-41 42 43 44 45}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
N_SAMPLES="${N_SAMPLES:-48}"
USE_AUGMENT="${USE_AUGMENT:-0}"
RUN_PREFIX="${RUN_PREFIX:-pilot2}"
OUT_JSON="${OUT_JSON:-results/topology_training_sweep_pilot2_multiseed.json}"

mkdir -p logs results

echo "Running pilot2 multiseed sweep"
echo "  seeds: $SEEDS"
echo "  epochs: $EPOCHS, lr: $LR, n_samples: $N_SAMPLES, use_augment: $USE_AUGMENT"

for seed in $SEEDS; do
  run_tag="${RUN_PREFIX}_seed${seed}"
  echo "\n=== Seed $seed (${run_tag}) ==="
  RUN_TAG="$run_tag" \
  EPOCHS="$EPOCHS" \
  LR="$LR" \
  USE_AUGMENT="$USE_AUGMENT" \
  N_SAMPLES="$N_SAMPLES" \
  SEED="$seed" \
  bash scripts/run_topology_training_sweep_pilot.sh

done

echo "\nAggregating results..."
PYTHONPATH=src "$PY" scripts/summarize_topology_training_multiseed.py \
  --run-prefix "$RUN_PREFIX" \
  --seeds $SEEDS \
  --output "$OUT_JSON"

echo "Done. Summary: $OUT_JSON"
