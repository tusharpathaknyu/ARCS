#!/usr/bin/env bash
# Pilot3: Stability-first topology-head-only recipe.
#
# Changes vs pilot2:
#   - LR=1e-5 (halved from 2e-5)
#   - 1 epoch (not 2)
#   - Topology-head only (no family-MoE)
#   - Alpha=0.1 (more conservative blending)
#   - No augmentation
#
# This tests whether a gentler finetuning regime can preserve baseline quality
# while adding topology-specific value specialization.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

BASE_CKPT="checkpoints/arcs_graph_transformer/best_model.pt"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing baseline checkpoint: $BASE_CKPT"
  exit 1
fi

EPOCHS="${EPOCHS:-1}"
N_SAMPLES="${N_SAMPLES:-48}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
VALUE_WEIGHT="${VALUE_WEIGHT:-5.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
RUN_TAG="${RUN_TAG:-pilot3}"
TOPO_ALPHA="${TOPO_ALPHA:-0.1}"

OUT_ROOT="checkpoints/topology_sweep_pilot/${RUN_TAG}"
LOG_DIR="logs"
RES_DIR="results"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$RES_DIR"

TOPO_OUT="$OUT_ROOT/topology_heads"
mkdir -p "$TOPO_OUT"

COMMON_ARGS=(
  --data data/combined
  --config small
  --model-type graph_transformer
  --epochs "$EPOCHS"
  --batch-size 64
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --value-weight "$VALUE_WEIGHT"
  --warmup-epochs "$WARMUP_EPOCHS"
  --valid-only
  --log-interval 1
  --save-interval "$EPOCHS"
  --seed "$SEED"
  --resume "$BASE_CKPT"
  --resume-weights-only
  --resume-allow-partial
  --init-experts-from-shared
)

# No augmentation for cleaner signal
TRAIN_ARGS=("${COMMON_ARGS[@]}")

echo "[1/2] Train topology-head-only (alpha=$TOPO_ALPHA, epochs=$EPOCHS, lr=$LR, seed=$SEED)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  "${TRAIN_ARGS[@]}" \
  --output "$TOPO_OUT" \
  --use-topology-value-heads \
  --topology-value-head-alpha "$TOPO_ALPHA" \
  2>&1 | tee "$LOG_DIR/train_topology_heads_${RUN_TAG}.log"

echo "[2/2] Evaluate baseline vs topology-head"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" scripts/evaluate_topology_ablation.py \
  --baseline-ckpt "$BASE_CKPT" \
  --topo-head-ckpt "$TOPO_OUT/best_model.pt" \
  --family-moe-ckpt "SKIP_NO_FAMILY_MOE" \
  --n-samples "$N_SAMPLES" \
  --seed "$SEED" \
  --output "$RES_DIR/topology_training_sweep_${RUN_TAG}_eval.json" \
  2>&1 | tee "$LOG_DIR/evaluate_topology_training_sweep_${RUN_TAG}.log"

echo "Done. Pilot3 artifact: $RES_DIR/topology_training_sweep_${RUN_TAG}_eval.json"
