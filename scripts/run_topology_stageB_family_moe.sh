#!/usr/bin/env bash
# Stage B: Family-MoE warm-start from pilot3b topology-head checkpoints.
#
# Two-stage finetuning strategy:
#   Stage A (pilot3b): baseline → topology-head-only (shared-init, 1 epoch, LR=1e-5)
#   Stage B (this):    pilot3b topo-head → add family-MoE experts (shared-init, 1 epoch, LR=1e-5)
#
# The key insight: warm-starting MoE from an already-stable topo-head checkpoint
# avoids catastrophic forgetting and lets family experts specialize gradually.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

BASE_CKPT="checkpoints/arcs_graph_transformer/best_model.pt"

EPOCHS="${EPOCHS:-1}"
N_SAMPLES="${N_SAMPLES:-48}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
VALUE_WEIGHT="${VALUE_WEIGHT:-5.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
RUN_TAG="${RUN_TAG:-stageB}"
TOPO_ALPHA="${TOPO_ALPHA:-0.1}"
FAMILY_ALPHA="${FAMILY_ALPHA:-0.1}"
STAGE_A_PREFIX="${STAGE_A_PREFIX:-pilot3b}"

# Stage A checkpoint (from pilot3b topo-head training)
STAGE_A_TAG="${STAGE_A_PREFIX}_seed${SEED}"
STAGE_A_CKPT="checkpoints/topology_sweep_pilot/${STAGE_A_TAG}/topology_heads/best_model.pt"

if [[ ! -f "$STAGE_A_CKPT" ]]; then
  echo "Missing Stage A checkpoint: $STAGE_A_CKPT"
  echo "Run pilot3b first to create topology-head checkpoints."
  exit 1
fi

OUT_ROOT="checkpoints/topology_sweep_pilot/${RUN_TAG}"
LOG_DIR="logs"
RES_DIR="results"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$RES_DIR"

FAMILY_OUT="$OUT_ROOT/family_moe"
mkdir -p "$FAMILY_OUT"

echo "[1/2] Train family-MoE on top of Stage A topo-head (alpha_t=$TOPO_ALPHA, alpha_f=$FAMILY_ALPHA, epochs=$EPOCHS, lr=$LR, seed=$SEED)"
echo "  Stage A checkpoint: $STAGE_A_CKPT"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --value-weight "$VALUE_WEIGHT" \
  --warmup-epochs "$WARMUP_EPOCHS" \
  --valid-only \
  --log-interval 1 \
  --save-interval "$EPOCHS" \
  --seed "$SEED" \
  --resume "$STAGE_A_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --init-experts-from-shared \
  --output "$FAMILY_OUT" \
  --use-topology-value-heads \
  --topology-value-head-alpha "$TOPO_ALPHA" \
  --use-topology-family-moe \
  --topology-family-moe-alpha "$FAMILY_ALPHA" \
  2>&1 | tee "$LOG_DIR/train_family_moe_${RUN_TAG}.log"

echo "[2/2] Evaluate baseline vs Stage A topo-head vs Stage B family-MoE"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" scripts/evaluate_topology_ablation.py \
  --baseline-ckpt "$BASE_CKPT" \
  --topo-head-ckpt "$STAGE_A_CKPT" \
  --family-moe-ckpt "$FAMILY_OUT/best_model.pt" \
  --n-samples "$N_SAMPLES" \
  --seed "$SEED" \
  --output "$RES_DIR/topology_training_sweep_${RUN_TAG}_eval.json" \
  2>&1 | tee "$LOG_DIR/evaluate_topology_training_sweep_${RUN_TAG}.log"

echo "Done. Stage B artifact: $RES_DIR/topology_training_sweep_${RUN_TAG}_eval.json"
