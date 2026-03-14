#!/usr/bin/env bash
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
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
VALUE_WEIGHT="${VALUE_WEIGHT:-5.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
N_AUGMENTATIONS="${N_AUGMENTATIONS:-1}"
USE_AUGMENT="${USE_AUGMENT:-1}"
RUN_TAG="${RUN_TAG:-pilot}"
TOPO_ALPHA="${TOPO_ALPHA:-0.2}"
FAMILY_TOPO_ALPHA="${FAMILY_TOPO_ALPHA:-0.5}"
FAMILY_ALPHA="${FAMILY_ALPHA:-0.5}"

OUT_ROOT="checkpoints/topology_sweep_pilot/${RUN_TAG}"
LOG_DIR="logs"
RES_DIR="results"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$RES_DIR"

TOPO_OUT="$OUT_ROOT/topology_heads"
FAMILY_OUT="$OUT_ROOT/family_moe"

mkdir -p "$TOPO_OUT" "$FAMILY_OUT"

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
)

if [[ "$USE_AUGMENT" == "1" ]]; then
  TRAIN_ARGS=("${COMMON_ARGS[@]}" --augment --n-augmentations "$N_AUGMENTATIONS")
else
  TRAIN_ARGS=("${COMMON_ARGS[@]}")
fi

echo "[1/3] Train topology-head candidate (alpha=$TOPO_ALPHA, epochs=$EPOCHS, lr=$LR)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  "${TRAIN_ARGS[@]}" \
  --output "$TOPO_OUT" \
  --use-topology-value-heads \
  --topology-value-head-alpha "$TOPO_ALPHA" \
  2>&1 | tee "$LOG_DIR/train_topology_heads_${RUN_TAG}.log"

echo "[2/3] Train family-moe candidate (alpha_f=$FAMILY_ALPHA, alpha_t=$FAMILY_TOPO_ALPHA, epochs=$EPOCHS, lr=$LR)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  "${TRAIN_ARGS[@]}" \
  --output "$FAMILY_OUT" \
  --use-topology-value-heads \
  --topology-value-head-alpha "$FAMILY_TOPO_ALPHA" \
  --use-topology-family-moe \
  --topology-family-moe-alpha "$FAMILY_ALPHA" \
  2>&1 | tee "$LOG_DIR/train_family_moe_${RUN_TAG}.log"

echo "[3/3] Evaluate baseline vs pilot candidates"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" scripts/evaluate_topology_ablation.py \
  --baseline-ckpt "$BASE_CKPT" \
  --topo-head-ckpt "$TOPO_OUT/best_model.pt" \
  --family-moe-ckpt "$FAMILY_OUT/best_model.pt" \
  --n-samples "$N_SAMPLES" \
  --seed "$SEED" \
  --output "$RES_DIR/topology_training_sweep_${RUN_TAG}_eval.json" \
  2>&1 | tee "$LOG_DIR/evaluate_topology_training_sweep_${RUN_TAG}.log"

echo "Done. Pilot artifact: $RES_DIR/topology_training_sweep_${RUN_TAG}_eval.json"
