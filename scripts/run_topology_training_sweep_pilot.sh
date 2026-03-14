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

OUT_ROOT="checkpoints/topology_sweep_pilot"
LOG_DIR="logs"
RES_DIR="results"
mkdir -p "$OUT_ROOT" "$LOG_DIR" "$RES_DIR"

TOPO_OUT="$OUT_ROOT/topology_heads_a02"
FAMILY_OUT="$OUT_ROOT/family_moe_a05"

mkdir -p "$TOPO_OUT" "$FAMILY_OUT"

echo "[1/3] Train topology-head candidate (alpha=0.2, epochs=$EPOCHS)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --lr 5e-5 \
  --weight-decay 0.1 \
  --value-weight 5.0 \
  --warmup-epochs 1 \
  --valid-only \
  --augment \
  --n-augmentations 1 \
  --output "$TOPO_OUT" \
  --log-interval 1 \
  --save-interval "$EPOCHS" \
  --seed "$SEED" \
  --resume "$BASE_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --use-topology-value-heads \
  --topology-value-head-alpha 0.2 \
  2>&1 | tee "$LOG_DIR/train_topology_heads_a02_pilot.log"

echo "[2/3] Train family-moe candidate (alpha_f=0.5, alpha_t=0.5, epochs=$EPOCHS)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --lr 5e-5 \
  --weight-decay 0.1 \
  --value-weight 5.0 \
  --warmup-epochs 1 \
  --valid-only \
  --augment \
  --n-augmentations 1 \
  --output "$FAMILY_OUT" \
  --log-interval 1 \
  --save-interval "$EPOCHS" \
  --seed "$SEED" \
  --resume "$BASE_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --use-topology-value-heads \
  --topology-value-head-alpha 0.5 \
  --use-topology-family-moe \
  --topology-family-moe-alpha 0.5 \
  2>&1 | tee "$LOG_DIR/train_family_moe_a05_pilot.log"

echo "[3/3] Evaluate baseline vs pilot candidates"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" scripts/evaluate_topology_ablation.py \
  --baseline-ckpt "$BASE_CKPT" \
  --topo-head-ckpt "$TOPO_OUT/best_model.pt" \
  --family-moe-ckpt "$FAMILY_OUT/best_model.pt" \
  --n-samples "$N_SAMPLES" \
  --seed "$SEED" \
  --output "$RES_DIR/topology_training_sweep_pilot_eval.json" \
  2>&1 | tee "$LOG_DIR/evaluate_topology_training_sweep_pilot.log"

echo "Done. Pilot artifact: $RES_DIR/topology_training_sweep_pilot_eval.json"
