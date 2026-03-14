#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

mkdir -p logs checkpoints/arcs_graph_transformer_topo_value_medium checkpoints/arcs_graph_transformer_family_moe_medium results

BASE_CKPT="checkpoints/arcs_graph_transformer/best_model.pt"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing baseline checkpoint: $BASE_CKPT"
  exit 1
fi

EPOCHS="${EPOCHS:-8}"
N_SAMPLES="${N_SAMPLES:-80}"

echo "[1/3] Train topology-value-head variant (medium run, epochs=$EPOCHS)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --value-weight 5.0 \
  --warmup-epochs 1 \
  --valid-only \
  --augment \
  --n-augmentations 3 \
  --output checkpoints/arcs_graph_transformer_topo_value_medium \
  --log-interval 1 \
  --save-interval "$EPOCHS" \
  --seed 42 \
  --resume "$BASE_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --use-topology-value-heads \
  --topology-value-head-alpha 0.5 \
  2>&1 | tee logs/train_topology_value_medium.log

echo "[2/3] Train topology-family-moe variant (medium run, epochs=$EPOCHS)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs "$EPOCHS" \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --value-weight 5.0 \
  --warmup-epochs 1 \
  --valid-only \
  --augment \
  --n-augmentations 3 \
  --output checkpoints/arcs_graph_transformer_family_moe_medium \
  --log-interval 1 \
  --save-interval "$EPOCHS" \
  --seed 42 \
  --resume "$BASE_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --use-topology-value-heads \
  --topology-value-head-alpha 0.5 \
  --use-topology-family-moe \
  --topology-family-moe-alpha 0.3 \
  2>&1 | tee logs/train_family_moe_medium.log

echo "[3/3] Run topology ablation evaluation (n_samples=$N_SAMPLES)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" scripts/evaluate_topology_ablation.py \
  --baseline-ckpt checkpoints/arcs_graph_transformer/best_model.pt \
  --topo-head-ckpt checkpoints/arcs_graph_transformer_topo_value_medium/best_model.pt \
  --family-moe-ckpt checkpoints/arcs_graph_transformer_family_moe_medium/best_model.pt \
  --n-samples "$N_SAMPLES" \
  --output results/topology_ablation_medium.json \
  2>&1 | tee logs/evaluate_topology_ablation_medium.log

echo "Done. Results: results/topology_ablation_medium.json"
