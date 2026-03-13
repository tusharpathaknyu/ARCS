#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing python at $PY"
  exit 1
fi

mkdir -p logs checkpoints/arcs_graph_transformer_topo_value checkpoints/arcs_graph_transformer_family_moe results

BASE_CKPT="checkpoints/arcs_graph_transformer/best_model.pt"
if [[ ! -f "$BASE_CKPT" ]]; then
  echo "Missing baseline checkpoint: $BASE_CKPT"
  exit 1
fi

echo "[1/3] Train topology-value-head variant (short run)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --value-weight 5.0 \
  --warmup-epochs 1 \
  --valid-only \
  --augment \
  --n-augmentations 3 \
  --output checkpoints/arcs_graph_transformer_topo_value \
  --log-interval 1 \
  --save-interval 5 \
  --seed 42 \
  --resume "$BASE_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --use-topology-value-heads \
  --topology-value-head-alpha 0.5 \
  2>&1 | tee logs/train_topology_value_short.log

echo "[2/3] Train topology-family-moe variant (short run)"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" -m arcs.train \
  --data data/combined \
  --config small \
  --model-type graph_transformer \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0.1 \
  --value-weight 5.0 \
  --warmup-epochs 1 \
  --valid-only \
  --augment \
  --n-augmentations 3 \
  --output checkpoints/arcs_graph_transformer_family_moe \
  --log-interval 1 \
  --save-interval 5 \
  --seed 42 \
  --resume "$BASE_CKPT" \
  --resume-weights-only \
  --resume-allow-partial \
  --use-topology-value-heads \
  --topology-value-head-alpha 0.5 \
  --use-topology-family-moe \
  --topology-family-moe-alpha 0.3 \
  2>&1 | tee logs/train_family_moe_short.log

echo "[3/3] Run topology ablation evaluation"
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PY" scripts/evaluate_topology_ablation.py \
  --baseline-ckpt checkpoints/arcs_graph_transformer/best_model.pt \
  --topo-head-ckpt checkpoints/arcs_graph_transformer_topo_value/best_model.pt \
  --family-moe-ckpt checkpoints/arcs_graph_transformer_family_moe/best_model.pt \
  --n-samples 48 \
  --output results/topology_ablation_short.json \
  2>&1 | tee logs/evaluate_topology_ablation_short.log

echo "Done. Results: results/topology_ablation_short.json"
