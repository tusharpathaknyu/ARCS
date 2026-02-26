#!/bin/bash
# ARCS Phase 6: Train GraphTransformerARCSModel on combined data
#
# Same data + hyperparams as baseline combined training,
# but with --model-type graph_transformer for topology-aware attention.
#
# Note: graph_transformer requires the tokenizer for graph feature
# computation during training. train.py handles this automatically.
#
# Expected: ~30 hours on M3 MacBook Air (slightly slower due to graph features)
# Usage: bash scripts/train_graph_transformer.sh

set -e

echo "================================================="
echo "ARCS Phase 6: Graph Transformer Model Training"
echo "================================================="

# Ensure combined data symlinks exist
COMBINED="data/combined"
if [ ! -d "$COMBINED" ] || [ -z "$(ls -A "$COMBINED" 2>/dev/null)" ]; then
    echo "Setting up combined data symlinks..."
    rm -rf "$COMBINED"
    mkdir -p "$COMBINED"
    for f in data/phase1/*.jsonl; do
        ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$COMBINED/"
    done
    for f in data/tier2/*.jsonl; do
        ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$COMBINED/"
    done
fi

echo "Combined data files: $(ls "$COMBINED"/*.jsonl | wc -l)"

source .venv/bin/activate
mkdir -p logs

PYTHONPATH=src python -m arcs.train \
    --data "$COMBINED" \
    --config small \
    --model-type graph_transformer \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --value-weight 5.0 \
    --warmup-epochs 5 \
    --valid-only \
    --augment \
    --n-augmentations 5 \
    --output checkpoints/arcs_graph_transformer \
    --log-interval 5 \
    --save-interval 25 \
    --seed 42 \
    2>&1 | tee logs/train_graph_transformer.log
