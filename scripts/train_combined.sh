#!/bin/bash
# ARCS Phase 4: Retrain model on combined Tier 1 + Tier 2 data
#
# Vocab expanded 676→686, so fresh training is required (no resume).
# Combined data: ~35K Phase 1 (power converters) + ~18K Tier 2 (signal processing)
#
# Usage: bash scripts/train_combined.sh

set -e

echo "================================================="
echo "ARCS Phase 4: Combined Model Training"
echo "================================================="

# Create combined data directory with symlinks
COMBINED="data/combined"
rm -rf "$COMBINED"
mkdir -p "$COMBINED"

echo "Linking Phase 1 data..."
for f in data/phase1/*.jsonl; do
    ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$COMBINED/"
done

echo "Linking Tier 2 data..."
for f in data/tier2/*.jsonl; do
    ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$COMBINED/"
done

echo "Combined data:"
ls -la "$COMBINED"/*.jsonl | wc -l
echo "files"

source .venv/bin/activate

PYTHONPATH=src python -m arcs.train \
    --data "$COMBINED" \
    --config small \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --value-weight 5.0 \
    --warmup-epochs 5 \
    --valid-only \
    --augment \
    --n-augmentations 5 \
    --output checkpoints/arcs_combined \
    --log-interval 5 \
    --save-interval 25 \
    --seed 42 \
    2>&1
