#!/bin/bash
# ARCS Phase 6: Train TwoHeadARCSModel on combined data
#
# Same data + hyperparams as baseline combined training,
# but with --model-type two_head for separate structure/value heads.
#
# Expected: ~27 hours on M3 MacBook Air (MPS)
# Usage: bash scripts/train_two_head.sh

set -e

echo "================================================="
echo "ARCS Phase 6: Two-Head Model Training"
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
    --model-type two_head \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --weight-decay 0.1 \
    --value-weight 5.0 \
    --warmup-epochs 5 \
    --valid-only \
    --augment \
    --n-augmentations 5 \
    --output checkpoints/arcs_two_head \
    --log-interval 5 \
    --save-interval 25 \
    --seed 42 \
    2>&1 | tee logs/train_two_head.log
