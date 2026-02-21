#!/bin/bash
# ARCS Phase 3: SPICE-in-the-Loop RL Training
# Usage: bash scripts/train_rl.sh [checkpoint_path] [n_steps]

set -e

CHECKPOINT=${1:-"checkpoints/arcs_small/best_model.pt"}
STEPS=${2:-5000}
OUTPUT="checkpoints/arcs_rl"

echo "================================================="
echo "ARCS Phase 3: SPICE-in-the-Loop RL Training"
echo "================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Steps:      $STEPS"
echo "Output:     $OUTPUT"
echo ""

source .venv/bin/activate

PYTHONPATH=src python -m arcs.rl \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --steps "$STEPS" \
    --batch-size 8 \
    --lr 1e-5 \
    --kl-coeff 0.1 \
    --entropy-coeff 0.01 \
    --temperature 0.8 \
    --top-k 50 \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --n-eval-samples 50 \
    --seed 42 \
    2>&1
