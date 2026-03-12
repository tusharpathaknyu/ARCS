#!/bin/bash
# ARCS Phase 13: GRPO RL Fine-Tuning of Graph Transformer
#
# Uses Group Relative Policy Optimization instead of vanilla REINFORCE.
# Per-topology z-scored advantages prevent cross-topology interference.
#
# Usage: bash scripts/train_grpo.sh

set -e

CHECKPOINT="checkpoints/arcs_graph_transformer/best_model.pt"
STEPS=3000
OUTPUT="checkpoints/arcs_grpo"

echo "================================================="
echo "ARCS Phase 13: GRPO RL Fine-Tuning"
echo "================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Steps:      $STEPS"
echo "Output:     $OUTPUT"
echo ""

source .venv/bin/activate
mkdir -p logs "$OUTPUT"

PYTHONUNBUFFERED=1 PYTHONPATH=src python -m arcs.rl \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --steps "$STEPS" \
    --batch-size 8 \
    --lr 1e-5 \
    --kl-coeff 0.1 \
    --entropy-coeff 0.01 \
    --temperature 0.8 \
    --top-k 50 \
    --grpo \
    --group-size 4 \
    --n-topos-per-step 3 \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --n-eval-samples 50 \
    --seed 42 \
    2>&1 | tee logs/train_grpo.log
