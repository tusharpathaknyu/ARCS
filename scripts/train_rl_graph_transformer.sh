#!/bin/bash
# ARCS Phase 6: SPICE-in-the-Loop RL Fine-Tuning of Graph Transformer
#
# Takes the best supervised Graph Transformer checkpoint and fine-tunes
# with REINFORCE + KL penalty for improved SPICE simulation quality.
#
# Expected: ~12-16 hours on M3 MacBook Air
# Usage: bash scripts/train_rl_graph_transformer.sh

set -e

CHECKPOINT="checkpoints/arcs_graph_transformer/best_model.pt"
STEPS=5000
OUTPUT="checkpoints/arcs_rl_graph_transformer"

echo "================================================="
echo "ARCS Phase 6: RL Fine-Tuning (Graph Transformer)"
echo "================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Steps:      $STEPS"
echo "Output:     $OUTPUT"
echo ""

source .venv/bin/activate
mkdir -p logs

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
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --n-eval-samples 50 \
    --seed 42 \
    2>&1
