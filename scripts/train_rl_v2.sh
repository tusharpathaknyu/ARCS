#!/bin/bash
# ARCS Phase 3 v2: RL Training with validity-regression fixes
#
# Key changes from v1:
#   - kl_coeff 0.1 → 0.3  (stronger KL penalty prevents drift)
#   - struct_bonus 1.0 → 2.0 (double structure reward to guard validity)
#   - adaptive_kl enabled with target=0.5 (auto-adjusts if KL drifts)
#   - lr 1e-5 → 5e-6 (slower, more stable updates)
#   - validity_early_stop at 60% (abort if structure breaks)
#   - Starts from best RL checkpoint (not pre-trained), for continued tuning
#
# Usage: bash scripts/train_rl_v2.sh [checkpoint_path] [n_steps]

set -e

CHECKPOINT=${1:-"checkpoints/arcs_rl/best_rl_model.pt"}
STEPS=${2:-3000}
OUTPUT="checkpoints/arcs_rl_v2"

echo "================================================="
echo "ARCS Phase 3 v2: RL Training (validity fix)"
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
    --lr 5e-6 \
    --kl-coeff 0.3 \
    --entropy-coeff 0.01 \
    --struct-bonus 2.0 \
    --adaptive-kl \
    --kl-target 0.5 \
    --validity-early-stop 0.6 \
    --temperature 0.8 \
    --top-k 50 \
    --log-interval 10 \
    --save-interval 500 \
    --eval-interval 100 \
    --n-eval-samples 50 \
    --seed 42 \
    2>&1
