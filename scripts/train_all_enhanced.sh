#!/bin/bash
# ARCS Phase 6: Train both enhanced models sequentially
#
# Runs two_head first, then graph_transformer, then comparison eval.
# Total time: ~57 hours (27 + 30)
#
# Usage: nohup bash scripts/train_all_enhanced.sh > logs/train_all_enhanced.log 2>&1 &

set -e

echo "================================================="
echo "ARCS Phase 6: Sequential Enhanced Model Training"
echo "Started: $(date)"
echo "================================================="

echo ""
echo ">>> Step 1/3: Training Two-Head Model..."
echo ">>> Start: $(date)"
bash scripts/train_two_head.sh
echo ">>> Two-Head complete: $(date)"

echo ""
echo ">>> Step 2/3: Training Graph Transformer Model..."
echo ">>> Start: $(date)"
bash scripts/train_graph_transformer.sh
echo ">>> Graph Transformer complete: $(date)"

echo ""
echo ">>> Step 3/3: Running Architecture Comparison..."
echo ">>> Start: $(date)"
source .venv/bin/activate
PYTHONPATH=src python scripts/compare_architectures.py \
    --n-samples 160 \
    --output results/arch_comparison.json \
    -v
echo ">>> Comparison complete: $(date)"

echo ""
echo "================================================="
echo "Phase 6 Complete: $(date)"
echo "================================================="
