#!/bin/bash
# Train ARCS model on Phase 1 data
# Usage: bash scripts/train_model.sh [small|base]

set -e

cd /Users/tushardhananjaypathak/Desktop/CircuitGenie
source .venv/bin/activate
export PYTHONPATH=src

CONFIG=${1:-small}
DATA_DIR="data/phase1"
OUTPUT_DIR="checkpoints/${CONFIG}"

echo "============================================================"
echo "ARCS Model Training"
echo "  Config:  ${CONFIG}"
echo "  Data:    ${DATA_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================================"

# Check data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory ${DATA_DIR} not found."
    echo "Run 'bash scripts/gen_data.sh' first."
    exit 1
fi

N_FILES=$(ls ${DATA_DIR}/*.jsonl 2>/dev/null | wc -l)
echo "Found ${N_FILES} JSONL files"

python -m arcs.train \
    --data "${DATA_DIR}" \
    --config "${CONFIG}" \
    --epochs 100 \
    --batch-size 64 \
    --lr 3e-4 \
    --value-weight 5.0 \
    --warmup-epochs 5 \
    --val-split 0.1 \
    --augment \
    --n-augmentations 5 \
    --output "${OUTPUT_DIR}" \
    --log-interval 5 \
    --save-interval 25 \
    --seed 42

echo ""
echo "Training complete. Checkpoints in: ${OUTPUT_DIR}"
echo "Run evaluation: PYTHONPATH=src python -m arcs.evaluate --checkpoint ${OUTPUT_DIR}/best_model.pt"
