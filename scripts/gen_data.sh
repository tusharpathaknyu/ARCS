#!/bin/bash
# Generate Phase 1 training data
cd /Users/tushardhananjaypathak/Desktop/CircuitGenie
source .venv/bin/activate
export PYTHONPATH=src
mkdir -p data
python3 -m arcs.datagen \
    --topologies buck boost buck_boost cuk sepic flyback forward \
    --samples 5000 \
    --output data/phase1
