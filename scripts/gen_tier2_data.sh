#!/bin/bash
# ARCS Phase 4: Generate Tier 2 (signal processing) training data
#
# Generates data for 9 new topologies:
#   - Amplifiers: inverting_amp, noninverting_amp, instrumentation_amp, differential_amp
#   - Filters: sallen_key_lowpass, sallen_key_highpass, sallen_key_bandpass
#   - Oscillators: wien_bridge, colpitts
#
# Usage: bash scripts/gen_tier2_data.sh [samples_per_topology]

set -e

SAMPLES=${1:-2000}
OUTPUT="data/raw"

echo "================================================="
echo "ARCS Phase 4: Tier 2 Data Generation"
echo "================================================="
echo "Samples per topology: $SAMPLES"
echo "Output: $OUTPUT"
echo ""

source .venv/bin/activate

PYTHONPATH=src python -m arcs.datagen \
    --topologies \
        inverting_amp \
        noninverting_amp \
        instrumentation_amp \
        differential_amp \
        sallen_key_lowpass \
        sallen_key_highpass \
        sallen_key_bandpass \
        wien_bridge \
        colpitts \
    --samples "$SAMPLES" \
    --output "$OUTPUT" \
    --timeout 60 \
    --seed 42 \
    2>&1
