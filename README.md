# ARCS: Autoregressive Representation for Circuit Synthesis

[![phase16-consistency](https://github.com/tusharpathaknyu/ARCS/actions/workflows/phase16-consistency.yml/badge.svg)](https://github.com/tusharpathaknyu/ARCS/actions/workflows/phase16-consistency.yml)

> **"Components are words. Circuits are sentences."**

A native autoregressive model that generates complete analog circuits — topology AND component values — conditioned on performance specifications in a single forward pass (~20ms). Unlike text-LLM approaches that predict characters and topology-only models (AnalogGenie) that require separate GA sizing, ARCS generates SPICE-simulatable designs directly from specs.

---

## The Problem

Analog circuit design is manual, slow, and requires deep expertise. Existing AI approaches have critical gaps:

| Approach | What It Does | What It's Missing |
|----------|-------------|-------------------|
| **AnalogGenie** (ICLR 2025) | Generates topology (pin-level wiring) | No component values, no spec conditioning, requires GA sizing |
| **CircuitSynth** (2024) | Fine-tunes GPT-Neo on netlist text | Model predicts characters, not components — error-prone |
| **AutoCircuit-RL** (2025) | Adds RL to CircuitSynth | Same text-LLM limitation |
| **cktFormer** (2025) | Transformer on graph tokens | IC-only, limited topology families |
| **FALCON** (NeurIPS 2025) | GNN + layout co-optimization | Parasitic-aware but not generative |

**ARCS is the first autoregressive model that jointly generates topology + component values, conditioned on specs, producing complete designs in a single forward pass.**

---

## Approach

### Core Idea

```
INPUT:  "Vout=5V, Vin=12V, Iout=1A, fsw=100kHz"

MODEL GENERATES (token by token, ~20ms):
  START -> TOPO_BUCK -> SEP -> SPEC_VIN 12.19 -> SPEC_VOUT 4.90 -> ... -> SEP ->
  CAPACITOR 38uF -> RESISTOR 130mOhm -> MOSFET_N 21mOhm -> INDUCTOR 300uH -> END

OUTPUT: Component list + topology -> SPICE netlist -> simulate -> validate
```

### Key Contributions

1. **Native Component Tokens** — Each token IS a circuit element with value, not a text character
2. **Value-Aware Generation** — Component values embedded in the token representation (not a separate post-hoc sizing step)
3. **Spec Conditioning** — Performance requirements as prefix, model generates circuits meeting those specs
4. **GRPO for Multi-Topology RL** — Per-topology advantage normalization fixes cross-topology reward mismatch
5. **Grammar-Constrained Decoding** — 100% structural validity by construction via token masking
6. **Hybrid Multi-Source Ranking** — VCG + CCFM generators with SPICE-based ranking (99.9% sim validity)

---

## Architecture

### Tokenizer (706 tokens)

- **Component tokens** (20): MOSFET_N/P, RESISTOR, CAPACITOR, INDUCTOR, DIODE, BJT_NPN/PNP, OPAMP, TRANSFORMER, ...
- **Value tokens** (500): Log-uniform bins from 10^-12 to 10^6 (~28 bins/decade)
- **Topology tokens** (34): TOPO_BUCK, TOPO_BOOST, ..., TOPO_ZETA_CONVERTER
- **Spec tokens** (20): SPEC_VIN, SPEC_VOUT, SPEC_IOUT, SPEC_FSW, SPEC_GAIN, ...
- **Special tokens** (5): START, END, PAD, SEP, INVALID

### Models

| Model | Params | Description |
|-------|--------|-------------|
| **Baseline GPT** | ~6.5M | Standard decoder-only transformer (d_model=256, 6 layers, 4 heads, SwiGLU, RMSNorm) |
| **Two-Head** | ~6.8M | Separate structure head (weight-tied) + value head (SiLU MLP with residual) |
| **Graph Transformer** | ~6.8M | Topology-aware causal attention + RWPE (K=8 random walks) |
| **VCG** | ~4.0M | Constrained VAE with 5 differentiable circuit constraints |
| **CCFM** | ~7.6M | Conditional flow matching with constraint-guided ODE sampling |
| **Reward Model** | ~666K | Bidirectional transformer for Best-of-N candidate ranking |

Select model type via `--model-type {baseline,two_head,graph_transformer}` in training, RL, evaluation, and demo scripts.

### Training Pipeline

1. **Supervised pre-training**: Next-token prediction on all circuit sequences, 5x value-token loss weight
2. **RL fine-tuning**: GRPO with per-topology z-scored advantages, KL penalty, SPICE simulation reward

---

## Results

### Autoregressive Architecture Comparison (160 samples x 5 seeds, mean +/- std)

| Model | Struct Valid | Sim Valid | Reward |
|-------|-------------|-----------|--------|
| Baseline GPT | 86.0+/-1.9% | 40.1+/-0.7% | 3.37+/-0.06 |
| Two-Head SL | 96.2+/-1.0% | 50.4+/-2.2% | 3.89+/-0.09 |
| Graph Transformer SL | 93.2+/-1.4% | 45.4+/-3.0% | 3.86+/-0.15 |
| GT + REINFORCE | 95.5+/-1.3% | 43.5+/-1.2% | 3.74+/-0.04 |
| **GT + GRPO (500 steps)** | **96.6+/-0.5%** | **53.1+/-3.1%** | **4.15+/-0.08** |

### Publication Evaluation (50 samples x 32 topologies, bootstrap 95% CI)

| Method | SPICE Evals | Reward (95% CI) | Sim Valid% | Wall Time |
|--------|-------------|-----------------|------------|-----------|
| Random Search | 1 | 5.18 [5.05, 5.31] | 91.4% | ~0.3s |
| Genetic Algorithm | 320 | 7.56 [7.53, 7.60] | 100.0% | ~120s |
| VCG v5 (alone) | 1 | 5.34 [5.27, 5.42] | 94.0% | ~0.02s |
| CCFM v5 (alone) | 1 | 5.51 [5.44, 5.58] | 95.7% | ~0.16s |
| **Hybrid v5 (VCG+CCFM)** | **8** | **6.43 [6.38, 6.48]** | **99.9%** | **~0.05s** |

All pairwise comparisons significant at p < 0.001 (Wilcoxon signed-rank).

### Graph-Based Models (34 topologies, 100% structural validity)

| Graph Model | Params | Structural Validity | Topologies at 100% |
|-------------|--------|--------------------|--------------------|
| VCG v5 (VAE) | 4.0M | 100.0% | 34/34 |
| CCFM v5 (Flow Matching) | 7.6M | 100.0% | 34/34 |

### ARCS vs AnalogGenie

| Dimension | AnalogGenie | ARCS |
|-----------|-------------|------|
| **Structural validity** | 93.2% (after PPO) | 96.6+/-0.5% (GRPO) / 100% (constrained) |
| **Sim validity** | N/A (no SPICE eval) | 53.1+/-3.1% (GRPO) / 99.9% (hybrid) |
| **Component values** | No (GA post-hoc) | Yes (in generation) |
| **Spec conditioning** | No | Yes |
| **Inference speed** | Minutes (GA sizing) | ~20ms |
| **Model size** | 11.8M params | ~6.8M params |
| **SPICE integration** | Post-hoc only | In-the-loop RL |

---

## 34 Circuit Topologies

- **Tier 1** (7 power converters): Buck, Boost, Buck-Boost, Cuk, SEPIC, Flyback, Forward
- **Tier 2** (9 signal circuits): Inverting/non-inverting/instrumentation/differential amplifiers, Sallen-Key LP/HP/BP filters, Wien bridge and Colpitts oscillators
- **Tier 2b** (18 extended): BJT amplifiers (CE/CC/CB/cascode/diff pair), current mirrors, regulators, Hartley/phase-shift oscillators, state-variable/twin-T filters, half bridge, push pull, charge pump, voltage doubler, zeta converter

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/tusharpathaknyu/ARCS.git
cd ARCS
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Requires ngspice for SPICE simulation
brew install ngspice  # macOS
# apt install ngspice  # Linux

# Quick demo
PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1 --simulate

# Interactive mode
PYTHONPATH=src python -m arcs.demo --interactive

# Full evaluation
PYTHONPATH=src python -m arcs.evaluate \
    --checkpoint checkpoints/arcs_graph_transformer/best_model.pt \
    --n-samples 160 --simulate

# RL with GRPO
PYTHONPATH=src python -m arcs.rl --checkpoint checkpoints/best_model.pt \
    --grpo --group-size 4 --n-topos-per-step 3

# Run tests
PYTHONPATH=src python -m pytest tests/ -v
```

Example output:
```
--- Design ---
  Topology:   buck
  Structure:  VALID
  Components (4):
    capacitor        = 38uF
    resistor         = 130mOhm
    mosfet_n         = 21mOhm
    inductor         = 300uH
  Inference:  470ms

  SPICE:      VALID
  Vout error: 5.0%
  Efficiency: 94.9%
  Ripple:     0.0027
  Reward:     6.36/8.0
```

---

## Paper

The paper is available at `paper/arcs_paper.tex`.

**Citation:**
```bibtex
@article{pathak2025arcs,
  title={ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning},
  author={Pathak, Tushar},
  year={2025}
}
```

---

## Key References

1. **AnalogGenie** — Gao et al., ICLR 2025. Autoregressive pin-level topology generation.
2. **CircuitSynth** — 2024. GPT-Neo fine-tuned on SPICE netlist text.
3. **AutoCircuit-RL** — 2025. CircuitSynth + GRPO RL fine-tuning.
4. **cktFormer** — IECON 2025. Transformer on graph tokens for analog circuits.
5. **FALCON** — NeurIPS 2025. GNN + layout-constrained analog co-optimization.

---

## License
