# ARCS: Autoregressive Representation for Circuit Synthesis

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

## Our Approach

### Core Idea

```
INPUT:  "Vout=5V, Vin=12V, Iout=1A, fsw=100kHz"

MODEL GENERATES (token by token, ~20ms):
  START -> TOPO_BUCK -> SEP -> SPEC_VIN 12.19 -> SPEC_VOUT 4.90 -> ... -> SEP ->
  CAPACITOR 38uF -> RESISTOR 130mOhm -> MOSFET_N 21mOhm -> INDUCTOR 300uH -> END

OUTPUT: Component list + topology -> SPICE netlist -> simulate -> validate
```

### What Makes This Novel

1. **Native Component Tokens** — Each token IS a circuit element with value, not a text character
2. **Value-Aware Generation** — Component values embedded in the token representation (not a separate post-hoc sizing step)
3. **Spec Conditioning** — Performance requirements as prefix, model generates circuits meeting those specs
4. **SPICE-in-the-Loop RL** — Simulation reward during RL fine-tuning for physics grounding
5. **Automated Data Pipeline** — SPICE templates generate unlimited training data (no manual curation)
6. **Amortized Inference** — Single forward pass (~20ms) vs. search baselines (1-5 min per design)

---

## Data Generation

### SPICE IS the Data Engine

No scraping PDFs or textbooks. No manual effort. Parameterized templates + random sweeps + ngspice.

```
For each topology (Buck, Boost, Flyback, Sallen-Key, Colpitts, ...):
    For each random sample of component values within physical bounds:
        1. Plug values into SPICE netlist template
        2. Run ngspice simulation (~1-3 sec)
        3. Extract performance metrics (efficiency, ripple, gain, BW, ...)
        4. Store: (topology, component values, performance metrics)
        5. Both valid and invalid results kept for training
```

### Current Dataset

| Phase | Topologies | Generated | Valid | Augmented |
|-------|-----------|-----------|-------|-----------|
| Tier 1: Power converters | 7 | 35,000 | 16,400 | 82,000 |
| Tier 2: Amplifiers, filters, oscillators | 9 | 18,000 | 15,842 | 79,210 |
| **Combined** | **16** | **53,000** | **32,242** | **~161,000** |

---

## Circuit Types Implemented

### Tier 1: Power Electronics
- Buck, Boost, Buck-Boost, Flyback, Forward, Cuk, SEPIC
- **Metrics**: efficiency, output ripple, Vout accuracy

### Tier 2: Analog Signal Processing
- Op-amp circuits: inverting, non-inverting, instrumentation, differential
- Active filters: Sallen-Key lowpass, highpass, bandpass
- Oscillators: Wien bridge, Colpitts
- **Metrics**: gain, bandwidth, oscillation amplitude

---

## Architecture

### Tokenizer (686 tokens)

```
Component tokens:  MOSFET_N, MOSFET_P, RESISTOR, CAPACITOR, INDUCTOR,
                   DIODE, BJT_NPN, BJT_PNP, OPAMP, TRANSFORMER, ...

Value tokens:      500 bins on log scale (1e-12 to 1e7)

Topology tokens:   TOPO_BUCK, TOPO_BOOST, ..., TOPO_COLPITTS (16 total)

Spec tokens:       SPEC_VIN, SPEC_VOUT, SPEC_IOUT, SPEC_FSW, SPEC_GAIN, ...

Special tokens:    START, END, PAD, SEP, INVALID
```

### Model: GPT-style Decoder (6.5M params) — `baseline`
- d_model=256, n_layers=6, n_heads=4, d_ff=1024
- SwiGLU activation, RMSNorm (more modern than AnalogGenie's ReLU + LayerNorm)
- Token-type embeddings (spec vs. component vs. value)
- Weight-tied LM head

### Model: Two-Head Architecture — `two_head`
- Shared transformer backbone, separate **structure head** (weight-tied) and **value head** (independent MLP with SiLU + residual)
- Value mask routes loss: structure tokens train the structure head, value tokens train the value head
- During generation, component-type tokens are sampled from the structure head; value tokens from the value head
- Motivation: structure prediction and value regression have different loss landscapes — decoupling them allows independent capacity

### Model: Graph Transformer — `graph_transformer`
- Topology-aware causal attention: learned `adj_bias` (per-head scalar for circuit-adjacent pairs) + `edge_type_bias` (component-pair type embeddings)
- Hardcoded `TOPOLOGY_ADJACENCY` tables for all 16 topologies define which components share circuit nets, derived from actual SPICE schematics
- Walk position embeddings (32 positions) encode component order within the circuit body
- Two-head output (structure + value) inherited from `TwoHeadARCSModel`
- Motivation: AnalogGenie encodes adjacency implicitly via pin-level Eulerian walks; ARCS uses component-level tokens, so adjacency must be **injected as structural attention bias**

Select model type via `--model-type {baseline,two_head,graph_transformer}` in training, RL, evaluation, and demo scripts.

### Training Pipeline
1. **Supervised pre-training**: Next-token prediction on all circuit sequences, 5x value-token loss weight
2. **RL fine-tuning**: REINFORCE with KL penalty + baseline, reward from SPICE simulation

---

## Results

### Full Comparison (160 conditioned specs, all 16 topologies)

| Method | Params | Sims/Design | Sim Success | Sim Valid | Avg Reward | Wall Time/Design |
|--------|--------|-------------|-------------|-----------|------------|------------------|
| Random Search (N=200) | — | 200 | 100.0% | 81.2% | 7.28/8.0 | 58.8s |
| Genetic Algorithm (30×20) | — | 630 | 100.0% | 80.0% | 7.48/8.0 | 271.2s |
| ARCS Baseline (SL) | 6.5M | 1 | 66.2% | 45.6% | 3.38/8.0 | ~0.02s |
| ARCS Baseline + RL | 6.5M | 1 | 71.2% | 55.0% | 3.64/8.0 | ~0.02s |
| ARCS Two-Head (SL) | 6.8M | 1 | 79.4% | 61.9% | 4.23/8.0 | ~0.02s |
| **ARCS Graph Transformer (SL)** | **6.8M** | **1** | **85.0%** | **71.9%** | **4.55/8.0** | **~0.02s** |
| ARCS Graph Transformer + RL | 6.8M | 1 | 86.9% | 55.0% | 4.35/8.0 | ~0.02s |

**Key insight**: ARCS trades per-design optimality for **amortized speed** — a single
20ms forward pass vs. 200-630 SPICE simulations (1-5 min). This is **2,941-13,560x
faster** at inference. The baselines also have an unfair advantage: they search directly
in parameter space with the correct topology and component count given, while ARCS must
predict everything from scratch using only the target specification.

**Architecture progression**: Each enhancement delivers measurable gains:
- Two-Head: Decoupling structure/value heads → +16 pp sim_valid over baseline
- Graph Transformer: Topology-aware attention → +10 pp sim_valid over Two-Head
- RL fine-tuning: Achieves 100% structural validity and highest sim_success (86.9%), but hurts sim_valid on power converter topologies due to reward distribution mismatch

### Per-Topology Highlights (Graph Transformer, best model)

| Topology | SL Sim Valid | RL Sim Valid | Key Metric (RL) |
|----------|-------------|-------------|------------------|
| Buck | 100% | 10% | verr=6.7% |
| Boost | 90% | 0% | verr=61.2% |
| Buck-Boost | 80% | 50% | verr=46.3% |
| SEPIC | 60% | 70% | verr=13.7%, eff=71% |
| Cuk | 60% | 40% | verr=81.5% |
| Flyback | 40% | 10% | verr=49.2% |
| Forward | 70% | 20% | verr=26.2%, eff=73% |
| Inverting Amp | 90% | 100% | gain=-22.7dB |
| Non-inverting Amp | 100% | 100% | gain=-5.2dB |
| Instrumentation Amp | 100% | 100% | gain=4.5dB |
| Differential Amp | 100% | 100% | gain=-16.4dB |
| Sallen-Key LP | 50% | 70% | gain=-52.2dB |
| Sallen-Key HP | 70% | 60% | gain=-37.0dB |
| Sallen-Key BP | 60% | 70% | gain=-24.6dB |
| Wien Bridge | 0% | 70% | oscillating |
| Colpitts | 80% | 10% | oscillating |

**Observation**: RL dramatically improves signal circuits (Wien Bridge: 0%→70%) but hurts some power converters (Buck: 100%→10%). The reward function favors topologies where sim convergence is easier.

### Honest Assessment: ARCS vs AnalogGenie

| Dimension | AnalogGenie | ARCS (Graph Transformer) |
|-----------|-------------|------|
| **Validity** | 93.2% (after PPO) | 71.9% (SL) / 55.0% (RL) |
| **Sim success** | N/A (no SPICE eval) | 85.0% (SL) / 86.9% (RL) |
| **Topology diversity** | 3,502 unique circuits | 16 template topologies |
| **Component values** | No (GA post-hoc) | Yes (in generation) |
| **Spec conditioning** | No | Yes |
| **Inference speed** | Minutes (GA sizing) | ~20ms |
| **Data source** | IEEE papers (manual) | Automated SPICE |
| **Model size** | 11.8M params | 6.8M params |
| **Architecture** | Standard GPT decoder | Graph Transformer with topology-aware attention |
| **SPICE integration** | Post-hoc only | In-the-loop RL |
| **Venue** | ICLR 2025 Spotlight | In progress |

**Where ARCS wins**: Joint topology+values+specs in one shot, 3 orders of magnitude faster inference, fully automated data pipeline, SPICE-in-the-loop training, topology-aware attention biases.

**Where AnalogGenie wins**: Higher validity rate, much larger topology diversity, IC-level circuits, novel topology discovery, published at top venue.

**Positioning**: ARCS solves a fundamentally different problem — **amortized single-shot design** vs. **iterative search-based optimization**. We sacrifice per-design optimality for zero-cost inference, targeting rapid prototyping and design space exploration.

---

## Constrained Decoding (Phase 8)

**100% structural validity by construction** — without RL, without post-hoc filtering.

Constrained decoding applies a grammar-based token mask at each autoregressive step, restricting the model's output distribution to only structurally valid next tokens. This guarantees correct circuit structure regardless of model quality.

### Three Constraint Levels

| Level | What It Enforces | Validity |
|-------|-----------------|----------|
| **GRAMMAR** | Correct COMP→VAL alternation + END termination | 100% structural |
| **TOPOLOGY** | + correct component types and counts per topology | 100% + component correct |
| **FULL** | + value tokens within physically valid ranges per parameter | 100% + physically valid |

### How It Works

At each decode step after the conditioning prefix (START→TOPO→SEP→specs→SEP), the constraint mask:

1. **Component positions**: Masks all tokens except valid COMP_* types still needed for the topology
2. **Value positions**: Masks all tokens except VAL_* tokens (optionally restricted to valid parameter ranges)
3. **Completion**: Forces END after all expected components are placed

```python
from arcs.constrained import ConstrainedGenerator, ConstraintLevel

gen = ConstrainedGenerator(model, tokenizer, level=ConstraintLevel.FULL)
output = gen.generate(prefix, topology="buck")
# Guaranteed: 4 components (INDUCTOR, CAPACITOR, RESISTOR, MOSFET_N) + END
```

### Comparison Results (random-init model, 50 samples)

| Level | Valid% | CompOK% | Avg Components | Time (ms) |
|-------|--------|---------|----------------|-----------|
| NONE | 0.0% | 0.0% | 0.0 | 280 |
| GRAMMAR | 100.0% | 0.0% | 20.6 | 138 |
| TOPOLOGY | 100.0% | 100.0% | 4.3 | 27 |
| FULL | 100.0% | 100.0% | 4.3 | 25 |

Constrained decoding is also **faster** — fewer tokens generated (no wasted invalid sequences), and the mask eliminates computation on impossible branches.

### Evaluation with Constraints

```bash
# Evaluate with constrained decoding (level 2 = TOPOLOGY)
PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/best_model.pt --constrained 2

# Compare all levels
PYTHONPATH=src python scripts/compare_constrained.py --checkpoint checkpoints/best_model.pt

# With SPICE simulation
PYTHONPATH=src python scripts/compare_constrained.py --checkpoint checkpoints/best_model.pt --simulate
```

### Lagrangian Constraint Loss (Training)

For supervised training, a differentiable Lagrangian constraint loss penalizes violations with adaptive multipliers:

```python
from arcs.constrained import LagrangianConstraintLoss

loss_fn = LagrangianConstraintLoss(tokenizer)
constraint_loss, stats = loss_fn(logits, targets, input_ids, topologies)
total_loss = ce_loss + constraint_loss
```

---

## Demo

```bash
# Quick buck converter design (~20ms)
PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1

# With SPICE validation
PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1 --simulate

# Multiple candidates ranked by reward
PYTHONPATH=src python -m arcs.demo --topology boost --vin 5 --vout 12 --iout 0.5 -n 5 --simulate

# Interactive mode
PYTHONPATH=src python -m arcs.demo --interactive
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

## Roadmap

### Phase 1: Data Generation (complete)
- [x] 7 parameterized SPICE templates for power converters
- [x] Data generation pipeline: random sweep -> ngspice -> extract metrics
- [x] 35K generated, 16.4K valid across 7 topologies
- [x] 686-token vocabulary (components + values + specs + topologies)
- [x] Data augmentation via component shuffle (5x factor)

### Phase 2: Model Training (complete)
- [x] 6.5M param GPT decoder (SwiGLU, RMSNorm, token-type embeddings)
- [x] Trained on 175K augmented sequences, 100 epochs (best val_loss=1.279)
- [x] Spec conditioning via prefix tokens
- [x] 100% conditioned structural validity, 77.1% unconditioned

### Phase 3: SPICE-in-the-Loop RL (complete)
- [x] REINFORCE with KL penalty + baseline
- [x] Reward function: structure + convergence + vout_accuracy + efficiency + ripple (max 8.0)
- [x] RL v1: 5000 steps, best reward 7.02 but validity crashed to 22%
- [x] RL v2: adaptive KL + struct_bonus, stable 96-100% validity

### Phase 4: Tier 2 Expansion (complete)
- [x] 9 new templates: amplifiers, Sallen-Key filters, oscillators
- [x] 18K Tier 2 samples (88% yield)
- [x] Combined retraining: val_loss=1.237
- [x] Simulation-based evaluation for all 16 topologies

### Phase 5: Baselines, Ablations, Demo (complete)
- [x] Baselines: Random Search (200 trials) + GA (pop=30, 20 gens)
- [x] Efficiency metric fix: sign-correct current measurement
- [x] Ablation studies: RL effect, spec conditioning effect, data expansion effect
- [x] Demo CLI: interactive circuit design tool

### Phase 5b: Enhanced Architectures (complete)
- [x] Two-head model: separate structure head (weight-tied) + value head (SiLU MLP)
- [x] Graph transformer: topology-aware causal attention with adjacency bias for all 16 topologies
- [x] Model factory: `create_model` / `load_model` with automatic type detection from checkpoints
- [x] Unified `--model-type` flag across train, RL, evaluate, and demo scripts
- [x] 30 new tests for enhanced architectures (96 total)

### Phase 6: Enhanced Model Training & Evaluation (complete)
- [x] Two-Head training: 100 epochs, best val_loss=0.954 (vs baseline 1.237)
- [x] Graph Transformer training: 100 epochs, best val_loss=0.990
- [x] Architecture comparison: Graph Transformer wins (sim_valid=71.9% vs 61.9% vs 45.6%)
- [x] RL fine-tuning of Graph Transformer: 5000 steps, best eval reward=7.468/8.0
- [x] Auto-detection of model type from checkpoint state dict keys

### Phase 7: Paper (next)
- [ ] Write paper targeting DAC / ICCAD / NeurIPS workshop

---

## Setup

```bash
# Clone
git clone https://github.com/tusharpathaknyu/ARCS.git
cd ARCS

# Environment
python -m venv .venv
source .venv/bin/activate
pip install torch numpy

# Requires ngspice
brew install ngspice  # macOS
# apt install ngspice  # Linux

# Quick demo
PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1 --simulate

# Full evaluation (best model: Graph Transformer supervised)
PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/arcs_graph_transformer/best_model.pt --n-samples 160 --simulate

# Compare all architectures
PYTHONPATH=src python scripts/compare_architectures.py --n-samples 160 -v

# Ablation studies
PYTHONPATH=src python scripts/run_ablations.py --n-samples 160

# Baselines
PYTHONPATH=src python -m arcs.baselines --method ga --n-repeats 10
```

---

## Key References

1. **AnalogGenie** — Gao et al., ICLR 2025. Autoregressive pin-level topology generation. [Paper](https://arxiv.org/abs/2503.00205) | [Code](https://github.com/xz-group/AnalogGenie)
2. **CircuitSynth** — 2024. GPT-Neo fine-tuned on SPICE netlist text.
3. **AutoCircuit-RL** — 2025. CircuitSynth + GRPO RL fine-tuning.
4. **cktFormer** — IECON 2025. Transformer on graph tokens for analog circuits.
5. **FALCON** — NeurIPS 2025. GNN + layout-constrained analog co-optimization.
6. **CktGNN** — Dong et al., 2023. Graph VAE for Op-Amp topology generation.
7. **LaMAGIC** — Chang et al., 2024. Masked language model for power converter topology.
8. **AutoCkt** — Berkeley, DATE 2020. DRL for analog circuit sizing.

---

## License

MIT
