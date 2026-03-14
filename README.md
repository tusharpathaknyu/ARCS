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

### Model: GPT-style Decoder (6,504,960 params) — `baseline`
- d_model=256, n_layers=6, n_heads=4, d_ff=1024
- SwiGLU activation, RMSNorm (more modern than AnalogGenie's ReLU + LayerNorm)
- Token-type embeddings (spec vs. component vs. value)
- Weight-tied LM head

### Model: Two-Head Architecture (6,811,648 params) — `two_head`
- Shared transformer backbone, separate **structure head** (weight-tied) and **value head** (independent MLP with SiLU + residual)
- Value mask routes loss: structure tokens train the structure head, value tokens train the value head
- During generation, component-type tokens are sampled from the structure head; value tokens from the value head
- Motivation: structure prediction and value regression have different loss landscapes — decoupling them allows independent capacity

### Model: Graph Transformer — `graph_transformer` (6,829,296 params)
- Topology-aware causal attention: learned `adj_bias` (per-head scalar for circuit-adjacent pairs) + `edge_type_bias` (component-pair type embeddings, 17 edge-type buckets)
- Hardcoded `TOPOLOGY_ADJACENCY` tables for all 16 topologies define which components share circuit nets, derived from actual SPICE schematics
- **Random-Walk Positional Encoding (RWPE)**: K=8 walk lengths encode graph structure. For each topology, computes the transition matrix $T = D^{-1}A$ and extracts return probabilities $[T^1_{ii}, T^2_{ii}, \ldots, T^8_{ii}]$ per node. Projected via 2-layer MLP (8→64→256, GELU) and added to token embeddings. Precomputed at import time for all 16 topologies.
- Two-head output (structure + value) inherited from `TwoHeadARCSModel`
- Motivation: AnalogGenie encodes adjacency implicitly via pin-level Eulerian walks; ARCS uses component-level tokens, so adjacency must be **injected as structural attention bias**. RWPE gives each node a unique structural fingerprint — hub nodes (degree-2) have different return-probability profiles than leaf nodes (degree-1).

### Learned Reward Model (663K params)
- Bidirectional transformer encoder (d_model=128, 4 heads, 2 layers) → mean-pool → 2-layer MLP → scalar reward
- Trained on 53K circuits with SPICE-computed rewards; used for Best-of-N ranking at inference

### Best-of-N Inference Scaling
- Generate N candidate circuits, rank by reward model score, return the best
- Tests N ∈ {1, 3, 5, 10, 20, 50} — quality scales log-linearly with N
- Achieves +2.0–2.7% sim_valid improvement at N=3 for both SL and RL generators

Select model type via `--model-type {baseline,two_head,graph_transformer}` in training, RL, evaluation, and demo scripts.

### ValidCircuitGen: Constrained VAE (Direction 5) — `~4.0M params`

A fundamentally different architecture from the autoregressive models above. ValidCircuitGen (VCG) is a Variational Autoencoder that generates **entire circuit graphs in one shot** in continuous space, with differentiable constraint projection guaranteeing structural validity by construction.

**Architecture:**
```
Encoder:     4-layer Bidirectional Graph Transformer (d_model=256, 4 heads)
             + RWPE (K=8 walks) + Spec cross-attention → mu, logvar
Decoder:     3-layer MLP backbone → soft_X (node types) + soft_A (adjacency) + soft_V (values)
Projection:  20-step Adam optimizer on 5 differentiable constraints
Training:    Reconstruction + β-KL + Lagrangian constraint penalties with dual ascent
```

**Five Differentiable Circuit Constraints:**
| # | Constraint | What It Checks |
|---|-----------|----------------|
| C1 | No floating nodes | Every active node has degree ≥ 1 |
| C2 | Device completeness | Multi-pin devices (MOSFET, BJT) have degree ≥ 2 |
| C3 | No short circuits | No forbidden voltage-source edge patterns |
| C4 | Graph connectivity | Single connected component (Fiedler value > 0) |
| C5 | Value bounds | Component values within physical ranges |

**Key Idea — Constraint Projection:**
Even if the VAE produces a poor initial output, the projection step uses gradient descent (20 steps) to minimize constraint violations, steering the soft graph toward the feasible set. After projection, discretization (argmax + thresholding) produces the final circuit. If projection converges (violation < ε), validity is **guaranteed** to within ε tolerance.

**Complementary to Autoregressive Models:**
- Autoregressive ARCS: Sequential token generation with grammar-based masking → 100% structural validity
- ValidCircuitGen: One-shot graph generation with constraint projection → validity by construction
- Both achieve near-100% structural validity through different mechanisms

**Training Results** (100 epochs, 32,281 valid circuits, MPS/Apple Silicon):
- Train loss: 0.90, Val loss: 0.91
- Type reconstruction accuracy: 100%, Adjacency accuracy: 100%
- Value error: 0.083 log10 (~20% relative), Latent space smoothness: 0.992
- **100% structural validity** on all 16/16 topologies
- Topology-aware validity checks now pass across all templates
- Validity identical with and without constraint projection — the model itself learned valid generation

```bash
# Train VCG
PYTHONPATH=src python scripts/train_vcg.py --data data/combined --epochs 100

# Evaluate VCG
PYTHONPATH=src python scripts/evaluate_vcg.py \
    --vcg-checkpoint checkpoints/vcg/best_model.pt \
    --data data/combined --n-samples 160

# Compare VCG vs autoregressive ARCS
PYTHONPATH=src python scripts/evaluate_vcg.py \
    --vcg-checkpoint checkpoints/vcg/best_model.pt \
    --arcs-checkpoint checkpoints/arcs_graph_transformer/best_model.pt \
    --data data/combined --n-samples 160 -v
```

### Constrained Circuit Flow Matching (CCFM) — `~7.6M params`

A novel generative model that replaces VCG's VAE with Conditional Flow Matching (Lipman et al. 2023), augmented with differentiable constraint guidance during ODE sampling. **To our knowledge, this is the first application of flow matching to electronic circuit generation.**

**Architecture:**
```
┌────────────┐      ┌───────────────────┐      ┌──────────────┐
│ z_0 ~ N(0,I)│ ──→ │ FlowVelocityNet   │ ──→ │ z_1 (circuit) │
└────────────┘      │ 4-layer DiT       │      └──────┬───────┘
                    │ + AdaptiveLayerNorm│             │
┌────────────┐      │ + constraint      │      ┌──────▼───────┐
│ Spec cond. │ ──→ │   guidance ∇c     │      │ VCGDecoder   │
│ (cross-attn)│     └───────────────────┘      │ → circuit    │
└────────────┘                                  └──────────────┘
```

**Key innovations:**
- **DiT-style transformer blocks**: Adaptive Layer Norm conditioned on (time, spec) — scale/shift params predicted from conditioning, zero-initialized for stable training
- **Constraint-guided sampling**: During ODE integration, velocity is steered away from constraint violations: $v_{\\text{guided}} = v_\\theta(z_t, t, c) - \\lambda \\cdot \\nabla_z \\sum_i w_i \\cdot c_i(\\text{decode}(z_t))$
- **Learnable guidance weights**: Per-constraint softplus weights automatically balance constraint importance
- **Consistency regularization**: Additional loss term encouraging predicted $z_1$ endpoints to match targets
- **Reuses VCG infrastructure**: Encoder/decoder/constraints from VCG are frozen; only the flow network (3.7M params) is trained

**Training**: $L = \\mathbb{E}_{t,z_0,z_1} \\| v_\\theta(z_t, t, c) - (z_1 - z_0) \\|^2 + 0.1 \\cdot L_{\\text{consistency}}$

**Sampling**: 50-step Euler integration from noise to circuit, with constraint guidance after $t=0.3$

```bash
# Train CCFM (reuses VCG encoder/decoder, trains flow network)
PYTHONPATH=src python scripts/train_ccfm.py \
    --data data/combined --vcg-checkpoint checkpoints/vcg/best_model.pt
```

### GRPO: Group Relative Policy Optimization

Fixes the RL regression problem where power converters lose performance. Instead of a single global reward baseline, GRPO generates **groups** of circuits per topology and computes z-scored advantages within each group:

$$\\text{advantage}_i = \\frac{r_i - \\mu_{\\text{group}}}{\\sigma_{\\text{group}} + \\epsilon}$$

This prevents cross-topology interference: power converters (max reward ≈ 8) and signal circuits (max reward ≈ 3) each have their own normalization.

```bash
# RL with GRPO (recommended over vanilla REINFORCE)
PYTHONPATH=src python -m arcs.rl --checkpoint checkpoints/best_model.pt \
    --grpo --group-size 4 --n-topos-per-step 3
```

### Training Pipeline
1. **Supervised pre-training**: Next-token prediction on all circuit sequences, 5x value-token loss weight
2. **RL fine-tuning**: REINFORCE or GRPO with KL penalty, reward from SPICE simulation

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
| **ARCS Graph Transformer (SL)** | **6.83M** | **1** | **85.0%** | **71.9%** | **4.55/8.0** | **~0.02s** |
| ARCS Graph Transformer + RL | 6.83M | 1 | 86.9% | 55.0% | 4.35/8.0 | ~0.02s |
| ValidCircuitGen (VCG) | 4.0M | 1 | N/A† | 100.0%‡ | N/A† | ~0.01s |

†VCG generates in continuous graph space — no SPICE simulation integrated yet. ‡Structural validity: 100% on 16/16 topologies.

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
| **Model size** | 11.8M params | 6.83M params |
| **Architecture** | Standard GPT decoder | Graph Transformer with RWPE + topology-aware attention |
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
- [x] 273 tests across 14 test files

### Phase 6: Enhanced Model Training & Evaluation (complete)
- [x] Two-Head training: 100 epochs, best val_loss=0.954 (vs baseline 1.237)
- [x] Graph Transformer training: 100 epochs, best val_loss=0.990 (pre-RWPE)
- [x] Architecture comparison: Graph Transformer wins (sim_valid=71.9% vs 61.9% vs 45.6%)
- [x] RL fine-tuning of Graph Transformer: 5000 steps, best eval reward=7.468/8.0
- [x] Auto-detection of model type from checkpoint state dict keys

### Phase 7: Paper (complete)
- [x] ICCAD 2026 paper written: *ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning*
- [x] Full experimental evaluation: 3 architectures, 2 baselines (RS + GA), ablation studies
- [x] Honest comparison with AnalogGenie (ICLR 2025)

### Phase 8: RWPE Implementation (complete)
- [x] Replaced fake walk position embeddings with real Random-Walk Positional Encoding (K=8)
- [x] Precomputed RWPE for all 16 topologies with transition matrix powers
- [x] 2-layer MLP projection (8→64→256, GELU), +17.2K params
- [x] Updated paper §3.3, abstract, intro to match implementation

### Phase 9: RWPE Retraining & Evaluation (complete)
- [x] Retrained Graph Transformer with RWPE: 100 epochs, best val_loss=0.8718 (epoch 80)
- [x] Significant improvement over pre-RWPE (0.990 → 0.8718)
- [x] Structural accuracy: 91.1%, Value accuracy: 72.4%
- [x] RL ranking comparison: +2.0% SL, +2.1% RL at N=3 (Table 9 in paper)
- [x] Added --resume flag to training script with optimizer state loading

### Phase 10: Visualization & Documentation (complete)
- [x] Training curves dashboard: 6-panel plot (loss, perplexity, accuracy, LR, gen gap, time)
- [x] Algorithm architecture diagrams: 5 publication-quality figures (pipeline, transformer block, RWPE, constrained FSM, tokenization)
- [x] Full algorithm explanation document (`docs/ARCS_Algorithm_Explained.md`)

### Phase 11: ValidCircuitGen — Direction 5 (complete)
- [x] Constrained VAE architecture: GNN encoder + MLP decoder + constraint projection (~4.0M params)
- [x] 5 differentiable circuit constraints: floating nodes, device completeness, short circuits, connectivity, value bounds
- [x] Lagrangian training with dual ascent on adaptive multipliers
- [x] Spec-conditioned generation via cross-attention SpecEncoder
- [x] Bidirectional Graph Transformer encoder with RWPE and adjacency bias
- [x] Constraint projection: 20-step Adam optimization on soft graph logits
- [x] Full training script with KL annealing, cosine LR, WandB logging
- [x] Evaluation pipeline: validity, reconstruction, latent space smoothness, projection ablation
- [x] 47 tests across 14 test classes

### Phase 12: VCG Training & Evaluation (complete)
- [x] Fixed critical NaN bug: all-zero spec_mask → softmax over all -inf in cross-attention
- [x] Fixed MPS eigvalsh: CPU fallback for graph connectivity constraint (eigvalsh not on MPS)
- [x] Added 2 NaN regression tests (49 total tests, all passing)
- [x] Trained 100 epochs on 32,281 valid circuits (MPS, ~71s/epoch)
- [x] Best val_loss: 0.91, train_loss: 0.90
- [x] 100% type accuracy, 100% adjacency accuracy, 0.083 log10 value error
- [x] 100% structural validity on all 16/16 topologies
- [x] Latent space smoothness: 0.992 ± 0.009
- [x] Projection not needed: model itself learns valid generation

### Phase 13: Novel Architecture Improvements (complete)
- [x] **GRPO Per-Topology RL** — Group Relative Policy Optimization replacing global-baseline REINFORCE
  - Per-topology z-scored advantages: advantage = (r - μ_group) / (σ_group + ε)
  - Fixes RL regression where power converters (max reward ~8) got negative advantage against easy signal circuits
  - CLI: `--grpo --group-size 4 --n-topos-per-step 3`
  - 5 new tests (10 total RL tests), backward compatible with existing REINFORCE mode
- [x] **Constrained Circuit Flow Matching (CCFM)** — Novel generative model (3.7M flow network + 4M VCG)
  - Conditional flow matching replaces VAE's KL-constrained latent space with optimal-transport paths
  - DiT-style transformer blocks with adaptive layer norm for time + spec conditioning
  - Constraint-guided ODE sampling: velocity projected toward feasible circuit set during integration
  - Sinusoidal time embedding + per-constraint learnable guidance weights
  - 21 new tests covering components, integration, theory (flow paths, loss convergence)
- [x] **Hybrid Generation Pipeline** — Multi-source circuit generation with SPICE-based ranking
  - VCG→SPICE bridge: CircuitGraph → token sequence → DecodedCircuit → simulation → reward
  - CCFM→SPICE bridge: same pipeline with flow-matching-generated circuits
  - `HybridGenerator`: generates from ARCS + VCG + CCFM, ranks by simulated reward
  - `evaluate_generator()`: standardized evaluation benchmark for any circuit generator
  - 7 new tests
- [x] 355 total tests passing (82 new tests added)
- [x] **Training & Evaluation completed:**
  - CCFM: 100 epochs, final val_loss=0.1397 (from 0.68), ~25s/epoch on MPS
  - GRPO: 500 steps, best reward=1.578 (from 0.22), ~7s/step with 12 SPICE sims
  - Extended GRPO: 3000 additional steps (3500 total), best eval reward=5.235, 76.7% sim_valid
  - **GRPO (500 steps) is best autoregressive model**: 73.8% sim_valid (vs 65.0% SL, 50.0% RL), reward 4.701

### Phase 15: Hybrid Pipeline Reliability Fixes (complete)
- [x] Fixed hybrid graph→token topology aliasing for Sallen-Key (`TOPO_SALLEN_KEY_LP/HP/BP`)
- [x] Added topology-aware repair fallback in hybrid generation (reference adjacency + component types)
- [x] Added regression test for Sallen-Key topology token aliases
- [x] Hybrid benchmark script added: `scripts/evaluate_hybrid.py`
- [x] **Hybrid benchmark (n=4 candidates/source, VCG+CCFM ranking):**
  - Structural validity: **100.0%** (16/16 topologies)
  - Simulation success: **100.0%**
  - Simulation validity: **100.0%**
  - Mean reward: **6.225**

**Evaluation Results (80 circuits, SPICE simulation):**

| Model | Params | Struct% | SimOK% | SimValid% | Reward | Eff% | Verr% |
|-------|--------|---------|--------|-----------|--------|------|-------|
| ARCS-SL (GraphTransformer) | 6.8M | 86.2 | 77.5 | 65.0 | 4.229 | 60.7 | 24.9 |
| ARCS-RL (REINFORCE) | 6.8M | 100.0 | 83.8 | 50.0 | 4.007 | 58.1 | 40.0 |
| **ARCS-GRPO (500 steps)** | **6.8M** | **91.2** | **86.2** | **73.8** | **4.701** | **74.8** | **29.5** |
| ARCS-GRPO (3500 steps) | 6.8M | 92.5 | 82.5 | 66.2 | 4.683 | 74.0 | 28.2 |

| Graph Model | Params | Structural Validity | Topologies at 100% |
|-------------|--------|--------------------|--------------------|
| VCG (VAE) | 4.0M | 100.0% | 16/16 |
| CCFM (Flow Matching) | 7.6M | 100.0% | 16/16 |

---

## Setup

```bash
# Clone
git clone https://github.com/tusharpathaknyu/ARCS.git
cd ARCS

# Environment
python -m venv .venv
source .venv/bin/activate
pip install -e .   # installs torch, numpy, networkx, pandas, matplotlib, tqdm, pyyaml, pyspice

# Requires ngspice for SPICE simulation
brew install ngspice  # macOS
# apt install ngspice  # Linux

# Quick demo
PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1 --simulate

# Full evaluation (best model: Graph Transformer supervised)
PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/arcs_graph_transformer/best_model.pt --n-samples 160 --simulate

# Compare all architectures
PYTHONPATH=src python scripts/compare_architectures.py --n-samples 160 -v

# Evaluate topology-aware ablations (baseline vs topology-head vs family-MoE)
PYTHONPATH=src python scripts/evaluate_topology_ablation.py --n-samples 48

# End-to-end short run: train new variants from baseline + evaluate
bash scripts/run_topology_ablation_short.sh

# End-to-end medium run (defaults: EPOCHS=8, N_SAMPLES=80)
bash scripts/run_topology_ablation_medium.sh

# Best-of-N inference scaling
PYTHONPATH=src python scripts/run_bestofn.py --checkpoint checkpoints/arcs_graph_transformer/best_model.pt --simulate

# Training with resume support
PYTHONPATH=src python -m arcs.train --data data/combined --config small --model-type graph_transformer --epochs 100 --resume checkpoints/arcs_graph_transformer/best_model.pt

# Ablation studies
PYTHONPATH=src python scripts/run_ablations.py --n-samples 160

# Baselines
PYTHONPATH=src python -m arcs.baselines --method ga --n-repeats 10

# Run all 273 tests
PYTHONPATH=src python -m pytest tests/ -v
```

### Topology Ablation Snapshot (Graph Transformer)

| Run | Model | Structural | Sim Valid | Reward |
|-----|-------|------------|-----------|--------|
| Short (5 epochs, 48 samples) | Baseline | 75.0% | 60.4% | 3.890 |
| Short (5 epochs, 48 samples) | + Topology Value Heads | **87.5%** | **72.9%** | **4.495** |
| Short (5 epochs, 48 samples) | + Family MoE | 83.3% | 60.4% | 4.031 |
| Medium (8 epochs, 80 samples) | Baseline | **82.5%** | **60.0%** | **3.961** |
| Medium (8 epochs, 80 samples) | + Topology Value Heads | 81.2% | 58.8% | 3.923 |
| Medium (8 epochs, 80 samples) | + Family MoE | 77.5% | 57.5% | 3.695 |

Artifacts:
- `results/topology_ablation_short.json`
- `results/topology_ablation_medium.json`

---

## Consistency Automation

Phase 16 metric consistency is now checked automatically in CI and can be enforced locally before push.

```bash
# Run checker manually
python scripts/check_phase16_consistency.py

# Install local pre-push hook (one-time per clone)
bash scripts/install_git_hooks.sh
```

CI workflow: `.github/workflows/phase16-consistency.yml`

---

## Visualizations

Generated figures live in `results/`:

- **`results/training_curves.png`** — 6-panel dashboard: loss curves, perplexity, accuracy breakdown (structure vs. value), learning rate schedule, generalization gap, time per epoch
- **`results/algorithm_figures/`** — 5 publication-quality architecture diagrams:
  - `1_pipeline.png` — End-to-end system flow
  - `2_transformer_block.png` — GraphTransformer internal architecture
  - `3_rwpe.png` — RWPE computation walkthrough (Buck converter example)
  - `4_constrained_gen.png` — Grammar FSM + 3 constraint levels
  - `5_tokenization.png` — Log-scale value discretization + vocabulary layout

Regenerate:
```bash
PYTHONPATH=src python scripts/visualize_training.py
PYTHONPATH=src python scripts/visualize_algorithm.py
```

---

## Documentation

- **[`docs/ARCS_Algorithm_Explained.md`](docs/ARCS_Algorithm_Explained.md)** — Comprehensive algorithm explanation with full glossary of ML and circuit design terms, step-by-step walkthrough of tokenization, RWPE, attention, constrained generation, and a complete Buck converter example
- **[`paper/arcs_paper.tex`](paper/arcs_paper.tex)** — ICCAD 2026 submission

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
