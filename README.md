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
| Tier 2b: BJT, regulators, power (Phase 14+17) | 18 | 27,000 | 27,000 | 135,000 |
| **Combined v2 (re-validated)** | **34** | **86,000** | **63,180** | **~297,000** |

Phase 17 balanced the dataset: all Tier-2b topologies expanded from 500→2000 samples each to eliminate mode collapse in VCG training. Phase 22 re-validated all samples with corrected per-topology validation (18 topologies previously fell through to `len(metrics)>0`). Note: zeta_converter has 0 valid samples (vout_error consistently >86%) and wien_bridge has only 54 valid (oscillation hard to achieve with random sweeps) — both need targeted data generation.

---

## Circuit Types Implemented (34 Topologies)

### Tier 1: Power Electronics (7)
- Buck, Boost, Buck-Boost, Flyback, Forward, Cuk, SEPIC
- **Metrics**: efficiency, output ripple, Vout accuracy

### Tier 2: Analog Signal Processing (9)
- Op-amp circuits: inverting, non-inverting, instrumentation, differential
- Active filters: Sallen-Key lowpass, highpass, bandpass
- Oscillators: Wien bridge, Colpitts
- **Metrics**: gain, bandwidth, oscillation amplitude

### Tier 2b: BJT, Regulators & Extended Power (18, Phase 14+17)
- BJT amplifiers: common emitter, common collector, common base, cascode, differential pair
- Current sources: current mirror, Wilson current mirror
- Op-amp topologies: folded cascode, telescopic cascode, two-stage opamp, rail-to-rail
- Regulators: series regulator, shunt regulator
- Oscillators: Hartley, phase shift
- Filters: state variable, twin-T notch
- Signal processing: inverting summing amp, transimpedance amp
- Extended power: half bridge, push pull, charge pump, voltage doubler, zeta converter
- **Metrics**: topology-specific (gain, bandwidth, regulation, THD, etc.)

---

## Architecture

### Tokenizer (706 tokens)

```
Component tokens:  MOSFET_N, MOSFET_P, RESISTOR, CAPACITOR, INDUCTOR,
                   DIODE, BJT_NPN, BJT_PNP, OPAMP, TRANSFORMER, ...

Value tokens:      500 bins on log scale (1e-12 to 1e7)

Topology tokens:   TOPO_BUCK, TOPO_BOOST, ..., TOPO_ZETA_CONVERTER (34 total)

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
- Hardcoded `TOPOLOGY_ADJACENCY` tables for all 34 topologies define which components share circuit nets, derived from actual SPICE schematics
- **Random-Walk Positional Encoding (RWPE)**: K=8 walk lengths encode graph structure. For each topology, computes the transition matrix $T = D^{-1}A$ and extracts return probabilities $[T^1_{ii}, T^2_{ii}, \ldots, T^8_{ii}]$ per node. Projected via 2-layer MLP (8→64→256, GELU) and added to token embeddings. Precomputed at import time for all 34 topologies.
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

**Training Results — VCG v5** (100 epochs, improved dataset with 63K circuits across 34 topologies, MPS/Apple Silicon):
- Val loss: 1.07 (expanded topology set)
- **100% structural validity** on all 34/34 topologies
- Topology-aware validity checks pass across all 34 templates
- Trained on balanced dataset (all topologies ≥2000 samples) to eliminate mode collapse

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

### Publication Evaluation (50 samples × 32 topologies, bootstrap 95% CI)

| Method | SPICE Evals | Reward (95% CI) | Sim Valid% | Wall Time |
|--------|-------------|-----------------|------------|-----------|
| Random Search | 1 | 5.18 [5.05, 5.31] | 91.4% | ~0.3s |
| Genetic Algorithm | 320 | 7.57 [7.53, 7.60] | 100.0% | ~120s |
| VCG v5 (alone) | 1 | 5.34 [5.27, 5.42] | 94.0% | ~0.02s |
| CCFM v5 (alone) | 1 | 5.51 [5.44, 5.58] | 95.7% | ~0.16s |
| **Hybrid v5 (VCG+CCFM)** | **8** | **6.43 [6.38, 6.48]** | **99.9%** | **~0.05s** |

All pairwise comparisons significant at p < 0.001 (Wilcoxon signed-rank, n=32 topologies).

**Key findings**:
- **At equal compute (1 SPICE sim)**: Learned models (VCG: 5.34, CCFM: 5.51) beat random sampling (5.18) with 94-96% vs 91% sim validity
- **Hybrid ranking** over 8 candidates achieves **99.9% sim validity** and 6.43 reward — competitive with GA's 7.57 using **40× fewer SPICE evaluations**
- **Ablation**: VCG alone (5.34) → +CCFM diversity (5.51) → +hybrid ranking (6.43) — each component adds measurable value
- **Spec-aware reward**: Signal circuits now measured on gain/cutoff/frequency accuracy vs target specs, not just functional correctness

**Architecture progression** (unified evaluation, 160 samples, 5 seeds):
- Baseline GPT: 40.1±0.7% sim_valid, 86.0±1.9% struct, reward 3.37±0.06
- Two-Head: 50.4±2.2% sim_valid (+10.3 pp over baseline)
- Graph Transformer SL: 45.4±3.0% sim_valid (+5.3 pp over baseline)
- **GT + GRPO (500 steps)**: 53.1±3.1% sim_valid (+9.6 pp over REINFORCE)
- VCG/CCFM: 100% structural validity by construction
- Hybrid ranking: 99.9% sim validity, reward 6.43 [6.38, 6.48]

### Honest Assessment: ARCS vs AnalogGenie

| Dimension | AnalogGenie | ARCS |
|-----------|-------------|------|
| **Structural validity** | 93.2% (after PPO) | 96.6±0.5% (GRPO) / 100% (constrained) |
| **Sim validity** | N/A (no SPICE eval) | 53.1±3.1% (GRPO) / 99.9% (hybrid) |
| **Topology diversity** | 3,502 unique circuits | 32 template topologies |
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
- [x] 751 tests across 18 test files (Phase 17: +278 parametric; Phase 18: +18 smoke tests)

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
  - Structural validity: **100.0%** (34/34 topologies)
  - Simulation success: **100.0%**
  - Simulation validity: **100.0%**
  - Mean reward: **6.225**

### Phase 16: Reward Routing Fixes (complete)
- [x] Fixed reward computation routing for power topologies (charge_pump, voltage_doubler, half_bridge, push_pull, zeta_converter)
- [x] Added dedicated reward functions for regulators and current mirrors
- [x] CI consistency check and pre-push hook

### Phase 17: Production Hardening & Balanced Retraining (complete)
- [x] Removed hardcoded ngspice path — `NGSPICE_PATH` env var with auto-detection fallback
- [x] Pinned dependency versions in pyproject.toml (upper bounds added)
- [x] Added pytest configuration with consistent test discovery
- [x] 278 parametric integration tests covering all 34 topologies
- [x] Generated 1500 additional samples for 18 undersampled topologies (500→2000)
- [x] Merged balanced dataset: `data/combined_v2/` with 34 topologies
- [x] Retrained VCG v5: val_loss=0.834, 100% validity on 34/34 topologies
- [x] Retrained CCFM v5: val_loss=0.094, 100% validity on 34/34 topologies
- [x] Retrained latent reward v5: val_loss=0.251, corr=0.94
- [x] Deduplicated rl.py, improved error handling (bare excepts → specific types)
- [x] Marked legacy scripts deprecated
- [x] 733 total tests passing

**Evaluation Results (Phase 17 — 340 circuits across 34 topologies, SPICE simulation):**

| Model | Params | Struct% | SimOK% | SimValid% | Reward |
|-------|--------|---------|--------|-----------|--------|
| ARCS-SL (GraphTransformer) | 6.8M | 86.2 | 77.5 | 65.0 | 4.229 |
| ARCS-RL (REINFORCE) | 6.8M | 100.0 | 83.8 | 50.0 | 4.007 |
| **ARCS-GRPO (500 steps)** | **6.8M** | **91.2** | **86.2** | **73.8** | **4.701** |
| **Hybrid v5 (VCG+CCFM, 32 topos)** | **~12M** | **100.0** | **97.1** | **99.9** | **6.33** |

| Graph Model | Params | Structural Validity | Topologies at 100% |
|-------------|--------|--------------------|--------------------|
| VCG v5 (VAE) | 4.0M | 100.0% | 34/34 |
| CCFM v5 (Flow Matching) | 7.6M | 100.0% | 34/34 |

### Phase 20: 34-Topology Retraining & Evaluation (complete)
- [x] Retrained reward model v2 on 89K samples: r=0.986, val_mae=0.113
- [x] Retrained Graph Transformer v2 on 34-topology dataset (89K samples, 50 epochs)
  - Best val_loss=1.957 at epoch 19, accuracy=71.0%, struct_acc=90.7%
- [x] GRPO reinforcement learning (200 steps): reward 2.10→3.74, struct 62%→92%
- [x] Re-ran RS/GA baselines on all 34 topologies: RS=7.32, GA=7.41 avg reward
- [x] Fixed baseline reward routing bug: 8 topologies (regulators, extended power, current mirror) were missing reward functions in baselines.py — rewards jumped from 2.0 to 5.0-8.0
- [x] Trained latent reward predictor v2 on VCG-encoded data
- [x] Added hartley to multi-component topology tests
- [x] 751 tests passing across 18 test files

### Phase 21: Inference-Time Scaling & SPICE Evaluation (complete)
- [x] VCG v5: 100% structural validity across all 34 topologies, avg SPICE reward 5.28
- [x] GT v2 Best-of-N scaling with SPICE simulation:
  - N=1: 100% struct valid, 70.6% sim valid, reward 4.97
  - N=5: 100% struct valid, 70.6% sim valid, reward 5.04
  - N=10: 100% struct valid, 79.4% sim valid, reward 5.14 (+8.8pp sim_valid)
- [x] Diagnosed and fixed latent reward predictor bug: v2 trained on proxy reward (bounds adherence ≈8.0, corr≈0) instead of actual SPICE rewards
- [x] Trained latent reward predictor v5 with real SPICE rewards: val_loss=0.251, val_corr=0.94 (vs v2's corr≈0.001)
- [x] VCG + latent refinement (v5): avg reward 5.28→5.42 (+3%), top gains on state_variable_filter (+1.67), series_regulator (+1.33), transimpedance_amp (+1.26)
- [x] Fixed train_latent_reward.py to use RewardGraphDataset with actual SPICE rewards

### Phase 22: Full Codebase Audit & Pre-Retrain Fixes (complete)
- [x] **Datagen validation overhaul**: 18/27 Tier-2 topologies had fallthrough to `len(metrics)>0`
  - Extended power topos (half_bridge, push_pull, etc.) now routed through `_is_valid_power`
  - BJT amps, filters, oscillators, current mirrors all get proper topology-specific validation
  - Data re-labeled: 67,456→59,360→63,180 valid samples after template fixes
  - **Zeta converter**: Fixed netlist topology (L1/diode placement) — 0→1,218 valid (61%)
  - **Wien bridge**: Fixed opamp (rail-limited, removed shorting Vkick) — 54→1,306 valid (65%)
  - **Phase shift**: Fixed opamp + kick injection — 1,251→1,181 valid (59%, now real oscillation)
  - Derived metrics (efficiency, vout_error_pct) re-computed for 5 extended power topologies
- [x] **Reward function fixes**:
  - 6 IC-level opamps + differential_pair added to `_signal_reward` amp_types (were getting reward=0)
  - `wilson_current_mirror` added to mirror reward routing
  - `_get_spec_to_cond()` fixed to check all power topos (not just Tier 1)
  - Amplifier gain: topology-aware sign check (inverting vs non-inverting)
  - Filter gain capped at +20dB, oscillation threshold raised to 100mV
  - Oscillator frequency: checks f_osc/frequency/f_peak (was unreachable)
- [x] **RL training fixes**:
  - GRPO advantage: Bessel-corrected sample std (ddof=1) instead of population std
  - Entropy bonus: now grad-enabled (was dead code from .item() detach)
- [x] **Tokenizer**: negative values now encode by magnitude (was collapsing all to VAL_0)
- [x] **VCG decoder**: value outputs clamped to [-12, 7] log10 scale (was unbounded)
- [x] **Flow matching**: autograd.grad crash guard (allow_unused=True)
- [x] **Hybrid pipeline**: tracks raw vs repaired validity separately
- [x] **Training scripts**: fixed --valid-only flags, VCG resume best_val_loss, CCFM device placement
- [x] **Latent reward**: configurable drift_weight, proxy reward range [2,8] to match real rewards
- [x] 751 tests passing across 18 test files

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

# Multiseed robustness eval on medium checkpoints (no retraining)
PYTHONPATH=src .venv/bin/python scripts/run_topology_ablation_multiseed.py --n-samples 80 --seeds 41 42 43 44 45 --bootstrap-iters 5000 --ci 0.95 --output results/topology_ablation_medium_multiseed_5seed.json

# Inference-time alpha sweep (topology heads + family MoE) on medium checkpoints
PYTHONPATH=src .venv/bin/python scripts/run_topology_alpha_sweep.py --n-samples 48 --seeds 41 42 43 44 45 --topology-alphas 0.2 0.5 0.8 --family-alphas 0.1 0.3 0.5 --family-topology-alpha 0.5 --output results/topology_alpha_sweep_5seed.json

# Best-of-N inference scaling
PYTHONPATH=src python scripts/run_bestofn.py --checkpoint checkpoints/arcs_graph_transformer/best_model.pt --simulate

# Training with resume support
PYTHONPATH=src python -m arcs.train --data data/combined --config small --model-type graph_transformer --epochs 100 --resume checkpoints/arcs_graph_transformer/best_model.pt

# Ablation studies
PYTHONPATH=src python scripts/run_ablations.py --n-samples 160

# Baselines
PYTHONPATH=src python -m arcs.baselines --method ga --n-repeats 10

# Run all 751 tests
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

### Topology Ablation Robustness (Medium Checkpoints, 5 seeds + bootstrap CI)

| Model | Structural (mean ± std) | Sim Valid (mean ± std) | Reward (mean ± std) |
|-------|--------------------------|-------------------------|---------------------|
| Baseline | **84.5% ± 3.4%** | **63.0% ± 5.0%** | **4.101 ± 0.186** |
| + Topology Value Heads | 81.5% ± 2.2% | 59.5% ± 1.9% | 3.987 ± 0.168 |
| + Family MoE | 84.0% ± 4.5% | 62.5% ± 6.9% | 4.002 ± 0.357 |

Pairwise bootstrap deltas vs baseline (95% CI):
- `+ Topology Value Heads`: sim_valid Δ = -3.5 pp `[-7.5, +0.7]`, reward Δ = -0.114 `[-0.312, +0.089]`
- `+ Family MoE`: sim_valid Δ = -0.5 pp `[-7.2, +6.0]`, reward Δ = -0.099 `[-0.414, +0.210]`

Artifacts:
- `results/topology_ablation_medium_multiseed_5seed.json`
- `logs/evaluate_topology_ablation_medium_multiseed_5seed.log`
- `results/topology_ablation_medium_multiseed.json` (prior 3-seed run)
- `results/topology_ablation_multiseed/seed_41.json`
- `results/topology_ablation_multiseed/seed_42.json`
- `results/topology_ablation_multiseed/seed_43.json`
- `results/topology_ablation_multiseed/seed_44.json`
- `results/topology_ablation_multiseed/seed_45.json`

### Alpha Sweep (Inference-time, 5 seeds, n=48)

| Setting | Sim Valid (mean ± std) | Reward (mean ± std) | Structural (mean ± std) |
|---------|--------------------------|----------------------|--------------------------|
| Baseline | **64.2% ± 7.7%** | **4.271 ± 0.285** | **87.5% ± 4.9%** |
| FamilyMoE αf=0.50, αt=0.50 | **64.2% ± 7.4%** | 4.109 ± 0.471 | 85.8% ± 4.5% |
| FamilyMoE αf=0.30, αt=0.50 | 61.3% ± 6.4% | 3.998 ± 0.441 | 84.2% ± 6.4% |
| FamilyMoE αf=0.10, αt=0.50 | 60.8% ± 4.0% | 4.071 ± 0.226 | 84.2% ± 3.5% |
| TopologyHeads α=0.50 | 57.9% ± 7.7% | 3.781 ± 0.351 | 76.2% ± 5.8% |
| TopologyHeads α=0.20 | 54.2% ± 4.9% | 3.779 ± 0.184 | 79.2% ± 2.6% |
| TopologyHeads α=0.80 | 51.7% ± 4.5% | 3.703 ± 0.285 | 82.9% ± 6.8% |

Best by `sim_valid`: `FamilyMoE αf=0.50, αt=0.50` (tie with baseline on mean sim_valid, but lower mean reward).

Artifacts:
- `results/topology_alpha_sweep_5seed.json`
- `logs/evaluate_topology_alpha_sweep_5seed.log`

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
