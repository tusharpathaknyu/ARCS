# ARCS / CircuitGenie — System Architecture

> Multi-architecture neural circuit generation with SPICE-in-the-loop verification.
> 34 topologies | 6 model stages | 61,760 validated training samples

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Tokenizer](#3-tokenizer)
4. [ARCS Graph Transformer (Autoregressive)](#4-arcs-graph-transformer)
5. [Valid Circuit Generator (VCG)](#5-valid-circuit-generator-vcg)
6. [Constrained Circuit Flow Matching (CCFM)](#6-constrained-circuit-flow-matching-ccfm)
7. [Reward Model](#7-reward-model)
8. [RL / GRPO Fine-Tuning](#8-rl--grpo-fine-tuning)
9. [Hybrid Pipeline](#9-hybrid-pipeline)
10. [SPICE Simulation Engine](#10-spice-simulation-engine)
11. [Topology Library](#11-topology-library)
12. [Training Pipeline & Results](#12-training-pipeline--results)
13. [Inference & Generation](#13-inference--generation)
14. [Model Comparison](#14-model-comparison)

---

## 1. System Overview

ARCS is a multi-stage neural system that generates valid, simulatable electronic circuits from high-level specifications (voltage, current, frequency, gain, etc.). It combines autoregressive generation, graph-based VAE, flow matching, and reinforcement learning.

### High-Level Architecture Flow

```
                          TRAINING PIPELINE
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
  │  │ Template  │───>│  ngspice │───>│  JSONL   │              │
  │  │ Sampling  │    │  Sim     │    │  Dataset │              │
  │  └──────────┘    └──────────┘    └────┬─────┘              │
  │                                       │                     │
  │              ┌────────────────────────┼────────────┐        │
  │              │                        │            │        │
  │              v                        v            v        │
  │        ┌──────────┐           ┌───────────┐  ┌─────────┐   │
  │        │ ARCS GT  │           │   VCG     │  │  CCFM   │   │
  │        │ Pretrain │           │  (VAE)    │  │ (Flow)  │   │
  │        └────┬─────┘           └───────────┘  └─────────┘   │
  │             │                                               │
  │             v                                               │
  │        ┌──────────┐    ┌──────────┐    ┌──────────┐        │
  │        │  GRPO    │<───│  Reward  │<───│  Latent  │        │
  │        │  RL      │    │  Model   │    │  Reward  │        │
  │        └──────────┘    └──────────┘    └──────────┘        │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

                         INFERENCE PIPELINE
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  User Spec ──> ┌──────────┐    ┌──────────┐    ┌────────┐  │
  │  (Vin, Vout,   │  Model   │───>│  Decode  │───>│ ngspice│  │
  │   Iout, ...)   │  Generate│    │  Netlist │    │ Verify │  │
  │                └──────────┘    └──────────┘    └────────┘  │
  │                                                             │
  │  Three generation paths:                                    │
  │    Path A: ARCS GT ──> tokens ──> netlist ──> SPICE        │
  │    Path B: VCG ──> graph ──> netlist ──> SPICE             │
  │    Path C: CCFM ──> latent ──> VCG decode ──> SPICE        │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

### Parameter Summary

| Model | Parameters | Role |
|-------|-----------|------|
| ARCS Graph Transformer | 6.84M | Autoregressive circuit generation |
| VCG (VAE) | 4.00M | Structurally valid graph generation |
| CCFM (Flow Matching) | 7.66M | Spec-conditioned continuous generation |
| Reward Model | ~663K | Proxy reward for ranking |
| Latent Reward Predictor | ~1.7M | Reward prediction in VCG latent space |

---

## 2. Data Pipeline

### Generation Process

```
  For each of 34 topologies:
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  ComponentBounds ──> Random Sample ──> Netlist       │
  │       (min, max,       (log-uniform)    Template     │
  │        log_scale)                          │         │
  │                                            v         │
  │                                       ┌────────┐    │
  │                                       │ngspice │    │
  │                                       │  .tran │    │
  │                                       └───┬────┘    │
  │                                           │         │
  │                          ┌────────────────┼───┐     │
  │                          v                v   v     │
  │                     .measure          Derived       │
  │                     results           Metrics       │
  │                    (vout_avg,        (efficiency,    │
  │                     iout_avg,         vout_error,   │
  │                     ripple)           ripple_ratio) │
  │                                           │         │
  │                                           v         │
  │                                    Validity Check   │
  │                                    ──────────────   │
  │                                    Power: eff>0,    │
  │                                      verr<50%,     │
  │                                      ripple<0.5    │
  │                                    Signal: gain    │
  │                                      exists,      │
  │                                      |gain|<120dB │
  │                                    Oscillator:     │
  │                                      Vpp > 0.1V   │
  │                                           │         │
  │                                           v         │
  │                                      JSONL file     │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

### Dataset Statistics

```
  Dataset: Combined V2
  ─────────────────────────────────────────
  Total samples:      89,000
  Valid samples:      61,760  (69.4%)
  Topologies:         34
  ─────────────────────────────────────────

  Tier 1 — Power Converters (7 topologies):
  ┌────────────────┬────────┬───────┐
  │ Topology       │ Total  │ Valid │
  ├────────────────┼────────┼───────┤
  │ buck           │  5,000 │ ~4.2K │
  │ boost          │  5,000 │ ~4.0K │
  │ buck_boost     │  5,000 │ ~3.8K │
  │ cuk            │  5,000 │ ~3.5K │
  │ sepic          │  5,000 │ ~1.7K │
  │ flyback        │  5,000 │ ~0.9K │
  │ forward        │  5,000 │ ~2.1K │
  └────────────────┴────────┴───────┘

  Tier 2 — Signal Processing (27 topologies):
  ┌────────────────────────┬────────┬───────┐
  │ Topology               │ Total  │ Valid │
  ├────────────────────────┼────────┼───────┤
  │ inverting_amp          │  2,000 │ ~1.9K │
  │ noninverting_amp       │  2,000 │ ~1.9K │
  │ sallen_key_lowpass     │  2,000 │ ~1.8K │
  │ wien_bridge            │  2,000 │ ~1.3K │
  │ phase_shift            │  2,000 │ ~1.2K │
  │ zeta_converter         │  2,000 │ ~1.2K │
  │ ... (24 more)          │  2,000 │  var  │
  └────────────────────────┴────────┴───────┘
```

---

## 3. Tokenizer

### Vocabulary Structure (706 tokens)

```
  Token ID Range    Category          Count   Examples
  ──────────────    ────────────      ─────   ──────────────────────
  [0..4]            Special              5    PAD, START, END, SEP, INVALID
  [5..24]           Component Types     20    RESISTOR, CAPACITOR, INDUCTOR,
                                              MOSFET_N, OPAMP, TRANSFORMER...
  [25..64]          Topology            40    TOPO_BUCK, TOPO_BOOST,
                                              TOPO_SALLEN_KEY_LP...
  [65..84]          Specifications      20    SPEC_VIN, SPEC_VOUT,
                                              SPEC_GAIN, SPEC_BANDWIDTH...
  [85..105]         Pin Names           21    PIN_DRAIN, PIN_GATE,
                                              PIN_COLLECTOR, PIN_BASE...
  [106..205]        Net/Connection     100    NET_0 through NET_99
  [206..705]        Value Bins         500    Log-scale 1e-12 to 1e6
  ──────────────────────────────────────────────────────────────────
  Total:                               706
```

### Value Discretization

```
  Log-scale binning: 500 bins spanning 18 orders of magnitude

  1e-12  ──────────────────────────────────────────>  1e6
  (pF)    (nF)    (uF)    (mF)   (Ohm)  (kOhm)  (MOhm)

  bin_edges = linspace(log10(1e-12), log10(1e6), 501)
  bin_center[i] = 10^((edge[i] + edge[i+1]) / 2)

  Resolution: ~3.6% per bin (constant relative precision)
```

### Sequence Format

```
  ┌───────┬──────────┬─────┬──────────────────────┬─────┬──────────────────────────┬─────┐
  │ START │ TOPO_BUCK│ SEP │SPEC_VIN 12.0 SPEC_...│ SEP │INDUCTOR 100u CAPACITOR...│ END │
  └───────┴──────────┴─────┴──────────────────────┴─────┴──────────────────────────┴─────┘
     │         │        │         │                   │         │                      │
     │         │        │    Spec section              │    Component section          │
     │    Topology      │   (key-value pairs)          │   (type-value pairs)          │
     │                  │                              │                               │
  Always               Separators                    End of sequence
  first                between sections
```

### Token Types (7 categories)

```
  Type ID   Name         Used For
  ───────   ──────────   ─────────────────────────────
  0         SPECIAL      PAD, START, END, SEP, INVALID
  1         COMPONENT    RESISTOR, CAPACITOR, etc.
  2         TOPOLOGY     TOPO_BUCK, TOPO_BOOST, etc.
  3         SPEC         SPEC_VIN, SPEC_VOUT, etc.
  4         PIN          PIN_DRAIN, PIN_GATE, etc.
  5         CONNECTION   NET_0 through NET_99
  6         VALUE        VAL_0 through VAL_499
```

---

## 4. ARCS Graph Transformer

### Architecture

```
  Input: Token sequence (B, T)    T <= 128

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  Token Embedding     (706 -> 256)                   │
  │    + Position Emb    (128 -> 256)                   │
  │    + TokenType Emb   (  7 -> 256)                   │
  │         │                                           │
  │         v                                           │
  │    Dropout(0.1)                                     │
  │         │                                           │
  │         v                                           │
  │  ┌─────────────────────────────────────┐            │
  │  │  GraphTransformerBlock x 6          │            │
  │  │                                     │            │
  │  │  ┌───────────────────────────────┐  │            │
  │  │  │ RMSNorm                       │  │            │
  │  │  │    │                          │  │            │
  │  │  │    v                          │  │            │
  │  │  │ Adjacency-Biased Attention    │  │            │
  │  │  │  (4 heads, d_head=64)         │  │            │
  │  │  │  + RWPE positional encoding   │  │            │
  │  │  │  + Edge-type bias (6 types)   │  │            │
  │  │  │    │                          │  │            │
  │  │  │    + Residual                 │  │            │
  │  │  │    │                          │  │            │
  │  │  │ RMSNorm                       │  │            │
  │  │  │    │                          │  │            │
  │  │  │    v                          │  │            │
  │  │  │ SwiGLU FFN                    │  │            │
  │  │  │  (256 -> 1024 -> 256)         │  │            │
  │  │  │    │                          │  │            │
  │  │  │    + Residual                 │  │            │
  │  │  └───────────────────────────────┘  │            │
  │  └─────────────────────────────────────┘            │
  │         │                                           │
  │         v                                           │
  │    RMSNorm (final)                                  │
  │         │                                           │
  │    ┌────┴────────────────────┐                      │
  │    │                         │                      │
  │    v                         v                      │
  │  Structure Head           Value Head                │
  │  (256 -> 706)             (256 -> 256 -> 256        │
  │  [weight-tied to           -> 706)                  │
  │   token embed]            [independent MLP          │
  │                            + linear]                │
  │    │                         │                      │
  │    v                         v                      │
  │  Logits for              Logits for                 │
  │  topo/comp tokens        value tokens               │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  Total parameters: 6,839,536
  Config: d_model=256, n_layers=6, n_heads=4, d_ff=1024
```

### Graph-Aware Attention

```
  Standard causal attention + topology-aware bias:

  Attention(Q, K, V) = softmax(QK^T/sqrt(d) + A_bias + E_bias) V

  Where:
    A_bias: Binary adjacency matrix (components electrically connected)
    E_bias: Edge-type embeddings (6 relation types)
    RWPE:   Random Walk Positional Encoding (6 walk lengths)

  Per-topology adjacency example (buck converter):
    Component 0 (Inductor) <-> Component 3 (MOSFET)    [switch node]
    Component 0 (Inductor) <-> Component 1 (Capacitor)  [output node]
    Component 1 (Capacitor) <-> Component 2 (ESR)       [series]
```

### Two-Head Output

```
  At each generation step, the model uses last_token_type to route:

  If previous token was COMPONENT type:
    ──> Value Head (specialized for predicting component values)

  If previous token was VALUE, SPEC, TOPOLOGY, or SPECIAL:
    ──> Structure Head (predicts next structural token)

  Combined output:
    logits = alpha * structure_logits + (1-alpha) * value_logits
    (alpha determined by token type routing)
```

---

## 5. Valid Circuit Generator (VCG)

### VAE Architecture

```
  ┌────────────────────────────────────────────────────────────┐
  │                        VCG (VAE)                           │
  │                                                            │
  │  Input: Circuit Graph G = (nodes, edges, values)           │
  │         + topology index (0..33)                           │
  │         + spec embedding (from ARCS tokenizer)             │
  │                                                            │
  │  ┌──────────────────────────────────────────┐              │
  │  │            ENCODER                       │              │
  │  │                                          │              │
  │  │  Node features: one-hot(type) + log(val) │              │
  │  │       │                                  │              │
  │  │       v                                  │              │
  │  │  4x GNN Layers (d=256, 4 heads)          │              │
  │  │  [Message passing on adjacency]          │              │
  │  │       │                                  │              │
  │  │       v                                  │              │
  │  │  Graph-level readout (mean pool)         │              │
  │  │       │                                  │              │
  │  │       v                                  │              │
  │  │  ┌────┴────┐                             │              │
  │  │  │ mu      │  sigma                      │              │
  │  │  │ (256->64)  (256->64)                  │              │
  │  │  └────┬────┘                             │              │
  │  │       │                                  │              │
  │  │       v                                  │              │
  │  │  z ~ N(mu, sigma^2)   [64-dim latent]    │              │
  │  │                                          │              │
  │  └──────────────────────────────────────────┘              │
  │                                                            │
  │  ┌──────────────────────────────────────────┐              │
  │  │            DECODER                       │              │
  │  │                                          │              │
  │  │  z (64) + topo_embed + spec_embed        │              │
  │  │       │                                  │              │
  │  │       v                                  │              │
  │  │  3x MLP Layers (d_hidden=512)            │              │
  │  │       │                                  │              │
  │  │       v                                  │              │
  │  │  ┌────┴───────────────────────┐          │              │
  │  │  │ Node type   Node values   │          │              │
  │  │  │ logits      predictions   │          │              │
  │  │  │ (16 types)  (continuous)  │          │              │
  │  │  └────────────────────────────┘          │              │
  │  │                                          │              │
  │  └──────────────────────────────────────────┘              │
  │                                                            │
  │  Config: latent_dim=64, max_nodes=12, n_types=16           │
  │          beta_kl=0.1, encoder_layers=4, decoder_layers=3   │
  │  Parameters: 3,998,769                                     │
  │                                                            │
  └────────────────────────────────────────────────────────────┘
```

### 5 Differentiable Constraints

```
  After decoding z -> graph, apply constraint projection:

  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  1. No Floating Nodes                                    │
  │     Every node must have >= 1 edge                       │
  │     Penalty: sum of node degrees that are 0              │
  │                                                          │
  │  2. Device Completeness                                  │
  │     Multi-terminal devices (transistors, opamps)         │
  │     must have all required connections                   │
  │     Penalty: missing terminal count                      │
  │                                                          │
  │  3. No Short Circuits                                    │
  │     Power/ground nodes cannot be directly connected      │
  │     Penalty: edge weight between VCC-GND pairs           │
  │                                                          │
  │  4. Graph Connectivity                                   │
  │     Circuit must form expected # of connected            │
  │     components (topology-aware: most=1, some=2-3)        │
  │     Uses Laplacian eigenvalue check                      │
  │     Penalty: eigenvalue[K] where K = expected components │
  │                                                          │
  │  5. Value Bounds                                         │
  │     Component values must be within physical bounds      │
  │     Penalty: max(0, val - max) + max(0, min - val)       │
  │                                                          │
  │  Projection: 20 Adam steps on constraint loss            │
  │  Result: 100% structural validity on all 34 topologies   │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

---

## 6. Constrained Circuit Flow Matching (CCFM)

### Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │                        CCFM                              │
  │                                                          │
  │  Training:                                               │
  │                                                          │
  │  z_0 ~ N(0, I)          z_1 = VCG.encode(circuit)       │
  │     │                       │                            │
  │     └───────┬───────────────┘                            │
  │             v                                            │
  │     z_t = (1-t)*z_0 + t*z_1    t ~ U(0,1)               │
  │             │                                            │
  │             v                                            │
  │  ┌──────────────────────────┐                            │
  │  │    Flow Network          │                            │
  │  │                          │                            │
  │  │  z_t (64)                │                            │
  │  │  + t_embed (64)          │  ┌──────────────────┐      │
  │  │  + spec_embed (256)      │<─│ Spec Conditioner │      │
  │  │  + topo_idx              │  │ (topology + specs │      │
  │  │       │                  │  │  -> 256-dim embed)│      │
  │  │       v                  │  └──────────────────┘      │
  │  │  4x Transformer Layers  │                            │
  │  │  (d=256, 4 heads)       │                            │
  │  │       │                  │                            │
  │  │       v                  │                            │
  │  │  v_pred (64)             │  Predicted velocity field  │
  │  └──────────────────────────┘                            │
  │                                                          │
  │  Loss = MSE(v_pred, u_t) + 0.1 * consistency_loss        │
  │  Where u_t = z_1 - z_0 (optimal transport target)        │
  │                                                          │
  │  ────────────────────────────────────────────────         │
  │                                                          │
  │  Inference (ODE integration):                            │
  │                                                          │
  │  z_0 ~ N(0, I)                                          │
  │     │                                                    │
  │     │  for t in [0, 1] with 50 Euler steps:              │
  │     │                                                    │
  │     │    v = FlowNet(z_t, t, spec_embed, topo_idx)       │
  │     │                                                    │
  │     │    if t > 0.3:  (guidance kicks in)                │
  │     │      g = constraint_gradient(z_t, topo_idx)        │
  │     │      v = v + lambda * g                            │
  │     │                                                    │
  │     │    z_{t+dt} = z_t + dt * v                         │
  │     │                                                    │
  │     v                                                    │
  │  z_1 = final latent ──> VCG.decode(z_1) ──> circuit      │
  │                                                          │
  │  Classifier-Free Guidance (CFG):                         │
  │    v = v_uncond + cfg_scale * (v_cond - v_uncond)        │
  │    cfg_scale = 1.5, p_uncond = 0.1 (training dropout)    │
  │                                                          │
  │  Config: latent_dim=64, flow_d_model=256,                │
  │          flow_n_layers=4, n_sample_steps=50              │
  │  Parameters: 7,663,345                                   │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

---

## 7. Reward Model

### Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │                 Circuit Reward Model                      │
  │                                                          │
  │  Input: Token sequence (B, T) + attention_mask            │
  │                                                          │
  │  Token Embedding    (706 -> 128)                          │
  │    + Position Emb   (128 -> 128)                          │
  │       │                                                   │
  │       v                                                   │
  │  2x Bidirectional Transformer Encoder Blocks              │
  │    (d=128, 4 heads, d_ff=512, NO causal mask)             │
  │       │                                                   │
  │       v                                                   │
  │  LayerNorm                                                │
  │       │                                                   │
  │       v                                                   │
  │  Mean Pooling (over non-PAD tokens)                       │
  │       │                                                   │
  │       v                                                   │
  │  MLP Head:                                                │
  │    Linear(128 -> 256) -> GELU -> Dropout -> Linear(256->1)│
  │       │                                                   │
  │       v                                                   │
  │  Clamp to [0.0, 8.0]                                      │
  │       │                                                   │
  │       v                                                   │
  │  Scalar reward prediction                                 │
  │                                                          │
  │  Training: HuberLoss(delta=1.0), AdamW, cosine LR        │
  │  Parameters: ~663K                                        │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

### Reward Function (Ground Truth)

```
  Domain-Aware Reward Dispatch:

  ┌─────────────────────┐
  │ Valid structure?     │──No──> reward = 0.0
  │ (+1.0 struct bonus)  │
  └─────────┬───────────┘
            │ Yes (+1.0)
            v
  ┌─────────────────────┐
  │ SPICE converged?    │──No──> reward = struct_bonus
  │ (+1.0 convergence)   │
  └─────────┬───────────┘
            │ Yes (+1.0)
            v
  ┌─────────────────────────────────────────────────┐
  │ Topology type?                                   │
  ├───────────────────┬──────────────┬───────────────┤
  │ Power converter   │ Signal circ  │ Current mirror│
  │ (buck, boost...)  │ (amp, filter)│ (mirror, ...)│
  ├───────────────────┼──────────────┼───────────────┤
  │ Vout accuracy:    │ Gain:   3.0  │ Iref/Iout    │
  │   3.0 * max(0,    │ BW:     2.0  │  matching:   │
  │   1 - err/10)     │ Other:  1.0  │   up to 6.0  │
  │ Efficiency:       │              │              │
  │   2.0 * eff       │ Total:  6.0  │              │
  │ Low ripple:       │              │              │
  │   1.0 * max(0,    │              │              │
  │   1 - rip*10)     │              │              │
  │ Total:       6.0  │              │              │
  └───────────────────┴──────────────┴───────────────┘

  Max total reward: 1.0 (struct) + 1.0 (sim) + 6.0 (quality) = 8.0
```

---

## 8. RL / GRPO Fine-Tuning

### GRPO Algorithm

```
  ┌──────────────────────────────────────────────────────────┐
  │          GRPO: Group Relative Policy Optimization         │
  │                                                          │
  │  For each training step:                                  │
  │                                                          │
  │  1. Sample 3 topologies (n_topos_per_step)                │
  │                                                          │
  │  2. For each topology, generate 4 circuits (group_size)   │
  │     with log-probabilities                                │
  │                                                          │
  │  3. Simulate all 12 circuits with ngspice                 │
  │     ┌──────────────────────────────┐                      │
  │     │ Circuit ──> Netlist ──> Sim  │                      │
  │     │   r_1, r_2, r_3, r_4        │  per topology        │
  │     └──────────────────────────────┘                      │
  │                                                          │
  │  4. Compute rewards and z-score WITHIN each group:        │
  │     advantages = (rewards - mean(rewards)) / std(rewards) │
  │     Clip advantages to [-5, 5]                            │
  │                                                          │
  │  5. Policy gradient with KL penalty:                      │
  │     loss = -advantage * log_prob                          │
  │         + kl_coeff * KL(policy || reference)              │
  │         - entropy_coeff * entropy                         │
  │                                                          │
  │  Key insight: Z-scoring within topology groups prevents   │
  │  cross-topology interference (buck rewards vs filter      │
  │  rewards are incomparable)                                │
  │                                                          │
  │  Config:                                                  │
  │    steps=3000, lr=1e-5, kl_coeff=0.1                      │
  │    temperature=0.8, top_k=50                              │
  │    group_size=4, n_topos_per_step=3                       │
  │    grpo_clip_adv=5.0                                      │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
```

### SPICE-in-the-Loop Flow

```
  Model ──generate──> Tokens ──decode──> DecodedCircuit
                                              │
                                              v
                                    ┌──────────────────┐
                                    │ components_to_    │
                                    │ params()          │
                                    │ (inverse tokenize)│
                                    └────────┬─────────┘
                                             │
                                             v
                                    ┌──────────────────┐
                                    │ template.generate │
                                    │ _netlist(params)  │
                                    └────────┬─────────┘
                                             │
                                             v
                                    ┌──────────────────┐
                                    │  NGSpiceRunner   │
                                    │  .run(netlist)   │
                                    │  timeout=30s     │
                                    └────────┬─────────┘
                                             │
                                             v
                                    ┌──────────────────┐
                                    │ compute_derived  │
                                    │ _metrics()       │
                                    │ + is_valid_result │
                                    └────────┬─────────┘
                                             │
                                             v
                                    ┌──────────────────┐
                                    │ compute_reward() │
                                    │ domain-aware     │
                                    │ (power/signal/   │
                                    │  mirror/reg)     │
                                    └──────────────────┘
```

---

## 9. Hybrid Pipeline

### VCG + CCFM + SPICE End-to-End

```
  User Specification
  (topology, Vin, Vout, Iout, ...)
       │
       v
  ┌──────────────────────────────────────────────────────────┐
  │                    Hybrid Generator                       │
  │                                                          │
  │  ┌──────────────┐         ┌──────────────┐              │
  │  │    VCG        │         │    CCFM       │              │
  │  │  Generate     │         │  Generate     │              │
  │  │  4 candidates │         │  4 candidates │              │
  │  └──────┬───────┘         └──────┬───────┘              │
  │         │                        │                       │
  │         └────────┬───────────────┘                       │
  │                  v                                       │
  │         8 candidate circuits                             │
  │                  │                                       │
  │                  v                                       │
  │    ┌─────────────────────────┐                           │
  │    │  Pre-rank with proxy    │                           │
  │    │  (Reward Model or       │                           │
  │    │   structural heuristic) │                           │
  │    └────────────┬────────────┘                           │
  │                 │                                        │
  │                 v                                        │
  │    ┌─────────────────────────┐                           │
  │    │  SPICE simulate top K   │                           │
  │    │  (K=4 by default)       │                           │
  │    └────────────┬────────────┘                           │
  │                 │                                        │
  │                 v                                        │
  │    Select best by SPICE reward                           │
  │                 │                                        │
  │                 v                                        │
  │    Final circuit with verified metrics                   │
  │                                                          │
  └──────────────────────────────────────────────────────────┘

  Results (34 topologies):
    VCG path:    95.6% sim_valid, reward=5.717, gen=21.5ms
    CCFM path:   91.2% sim_valid, reward=5.579, gen=191.6ms
    Hybrid best: 94.1% sim_valid, reward=6.593
```

---

## 10. SPICE Simulation Engine

### NGSpiceRunner

```
  ┌──────────────────────────────────────────────────────┐
  │                 NGSpiceRunner                         │
  │                                                      │
  │  Input: SPICE netlist string + metric_names list      │
  │                                                      │
  │  1. Write netlist to temp file                        │
  │     /tmp/arcs_XXXXX.cir                               │
  │                                                      │
  │  2. Execute:                                          │
  │     ngspice -b -o output.out netlist.cir              │
  │     Timeout: 30 seconds                               │
  │                                                      │
  │  3. Parse output for .measure results:                │
  │     Regex: (\w+)\s+=\s+([+-]?\d+\.?\d*(?:e[+-]?\d+)?)│
  │                                                      │
  │  4. Check for errors:                                 │
  │     - "Error" / "Fatal" in output                     │
  │     - "Timestep too small"                            │
  │     - Non-zero exit code                              │
  │     - Missing expected metrics                        │
  │                                                      │
  │  5. Clean up temp files                               │
  │                                                      │
  │  Output: SimulationResult                             │
  │    success: bool                                      │
  │    metrics: {vout_avg, iout_avg, iin_avg,              │
  │              vout_ripple, ...}                         │
  │    sim_time_seconds: float                            │
  │    error_message: str                                 │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

### Netlist Template System

```
  TopologyTemplate:
    name: str                    # e.g., "buck"
    category: str                # "power" | "amplifier" | "filter" | "oscillator"
    bounds: [ComponentBounds]    # parameter ranges
    operating_conditions: dict   # default vin, vout, iout, fsw, etc.
    netlist_fn: Callable         # params, conditions -> netlist string
    metric_names: [str]          # what to extract from .measure

  Example netlist structure (buck converter):
  ┌─────────────────────────────────────────────────────┐
  │  * ARCS Buck Converter                               │
  │  * Vin=12V, Vout=5V, Iout=1A, fsw=100kHz            │
  │                                                      │
  │  Vin input 0 DC 12                                   │
  │  S1 input sw_node pwm_ctrl 0 SMOD                    │
  │  L1 sw_node vout 100u IC=1                           │
  │  C1 vout 0 47u IC=5                                  │
  │  Resr vout vesr 0.01                                 │
  │  Rload vout load_mid 5                               │
  │  Vsense load_mid 0 DC 0                              │
  │                                                      │
  │  .model SMOD SW(VT=0.5 RON={r_dson})                 │
  │  Vpwm pwm_ctrl 0 PULSE(...)                          │
  │                                                      │
  │  .tran {tstep} {sim_time} {meas_start} UIC           │
  │  .measure TRAN vout_avg AVG V(vout) FROM=... TO=...  │
  │  .measure TRAN vout_ripple PP V(vout) FROM=... TO=.. │
  │  .measure TRAN iout_avg AVG par('-I(Vsense)') ...    │
  │  .measure TRAN iin_avg AVG par('-I(Vin)') ...        │
  │  .end                                                │
  └─────────────────────────────────────────────────────┘
```

---

## 11. Topology Library

### All 34 Topologies by Category

```
  POWER CONVERTERS (7)                    AMPLIFIERS (10)
  ──────────────────                      ─────────────
  buck          Step-down DC-DC           inverting_amp
  boost         Step-up DC-DC             noninverting_amp
  buck_boost    Inverting DC-DC           instrumentation_amp
  cuk           Non-inverting DC-DC       differential_amp
  sepic         Non-inverting DC-DC       inverting_summing_amp
  flyback       Isolated DC-DC            transimpedance_amp
  forward       Isolated DC-DC            common_emitter
                                          common_collector
  FILTERS (5)                             common_base
  ────────────                            cascode
  sallen_key_lowpass
  sallen_key_highpass                     OSCILLATORS (4)
  sallen_key_bandpass                     ──────────────
  twin_t_notch                            wien_bridge
  state_variable_filter                   colpitts
                                          hartley
  REGULATORS (2)                          phase_shift
  ──────────────
  shunt_regulator                         OTHER POWER (6)
  series_regulator                        ───────────────
                                          half_bridge
  CURRENT SOURCES (1)                     push_pull
  ────────────────                        charge_pump
  current_mirror                          voltage_doubler
                                          zeta_converter
```

### Component Bounds Example

```
  buck:
    inductance    1e-6  to 1e-3   H     (1uH - 1mH)      log-scale
    capacitance   1e-6  to 1e-3   F     (1uF - 1mF)      log-scale
    esr           0.001 to 1.0    Ohm   (1mOhm - 1Ohm)   log-scale
    r_dson        0.001 to 1.0    Ohm   (1mOhm - 1Ohm)   log-scale

  inverting_amp:
    r_input       1e3   to 1e6    Ohm   (1kOhm - 1MOhm)  log-scale
    r_feedback    1e3   to 1e6    Ohm   (1kOhm - 1MOhm)  log-scale

  wien_bridge:
    r_freq        1e3   to 100e3  Ohm                     log-scale
    c_freq        1e-9  to 1e-6   F                       log-scale
    r_feedback    4e3   to 100e3  Ohm   (gain >= 3)       log-scale
    r_ground      1e3   to 33e3   Ohm                     log-scale
```

---

## 12. Training Pipeline & Results

### Training Phases

```
  Phase 1: ARCS Graph Transformer Pre-training
  ─────────────────────────────────────────────
  Data:       61,760 valid + invalid samples (89K total)
  Epochs:     100
  Batch:      64
  LR:         3e-4 -> cosine decay
  Value wt:   5x on component value tokens
  Duration:   ~16 hours

  Results:    val_loss=2.53, accuracy=70.9%
              struct_accuracy=91.1%, value_accuracy=41.5%

  Phase 2: VCG (VAE) Training
  ────────────────────────────
  Data:       41,064 valid circuit graphs
  Epochs:     100
  Beta KL:    0.1
  Duration:   ~3 hours

  Results:    100% structural validity (34/34 topologies)

  Phase 3: CCFM (Flow Matching) Training
  ───────────────────────────────────────
  Data:       41,064 valid circuits (via VCG encoder)
  Epochs:     100
  Duration:   ~7 hours

  Results:    100% structural validity (34/34 topologies)

  Phase 4: Latent Reward Predictor
  ─────────────────────────────────
  Data:       VCG latent vectors + SPICE rewards
  Epochs:     50
  Duration:   ~1 hour

  Phase 5: Reward Model Training
  ───────────────────────────────
  Data:       53K token sequences + SPICE rewards
  Loss:       Huber (delta=1.0)
  Duration:   ~30 min

  Phase 6: GRPO RL Fine-Tuning
  ─────────────────────────────
  Steps:      3,000
  Batch:      3 topos x 4 circuits = 12 per step
  LR:         1e-5
  Duration:   ~15 hours

  Results:    struct_valid 58% -> 90%
              sim_success  46% -> 73%
              reward       1.88 -> 3.80
```

### Training Curves

```
  ARCS GT Val Loss (100 epochs):

  loss
  3.0 |*
      |  *
  2.8 |    *
      |      * *
  2.6 |          * * * *
      |                    * * * * * * *
  2.5 |─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ best=2.529
      |                                      * * * *
  2.4 |
      └────────────────────────────────────────────────
       0    20    40    60    80    100  epochs


  GRPO Reward (3000 steps):

  reward
  4.0 |                                          * *
      |                                  * * * *
  3.5 |                          * * * *
      |                  * * * *
  3.0 |          * * * *
      |      * *
  2.5 |  * *
      | *
  2.0 |*
      └────────────────────────────────────────────────
       0     500   1000  1500  2000  2500  3000  steps
```

---

## 13. Inference & Generation

### Three Generation Paths

```
  PATH A: ARCS Autoregressive (Token-by-Token)
  ─────────────────────────────────────────────
  Input:  Spec prefix (START, TOPO, SEP, specs, SEP)
  Output: Full token sequence -> decoded circuit -> netlist
  Speed:  ~20-30ms per circuit
  Quality: 88% struct valid, 47% sim valid, reward=3.77

  PATH B: VCG Direct (Graph VAE)
  ──────────────────────────────
  Input:  Topology index + spec embedding
  Output: z ~ prior -> decode -> project constraints -> circuit
  Speed:  ~20-50ms per circuit
  Quality: 100% struct valid, 95.6% sim valid, reward=5.72

  PATH C: CCFM Flow (Latent ODE)
  ───────────────────────────────
  Input:  Topology index + spec embedding
  Output: z_0 ~ N(0,I) -> 50 ODE steps -> z_1 -> VCG decode -> circuit
  Speed:  ~45-190ms per circuit
  Quality: 100% struct valid, 91.2% sim valid, reward=5.58
```

### Constrained Generation

```
  ConstraintLevel:
    NONE     = 0   No constraints (raw sampling)
    GRAMMAR  = 1   + Enforce COMP/VAL alternation
    TOPOLOGY = 2   + Correct component types per topology
    FULL     = 3   + Value ranges within physical bounds

  Applied as logit masks at each generation step:
    forbidden_tokens -> logits[forbidden] = -inf
    Result: 100% structural validity at FULL level
```

### Best-of-N Selection

```
  Generate N candidates (N=1 to 50)
  Score each by: confidence | entropy | reward_model
  Select top-1

  Performance scaling:
  ┌────┬──────────┬──────────┐
  │  N │ Time(ms) │ Validity │
  ├────┼──────────┼──────────┤
  │  1 │     31.7 │   100%   │
  │  3 │     94.6 │   100%   │
  │  5 │    166.6 │   100%   │
  │ 10 │    318.5 │   100%   │
  │ 20 │    622.2 │   100%   │
  │ 50 │   1554.8 │   100%   │
  └────┴──────────┴──────────┘
```

---

## 14. Model Comparison

### Final Evaluation Results

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    AUTOREGRESSIVE MODELS                            │
  │  (100 circuits, SPICE simulation, spec-conditioned)                │
  │                                                                     │
  │  Model              Params  Struct  SimOK  Valid  Reward   Eff     │
  │  ─────────────────  ──────  ──────  ─────  ─────  ──────  ─────   │
  │  ARCS-SL v3 (GT)    6.8M    88.0%  75.0%  47.0%   3.769  67.3%   │
  │  ARCS-GRPO v2       6.8M    90.0%  73.0%  43.0%   3.801  64.2%   │
  │                                                                     │
  ├─────────────────────────────────────────────────────────────────────┤
  │                      GRAPH MODELS                                   │
  │  (10 per topology, structural validity)                            │
  │                                                                     │
  │  Model              Params  Valid   100% Topos   Total             │
  │  ─────────────────  ──────  ──────  ──────────   ─────             │
  │  VCG v4 (VAE)       4.0M   100.0%    34/34      340/340           │
  │  CCFM v4 (Flow)     7.7M   100.0%    34/34      340/340           │
  │                                                                     │
  ├─────────────────────────────────────────────────────────────────────┤
  │                    HYBRID (VCG + CCFM + SPICE)                     │
  │                                                                     │
  │  Path       Samples  Sim Valid  Reward   Gen Time                  │
  │  ─────────  ───────  ─────────  ──────   ────────                  │
  │  VCG          136      95.6%    5.717    21.5ms                    │
  │  CCFM         136      91.2%    5.579    191.6ms                   │
  │  Hybrid        34      94.1%    6.593    73.9ms                    │
  │                                                                     │
  ├─────────────────────────────────────────────────────────────────────┤
  │                       BASELINES                                     │
  │                                                                     │
  │  Method          Reward   Time/Design                              │
  │  ──────────────  ──────   ───────────                              │
  │  Random Search    7.315   ~5 min                                   │
  │  Genetic Algo     7.411   ~3 min                                   │
  │  ARCS (ours)      6.593   ~74 ms    (1000x faster)                 │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

  Key insight: ARCS trades ~10% reward quality for ~1000x speed.
  Suitable for real-time circuit design assistance where
  "good enough" circuits are needed instantly.
```

---

*Generated 2026-03-22. See NEXT_STEPS.md for audit findings and pending improvements.*
