# ARCS: Autoregressive Representation for Circuit Synthesis

> **"Components are words. Circuits are sentences."**

A native autoregressive model that generates complete analog circuits — topology AND component values — conditioned on performance specifications. Unlike text-LLM approaches (CircuitSynth, AutoCircuit-RL) that predict characters, and unlike topology-only models (AnalogGenie), ARCS treats circuit components as first-class tokens with embedded values and generates SPICE-simulatable designs from specs.

---

## The Problem

Analog circuit design is manual, slow, and requires deep expertise. An engineer spends days/weeks selecting a topology, picking component values, simulating, iterating. Existing AI approaches have critical gaps:

| Approach | What It Does | What It's Missing |
|----------|-------------|-------------------|
| **AnalogGenie** (ICLR 2025) | Generates topology (pin-level wiring) | No component values, no spec conditioning, IC-only |
| **CircuitSynth** (2024) | Fine-tunes GPT-Neo on netlist text | Model predicts characters, not components — error-prone |
| **AutoCircuit-RL** (2025) | Adds RL to CircuitSynth | Same text-LLM limitation |
| **cktFormer** (2025) | Transformer on graph tokens | IC-only, limited topology families |
| **FALCON** (NeurIPS 2025) | GNN + layout co-optimization | Parasitic-aware but not generative |

**Nobody has built an autoregressive model that jointly generates topology + values, conditioned on specs, across broad analog circuit families.** ARCS fills this gap.

---

## Our Approach

### Core Idea

```
INPUT:  "Vout=5V, Vin=12V, Iout=2A, ripple<50mV, efficiency>90%"

MODEL GENERATES (token by token):
  [SPEC_EMBED] → MOSFET_Q1(IRF540) → INDUCTOR_L1(22μH) →
  DIODE_D1(MBR340) → CAPACITOR_C1(470μF) → RESISTOR_R1(10kΩ) →
  RESISTOR_R2(3.3kΩ) → IC_U1(LM2596) →
  CONNECT(Q1_G, U1_OUT) → CONNECT(Q1_D, L1_P) → ...

OUTPUT: Complete SPICE netlist ready for simulation
```

### What Makes This Novel

1. **Native Component Tokens** — Each token IS a circuit element with value, not a text character
2. **Value-Aware Generation** — Component values embedded in the token representation (not a separate post-hoc sizing step)
3. **Spec Conditioning** — Performance requirements as prefix, model generates circuits meeting those specs
4. **SPICE-in-the-Loop** — Simulation feedback during training / generation for physics grounding
5. **Broad Scope** — Not limited to IC-level; covers discrete + IC analog circuits
6. **Invalid = Useful** — Failed simulations kept as negative examples (model learns what doesn't work)

---

## Data Generation Strategy

### The Key Insight: SPICE IS the Data Engine

No scraping PDFs or textbooks. No manual effort. Just simulate.

```
For each topology (Buck, Boost, Flyback, Sallen-Key, Colpitts, ...):
    For each random sample of component values within physical bounds:
        1. Plug values into SPICE netlist template
        2. Run ngspice simulation (~1-3 sec)
        3. Extract performance metrics (efficiency, ripple, gain, BW, THD, ...)
        4. Store: (topology, component values, performance metrics)
        5. Keep failures too — labeled as "invalid" (model learns constraints)
```

### Scale Estimates

| Phase | Topologies | Samples/Topology | Total | Sim Time (M3) |
|-------|-----------|------------------|-------|---------------|
| Phase 1: Power converters | 7 | 2,000 | 14,000 | ~8 hours |
| Phase 2: + Filters & amps | +15 | 2,000 | 44,000 | ~24 hours |
| Phase 3: + Oscillators & sensors | +10 | 2,000 | 64,000 | ~36 hours |
| With Eulerian augmentation (20×) | — | — | 1,280,000 | — |

AnalogGenie trained on 227K sequences. Phase 2 with augmentation already exceeds that.

---

## Circuit Types In Scope

### Tier 1: Power Electronics (Starting Here)
- Buck, Boost, Buck-Boost, Flyback, Forward, Cuk, SEPIC, LLC
- PFC circuits, inverters
- **Metrics**: efficiency, output ripple, Vout accuracy, THD, transient response

### Tier 2: Analog Signal Processing
- Op-amp circuits (inverting, non-inverting, instrumentation, differential)
- Active/passive filters (Sallen-Key, Butterworth, Chebyshev, band-pass, notch)
- Oscillators (Wien bridge, Colpitts, Hartley, ring)
- Comparators, Schmitt triggers, peak detectors
- **Metrics**: gain, bandwidth, phase margin, CMRR, noise figure, THD

### Tier 3: Sensor & Interface Circuits
- Wheatstone bridges, strain gauge amplifiers
- Current sense (high-side, low-side), level shifters
- Voltage references, thermocouple circuits
- **Metrics**: accuracy, drift, CMRR, linearity

### Tier 4: Audio Electronics
- Preamplifiers, tone controls, equalizers
- Headphone amps, Class-AB/D power amps
- Crossover networks
- **Metrics**: THD, frequency response flatness, SNR, output power

### Tier 5: RF & Communications
- LNAs, mixers, VCOs, PLLs
- Impedance matching, antenna tuning
- **Metrics**: gain, NF, IP3, phase noise

### Tier 6: Transistor-Level IC (AnalogGenie's Territory, But With Values)
- Current mirrors, bandgap references, OTAs
- Folded cascode, two-stage op-amps, LDOs
- **Metrics**: gain, GBW, power, PSRR, TC

---

## Architecture (Planned)

### Tokenizer
Unlike AnalogGenie's pin-only vocabulary (1029 tokens), our vocabulary includes:

```
Component tokens:     MOSFET_N, MOSFET_P, RESISTOR, CAPACITOR, INDUCTOR,
                      DIODE, BJT_NPN, BJT_PNP, OPAMP, TRANSFORMER, ...

Value tokens:         Discretized or continuous embedding of component value
                      (e.g., 22μH, 470μF, 10kΩ — using E-series standard values
                       or log-scale discretization)

Pin tokens:           _D, _G, _S (MOSFET), _P, _N (passive), _INP, _INN, _OUT (op-amp)

Connection tokens:    CONNECT(pin_a, pin_b), NET_1, NET_2, ...

Spec tokens:          SPEC_VIN, SPEC_VOUT, SPEC_IOUT, SPEC_RIPPLE, SPEC_GAIN, ...

Special tokens:       START (VSS), END, PAD
```

### Model Options
- **Baseline**: GPT-style decoder-only transformer (like AnalogGenie, but bigger vocab + value embeddings)
- **Enhanced**: Two-head architecture — one head predicts next component/pin, another predicts value
- **Advanced**: Graph transformer with Eulerian circuit sequentialization + value conditioning

### Training
1. **Pre-training**: Next-token prediction on all circuit sequences (unsupervised)
2. **Fine-tuning**: Spec-conditioned generation (spec prefix → circuit sequence)
3. **RLHF / RL-from-SPICE**: Reward = simulation performance metrics (like our existing RL pipeline)

---

## Comparison with Prior Art

| Feature | AnalogGenie | CircuitSynth | AutoCircuit-RL | **ARCS** |
|---------|-------------|--------------|----------------|------------------|
| Representation | Pin-level Eulerian | SPICE text | SPICE text | **Native component tokens** |
| Values/Sizing | ❌ (GA post-hoc) | In text (not understood) | In text | **Embedded in token** |
| Spec conditioning | ❌ | ❌ | ❌ | **✅** |
| Domain | IC only (11 types) | IC only | IC only | **All analog** |
| Simulation feedback | Post-hoc only | Post-hoc only | Post-hoc RL | **In-the-loop** |
| Hierarchy | Flat | Flat | Flat | **Subcircuit tokens (planned)** |
| Invalid examples | Discarded | Discarded | Discarded | **Kept as training signal** |
| Data source | Manual (textbooks) | Manual | Manual | **Automated SPICE** |
| Model size | 11.8M | 2.7B | 2.7B | **TBD (~50-100M)** |
| Venue | ICLR 2025 | arXiv 2024 | arXiv 2025 | **In progress** |

---

## Roadmap

### Phase 1: Data Generation & Proof of Concept (Weeks 1-3)
- [ ] Build parameterized SPICE templates for 7 power converter topologies
- [ ] Write data generation pipeline (random sweep + simulate + extract metrics)
- [ ] Generate ~14K circuit samples with performance labels
- [ ] Design tokenizer vocabulary (components + values + pins + specs)
- [ ] Implement Eulerian circuit representation + augmentation

### Phase 2: Model Training (Weeks 3-5)
- [ ] Implement GPT-style decoder model with circuit tokenizer
- [ ] Pre-train on unconditional next-token prediction
- [ ] Add spec-conditioning (spec prefix tokens)
- [ ] Fine-tune for spec → circuit generation
- [ ] Evaluate: validity rate, spec compliance, diversity

### Phase 3: SPICE-in-the-Loop RL (Weeks 5-7)
- [ ] Implement reward function from SPICE simulation metrics
- [ ] RL fine-tuning (PPO or GRPO) to improve generated circuit quality
- [ ] Compare: pre-trained only vs. RL-refined

### Phase 4: Expand Circuit Families (Weeks 7-10)
- [ ] Add filter, amplifier, oscillator templates
- [ ] Retrain with broader dataset
- [ ] Evaluate cross-domain generation

### Phase 5: Paper (Weeks 10-12)
- [ ] Baselines: random search, genetic algorithm, AnalogGenie + GA sizing
- [ ] Ablation: with/without spec conditioning, with/without RL, with/without invalid examples
- [ ] Write paper targeting ICLR / NeurIPS / DAC

---

## Key References

1. **AnalogGenie** — Gao et al., ICLR 2025. Autoregressive pin-level topology generation. [Paper](https://arxiv.org/abs/2503.00205) | [Code](https://github.com/xz-group/AnalogGenie)
2. **CircuitSynth** — 2024. GPT-Neo fine-tuned on SPICE netlist text. 11 citations.
3. **AutoCircuit-RL** — 2025. CircuitSynth + GRPO RL fine-tuning. 4 citations.
4. **cktFormer** — IECON 2025. Transformer on graph tokens for analog circuits.
5. **FALCON** — NeurIPS 2025. GNN + layout-constrained analog co-optimization.
6. **CktGNN** — Dong et al., 2023. Graph VAE for Op-Amp topology generation.
7. **LaMAGIC** — Chang et al., 2024. Masked language model for power converter topology.
8. **AutoCkt** — Berkeley, DATE 2020. DRL for analog circuit sizing. 100+ citations.
9. **Bui et al.** — IEEE Access 2022. DNN surrogate + DRL for power converter sizing. 24 citations.

---

## Prior Work (MLEntry Project)

This project builds on the [MLEntry](../MLEntry/) project which implemented:
- PPO RL agent + neural surrogate + SPICE-in-the-loop for 7 DC-DC converter topologies
- 109-dimensional state space, 6D continuous action space
- Topology-specific reward shaping
- Trained on M3 MacBook Air

Key infrastructure to reuse:
- SPICE simulation pipeline (PySpice/ngspice)
- Performance metric extraction
- RL training loop (PPO with topology-specific rewards)
- Topology parameter bounds and physical constraints

---

## Setup

```bash
# TODO: Setup instructions will be added as the project develops
```

---

## License

TBD
