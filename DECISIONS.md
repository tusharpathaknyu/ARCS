# ARCS: Decision Log & Progress Tracker

> Last updated: 2026-02-26

---

## Project Timeline

| Commit | Date | Milestone |
|--------|------|-----------|
| `0b49bc6` | — | Initial commit: README with full research plan |
| `187926e` | — | Phase 1 infrastructure: SPICE pipeline, tokenizer, Euler repr |
| `3f141ef` | — | Fix all 7 topology templates (circuit physics, SW model) |
| `f5468a1` | — | Phase 2: GPT decoder model, dataset, training loop, evaluation |
| `15103f3` | — | Add DECISIONS.md, update README checkboxes |
| `4b786ba` | 2026-02-20 | Phase 3: SPICE-in-the-loop RL module (code + smoke tests) |
| `ba66f4b` | 2026-02-20 | Phase 3: RL training complete (5000 steps, best reward 7.02/8.0) |
| `def1cd8` | 2026-02-21 | Phase 4: Tier 2 templates (9 topologies) + RL v2 validity fixes |
| `72f162e` | 2026-02-21 | Fix AC simulation: add .save directive for subcircuit nodes |
| `aad340d` | 2026-02-21 | Phase 4 results: RL v2 training complete + evaluation |
| `8cf8dcf` | 2026-02-21 | Simulation-based evaluation: shared simulate.py + upgraded evaluate.py |
| `3bfa882` | 2026-02-21 | Fix efficiency metric >100% bug in power converter templates |
| `3dd7ada` | 2026-02-22 | Phase 5: Add random search + GA baselines |
| `4cc5170` | 2026-02-22 | Phase 5: ablations, demo CLI, paper scaffold, README rewrite |
| `0d5ebb2` | 2026-02-23 | Phase 5 polish: figures, warm-start, pytest suite (58 tests), paper |
| `50af8ea` | 2026-02-26 | Phase 5b: Two-head & graph transformer with topology-aware attention |

---

## Phase 1: Data Generation & Infrastructure

### Status: ✅ COMPLETE

#### Dataset (35,000 samples, ~6 hours generation)
| Topology | Samples | Valid | Yield |
|----------|---------|-------|-------|
| Buck | 5,000 | 3,811 | 76.2% |
| Boost | 5,000 | 3,156 | 63.1% |
| Buck-Boost | 5,000 | 2,033 | 40.7% |
| Ćuk | 5,000 | 2,791 | 55.8% |
| SEPIC | 5,000 | 1,690 | 33.8% |
| Flyback | 5,000 | 901 | 18.0% |
| Forward | 5,000 | 2,057 | 41.1% |
| **Total** | **35,000** | **16,439** | **47.0%** |

#### Infrastructure
- **7 SPICE templates** for DC-DC power converters: buck, boost, buck-boost, Ćuk, SEPIC, flyback, forward
- **NGSpice runner** (`spice.py`): batch-mode simulation with output file capture, temp file cleanup
- **Data generation pipeline** (`datagen.py`): random parameter sweep with E-series snapping, SPICE simulation, metric extraction, JSONL persistence
- **Tokenizer** (`tokenizer.py`): 676-token vocabulary (components, values, pins, nets, specs, topologies)
- **Eulerian representation** (`euler.py`): Circuit graph → Eulerian walk for data augmentation

---

## Phase 2: Model Training

### Status: ✅ TRAINING COMPLETE

#### Training Configuration
- **Data**: 35K samples × 5 augmentation = 175K training sequences
- **Model**: 6.5M params (d=256, layers=6, heads=4, SwiGLU/RMSNorm)
- **Optimizer**: AdamW lr=3e-4, warmup 5 epochs, cosine decay to 3e-5
- **Batch**: 64, val_split=0.1, value_weight=5×, gradient clip=1.0
- **Duration**: 100 epochs, ~27 hours on M3 MacBook Air (MPS)

#### Training Curve Summary
| Epoch | Train Loss | Val Loss | Val PPL | Accuracy | Value Acc | Struct Acc |
|-------|-----------|----------|---------|----------|-----------|------------|
| 1 | 2.789 | 1.959 | 7.1 | 63.7% | 44.6% | 77.3% |
| 25 | 1.399 | 1.462 | 4.3 | 69.0% | 57.3% | 77.3% |
| 50 | 1.108 | 1.300 | 3.7 | 71.5% | 63.5% | 77.3% |
| 60 | 1.028 | 1.282 | 3.6 | 72.0% | 64.7% | 77.3% |
| **68*** | — | **1.279** | **3.6** | — | — | — |
| 75 | 0.917 | 1.288 | 3.6 | 72.5% | 65.8% | 77.3% |
| 100 | 0.869 | 1.305 | 3.7 | 72.7% | 66.2% | 77.3% |

\* Best checkpoint (epoch 68, val_loss=1.2791)

#### Evaluation Results (210 samples per mode, best checkpoint)

**Spec-Conditioned Generation (30 samples × 7 topologies)**:
- Structural validity: **100%** (210/210)
- Avg components per circuit: 5.3
- Unique component combos: 29
- Diversity score: 0.138
- All 7 topologies generated with correct component types

**Unconditional Generation (210 samples)**:
- Structural validity: **77.1%** (162/210)
- Avg components per circuit: 5.2
- Unique component combos: 23
- Diversity score: 0.142
- All 7 topologies represented in output

#### Key Observations
1. **Structural tokens saturate immediately** — 77.3% struct accuracy from epoch 1, never improves. The model quickly memorizes the finite set of component/spec/topology tokens.
2. **Value accuracy is the bottleneck** — climbs from 44.6% → 66.2% over 100 epochs. This is the harder task (500 bins) and the primary target for RL improvement.
3. **Best model at epoch 68** — validation loss plateaus at ~1.28 from epoch 55-68, then mild overfitting begins.
4. **Conditioned >> Unconditioned** — 100% vs 77.1% validity shows spec conditioning strongly guides structurally valid generation.
5. **Diversity is moderate** — 29 unique component combos across 210 conditioned samples suggests some mode collapse per topology.

#### Checkpoints
- `checkpoints/arcs_small/best_model.pt` — Best val loss (epoch 68), 75MB
- `checkpoints/arcs_small/final_model.pt` — Epoch 100, 25MB
- `checkpoints/arcs_small/checkpoint_epoch{25,50,75,100}.pt` — Periodic saves
- `checkpoints/arcs_small/history.json` — Full training metrics
- `checkpoints/arcs_small/eval_conditioned.json` — Conditioned eval results
- `checkpoints/arcs_small/eval_unconditioned.json` — Unconditioned eval results

#### Pre-Existing Models (from `circuitgenie/` module — DIFFERENT architecture)
These are from earlier explorations using a simpler parameter-prediction approach:
- `checkpoints/` (v1): vocab=157, d=128, layers=4 — flat param model
- `checkpoints_v2/`: Same arch + RL fine-tuning (5K RL steps with SPICE reward)
- `checkpoints_v3/`: vocab=161, d=256, layers=6 — Eulerian walk model

**Decision: These are NOT compatible with the new ARCS architecture (676-token vocab, SwiGLU, RMSNorm). They will be kept as baselines for comparison but are not used going forward.**

---

## Major Design Decisions

### D1: Native Component Tokens vs. Text-Based SPICE (Foundational)
**Decision**: Each token IS a circuit element with embedded value, not a text character.
**Rationale**: Text-LLM approaches (CircuitSynth, AutoCircuit-RL) predict characters — a single wrong digit destroys a component value. Our approach makes component values first-class, with 500 log-discretized value bins spanning 1pF to 1MΩ. The model predicts "which bin" not "which character sequence."
**Impact**: 676-token vocabulary vs. ~100 for text. Much shorter sequences (25-30 tokens vs. hundreds of characters for a netlist).

### D2: Spec Prefix → Natural Conditioning (No Separate Conditioning Module)
**Decision**: Performance specs are encoded as prefix tokens in the same sequence. The model learns spec→circuit mapping through causal attention — spec tokens attend to themselves, circuit tokens attend to all preceding spec tokens.
**Rationale**: Simpler than a separate cross-attention conditioning module. Same architecture for unconditional and conditional generation. Follows GPT-2 "prompt → completion" paradigm.
**Format**: `START → TOPO → SEP → SPEC_VIN → val → SPEC_VOUT → val → ... → SEP → COMP → VAL → ... → END`

### D3: Value-Weighted Loss (5×)
**Decision**: Apply 5× loss weight to value token positions.
**Rationale**: In a typical 27-token sequence, ~11 are value tokens and ~16 are structural (topology, spec names, component types). The model quickly learns structural tokens (finite set, predictable patterns) but struggles with values (500 bins, continuous distribution). 5× weight forces the model to invest capacity in value prediction, which is the harder and more important task.
**Evidence**: From test_model.py smoke test — structural tokens converge faster than value tokens.

### D4: SwiGLU + RMSNorm (Modern Transformer Stack)
**Decision**: Use SwiGLU FFN and RMSNorm instead of GELU + LayerNorm.
**Rationale**: LLaMA showed ~30% quality improvement per parameter with SwiGLU. For our small models (6.5M), extracting maximum quality per parameter matters. RMSNorm is simpler and faster than LayerNorm with negligible quality difference.
**Trade-off**: SwiGLU has 3 weight matrices per FFN layer (w1, w2, w3) vs. 2 for standard — 50% more FFN parameters. But quality gain outweighs.

### D5: Small Model First (6.5M, Not 50-100M)
**Decision**: Start with the "small" config (d=256, layers=6, heads=4).
**Rationale**: Phase 1 data is ~35K samples (with augmentation ~175K). A 50M parameter model would severely overfit. The small model is appropriately sized for the dataset. Will scale up in Phase 4 when we add more circuit families and 10× more data.
**Parameter breakdown**: 
- Embeddings: 208K (token + position + type)
- Attention: 1.57M (6 layers × QKV + output proj)
- FFN: 4.72M (6 layers × SwiGLU with 3 matrices)
- Norm: 3.3K

### D6: Include Invalid Circuits in Training
**Decision**: Keep simulation failures and out-of-spec circuits in the dataset (labeled with `valid=False`).
**Rationale**: The model should learn what doesn't work, not just what does. Invalid circuits provide negative examples — "these component values led to instability/failure." This is unique to ARCS; all prior work (AnalogGenie, CircuitSynth, AutoCircuit-RL) discards failures.
**Implementation**: Dataset class has `valid_only` flag. Default training uses ALL samples; evaluation separates valid vs. invalid accuracy.

### D7: SW Model Switches (Not Behavioral Sources)
**Decision**: All 7 topologies use ngspice SW model (`S1 nodeA nodeB ctrl 0 SMOD`).
**Rationale**: Behavioral current source switches (`Bsw ... I = V(node)/R * V(pwm)`) caused energy accounting errors with coupled inductors (flyback showed 629% efficiency). The SW model properly simulates on/off resistance and integrates correctly with SPICE's matrix solver.
**Parameters**: `.model SMOD SW(RON=r_dson ROFF=1e6 VT=2.5 VH=0.1)`

### D8: 500 Switching Periods (Not 200)
**Decision**: Simulate 500 periods, measure from period 400-500.
**Rationale**: Initial condition energy on output capacitor (`IC=vout_target`) created artificial energy, causing efficiency >100% at 200 periods when IC transient hadn't fully decayed. 500 periods ensures steady-state. The measurement window (400-500) gives 100 periods of averaging.
**Trade-off**: ~2.5× slower simulation. Buck: 3 it/s → ~0.33s per sample.

### D9: Inverted Topology Support (Ćuk, Buck-Boost)
**Decision**: Output voltage comparison uses `abs(vout_avg) vs abs(vout_target)` for vout_error_pct.
**Rationale**: Buck-boost and Ćuk produce negative output voltages. The raw `abs(vout_avg - vout_target)` formula gives huge errors when target is negative (e.g., vout=-4.5V, target=-5.0V → error=|(-4.5)-(-5.0)|/5.0=10%, correct). But the previous formula was using `abs(vout_avg - vout_target)` where vout_avg was already abs'd, giving wrong results.

### D10: Component Shuffle Augmentation (Temporary)
**Decision**: For data augmentation, randomly shuffle the order of (COMP, VAL) pairs in the circuit body.
**Rationale**: Full Eulerian walk augmentation requires the `euler.py` module integration with the dataset pipeline. As a quick alternative, shuffling component order teaches the model that circuits are about what components exist and their values, not the order they're listed. This is a stepping stone to proper Eulerian augmentation.
**Factor**: 5× augmentation (1 original + 4 shuffled orderings).

### D11: REINFORCE over PPO/GRPO for Phase 3
**Decision**: Use REINFORCE with learned baseline + KL penalty instead of PPO or GRPO.
**Rationale**: REINFORCE is simpler to implement and debug. Each RL step requires 8 SPICE simulations (~8 sec), making each step slow enough that sample efficiency matters less than stability. The KL penalty from a frozen reference model serves the same role as PPO's clipping — preventing catastrophic divergence. Can upgrade to PPO in Phase 4 if REINFORCE plateaus.
**Trade-off**: Higher variance than PPO, but compensated by the baseline and small batch size relative to the slow simulation loop.

### D12: Log-Prob Mean (Not Sum) for Policy Gradient
**Decision**: Normalize policy gradient by sequence length using `log_probs.mean()` instead of `.sum()`.
**Rationale**: Generated sequences vary in length (5-30 tokens). Using sum would weight longer sequences disproportionately, causing unstable gradients. Mean normalization keeps gradient magnitude consistent regardless of generation length. Also added log-prob clamping (min=-20) to prevent -inf from top-k filtering mismatches between sampling and gradient passes.

---

## Architecture Comparison: Old vs. New

| Property | Old (`circuitgenie/`) | New (`src/arcs/`) |
|----------|----------------------|-------------------|
| Vocabulary | 157-161 tokens | 676 tokens |
| Representation | Flat param list OR Eulerian walk | Native component tokens + specs |
| Values | 128 log bins (1e-2 to 1e6) | 500 log bins (1e-12 to 1e6) |
| Value range | Limited (no pF/nH) | Full analog range |
| Spec conditioning | No | Yes (prefix tokens) |
| FFN type | GELU | SwiGLU |
| Normalization | LayerNorm | RMSNorm |
| Loss weighting | Optional value_weight | 5× on value tokens by default |
| Weight tying | Yes | Yes |
| Token types | No | Yes (7 categories) |
| Invalid examples | Separate valid flag | Included in training by default |

---

## README Roadmap Alignment

### Phase 1: Data Generation & Proof of Concept ✅
- [x] Build parameterized SPICE templates for 7 power converter topologies
- [x] Write data generation pipeline (random sweep + simulate + extract metrics)
- [x] Generate ~14K circuit samples with performance labels *(35K generated, 16.4K valid)*
- [x] Design tokenizer vocabulary (components + values + pins + specs)
- [x] Implement Eulerian circuit representation + augmentation

### Phase 2: Model Training ✅
- [x] Implement GPT-style decoder model with circuit tokenizer — `model.py`
- [x] Train on all circuit sequences with spec conditioning *(100 epochs, 175K samples)*
- [x] Add spec-conditioning (spec prefix tokens) — built into model + train.py
- [x] Evaluate: 100% conditioned validity, 77.1% unconditioned, 29 unique combos

### Phase 3: SPICE-in-the-Loop RL ✅
- [x] Implement reward function from SPICE simulation metrics — `rl.py`
- [x] RL fine-tuning (REINFORCE w/ KL penalty + baseline) — 5000 steps complete
- [x] Compare: pre-trained only vs. RL-refined (vout error: 53% → 3-9%)

### Phase 4: Tier 2 Expansion ✅
- [x] 9 new templates (amplifiers, filters, oscillators)
- [x] 18K Tier 2 samples (88% yield), combined retraining (val_loss=1.237)
- [x] RL v2 with adaptive KL (96-100% validity)
- [x] Simulation-based evaluation for all 16 topologies

### Phase 5: Paper ✅ (experiments done, writing remains)
- [x] Baselines: RS (200 trials, 81.2% valid, 7.28 reward) + GA (pop=30, 80.0%, 7.48)
- [x] Ablation studies: RL effect, spec conditioning effect, data expansion effect
- [x] Demo CLI: interactive circuit design tool
- [x] pytest suite: 96 tests, all passing
- [ ] Write paper targeting DAC / ICCAD / NeurIPS workshop

### Phase 5b: Enhanced Architectures ✅
- [x] TwoHeadARCSModel (6.8M params): separate structure + value heads
- [x] GraphTransformerARCSModel (6.8M params): topology-aware causal attention
- [x] Model factory + unified `--model-type` flag across all entry points
- [x] 30 new tests (96 total, all passing)

### Phase 6: Enhanced Model Training (next)
- [ ] Train two_head + graph_transformer on combined data
- [ ] Compare all 3 architectures on 160-spec evaluation
- [ ] RL fine-tune best enhanced model

---

## Additional Design Decisions (Phases 4-5b)

### D13: Valid-Only Training for Combined Model
**Decision**: Use `--valid-only` flag for combined retraining (Phase 4).
**Rationale**: With 53K samples (32K valid), training on invalids dilutes quality without adding enough negative-example benefit. Phase 2 included invalids (D6) for diversity, but Phase 4 prioritizes generation quality with a larger valid set.

### D14: Adaptive KL for RL v2
**Decision**: KL coefficient auto-increases when KL divergence exceeds 2.0 nats target.
**Rationale**: Phase 3 RL had validity degradation (100% → 22%) because fixed KL=0.1 allowed too much policy drift. Adaptive KL keeps the model close to the pre-trained distribution, preserving structural knowledge while optimizing values.

### D15: Topology-Aware Adjacency vs. Heuristic Position Adjacency
**Decision**: Hardcoded `TOPOLOGY_ADJACENCY` tables derived from SPICE schematics, NOT position-based heuristics.
**Rationale**: An earlier attempt used "components within 2 positions are adjacent" — this is wrong because component order in the sequence doesn't reflect circuit connectivity. A buck's inductor connects to the MOSFET (positions 0 and 3), not to the capacitor (position 1). Correct adjacency requires knowledge of the actual circuit topology, which we have from our SPICE templates.

### D16: Two-Head Architecture (Separate Value Head)
**Decision**: Two output heads — weight-tied structure head + independent value head with SiLU MLP + residual.
**Rationale**: Structure tokens (COMP_X, TOPO_X, specials) come from a finite categorical set. Value tokens span 500 bins in a quasi-continuous distribution. These loss landscapes benefit from independent capacity. The structure head can leverage weight tying (shared embedding matrix), while the value head gets its own MLP to learn the value distribution without being constrained by the embedding structure.

### D17: Graph Attention Bias over GNN Message Passing
**Decision**: Inject topology as attention bias terms, not as a separate GNN module.
**Rationale**: GNN message passing (like AnalogGenie's graph-to-walk approach) requires breaking the autoregressive property. Instead, we add learnable `adj_bias` (per-head scalar for adjacent pairs) and `edge_type_bias` (component-type pair → per-head bias) directly into the causal attention scores. This preserves the autoregressive generation property while still allowing the model to learn that adjacent circuit components should attend to each other more strongly.

### D18: Model Factory Pattern
**Decision**: Unified `create_model` / `load_model` factory with `model_type` stored in checkpoints.
**Rationale**: With 3 model types, all entry points (train, rl, evaluate, demo) need to handle all types. A factory pattern with auto-detection from checkpoint metadata avoids code duplication and ensures backward compatibility (old checkpoints without `model_type` key default to "baseline").

---

## Phase 3: SPICE-in-the-Loop RL

### Status: ✅ TRAINING COMPLETE

#### RL Architecture
- **Algorithm**: REINFORCE with learned baseline + KL divergence penalty
  - README originally planned PPO/GRPO; REINFORCE chosen as simpler first step
  - KL penalty (coeff=0.1) prevents catastrophic forgetting of pre-trained knowledge
  - Entropy regularization (coeff=0.01) maintains exploration
- **Reward function** (max 8.0):
  - Structure valid: 1.0 (component types match topology)
  - Simulation converges: 1.0 (ngspice doesn't crash)
  - Vout accuracy: 3.0 (graded: <5%→3.0, <10%→2.0, <20%→1.0)
  - Efficiency: 2.0 (graded: >90%→2.0, >80%→1.5, >70%→1.0, >50%→0.5)
  - Low ripple: 1.0 (graded: <2%→1.0, <5%→0.5)
- **Inverse mapping**: `components_to_params()` — decoded sequence → topology params → SPICE netlist

#### Training Config
- Checkpoint: `checkpoints/arcs_small/best_model.pt` (epoch 68)
- Steps: 5000, Batch: 8, LR: 1e-5, Temperature: 0.8, Top-k: 50
- Save interval: 500, Eval interval: 100, Log interval: 10
- Duration: 5000 steps in 12.3 hours on M3 MacBook Air

#### Evaluation Trajectory (Pre-trained → RL)
| Step | Reward | Sim Valid | Vout Err | Efficiency | Notes |
|------|--------|-----------|----------|------------|-------|
| Init | 4.10 | 28% | 53.3% | 1.73 | Pre-trained baseline |
| 500 | 4.43 | 36% | 43.2% | 0.86 | Early improvement |
| 1000 | 5.09 | 40% | 28.5% | 12.6 | Vout improving fast |
| 1600 | 5.75 | **54%** | 9.3% | 96.9 | Peak sim_valid |
| 1800 | 6.10 | 44% | 8.8% | 156 | |
| 2500 | 6.60 | 34% | 13.6% | 315 | |
| 2800 | 6.93 | 30% | **4.2%** | 418 | Near-peak reward |
| 3300 | 6.67 | 34% | 9.4% | 4.75 | Eff stabilizes |
| 4000 | 6.68 | 28% | 4.6% | 4.51 | |
| 4700 | 6.53 | 30% | **3.1%** | 3.97 | Best vout accuracy |
| **4800** | **7.02** | 12% | 4.0% | 3.57 | **Best reward** |
| 5000 | 5.64 | 22% | 8.9% | 3.94 | Final |

#### Final Results
- **Best reward**: 7.02/8.0 (step 4800)
- **Final eval**: reward=5.64, struct_valid=86%, sim_success=84%, sim_valid=22%, vout_err=8.9%

#### Pre-trained vs. RL Comparison
| Metric | Pre-trained | RL (best) | RL (final) | Change |
|--------|-------------|-----------|------------|--------|
| Reward | 4.10 | **7.02** | 5.64 | +71% |
| Vout Error | 53.3% | **3.1%** | 8.9% | **17× better** |
| Sim Valid | 28% | 54% (step 1600) | 22% | ↓ degraded |
| Struct Valid | 100% | 86% | 86% | ↓ slight regression |
| KL Divergence | 0.0 | — | ~2.0 | Significant drift |

#### Key Observations
1. **Vout accuracy is the big win** — 53% → 3-9% error. The model learned to generate component values that produce the correct output voltage. This was the primary bottleneck identified in Phase 2.
2. **Sim valid rate degraded** — peaked at 54% (step 1600) then declined to 22%. The reward function optimizes for circuits that simulate well, but KL drift caused the model to lose some structural knowledge. The KL coeff (0.1) may need to be higher.
3. **Efficiency metric is unreliable** — showed wild swings (0.86 → 2765 → 3.9). Some topologies (especially flyback/forward with coupled inductors) produce nonsensical efficiency values. The reward function caps efficiency reward at 2.0, so this doesn't corrupt training, but the reported metric is noisy.
4. **Best checkpoint != final** — Best reward at step 4800 (7.02), but final eval at step 5000 is lower (5.64). Eval-time stochastic sampling means high variance between evals.
5. **Baseline converged to ~6.4** — training reward averaged ~6.0-7.0 in the final 1000 steps, meaning the model consistently generates circuits scoring 75-88% of max reward.

#### Bug Fixes Applied
1. **loss=inf** — Generated tokens could fall outside the top-k set during
   gradient recomputation, producing -inf log-probs. Fixed by:
   - Clamping log_probs to min=-20.0 (≈ prob 2e-9, finite but negligible)
   - Using `log_probs.mean()` instead of `.sum()` for sequence-length normalization
   - Safety check: skip gradient update if loss is non-finite

#### Checkpoints
- `checkpoints/arcs_rl/best_rl_model.pt` — Best reward (step 4800, reward 7.02)
- `checkpoints/arcs_rl/final_rl_model.pt` — Final step 5000
- `checkpoints/arcs_rl/rl_checkpoint_step{500..5000}.pt` — Periodic saves (every 500)
- `checkpoints/arcs_rl/rl_history.json` — Full training history

#### AnalogGenie Comparison Notes
- AnalogGenie: 11.8M params, 6-layer GPT (384 dim), 1029-token pin-level vocab, block_size=1024
  - No values, no spec conditioning, no RL — pure topology generation
  - Exhaustive Eulerian walk augmentation (up to 2000 walks/circuit vs. our 5× shuffle)
  - Data: 3502 circuits → 227K augmented sequences (90/10 train/val split)
  - Training: 100K iterations, batch 64, lr 3e-4, vanilla CE loss
  - Smart metric: filtered loss (excluding TRUNCATE/PAD tokens)
- ARCS advantages: native value tokens, spec conditioning, SPICE-in-the-loop RL, broader scope
- Future improvement: integrate proper Eulerian walk augmentation from `euler.py` (Phase 4)

---

## Phase 4: Tier 2 Expansion

### Status: ✅ COMPLETE

#### New Topologies (9 signal-processing circuits)

| Topology | Samples | Valid | Yield | Metrics |
|----------|---------|-------|-------|---------|
| Inverting Amp | 2,000 | ~1,760 | 88% | Gain, BW |
| Non-inverting Amp | 2,000 | ~1,760 | 88% | Gain, BW |
| Instrumentation Amp | 2,000 | ~1,760 | 88% | Gain |
| Differential Amp | 2,000 | ~1,760 | 88% | Gain |
| Sallen-Key LP | 2,000 | ~1,760 | 88% | Cutoff, Q |
| Sallen-Key HP | 2,000 | ~1,760 | 88% | Cutoff, Q |
| Sallen-Key BP | 2,000 | ~1,760 | 88% | Center, BW, Q |
| Wien Bridge | 2,000 | ~1,760 | 88% | Oscillation freq |
| Colpitts | 2,000 | ~1,760 | 88% | Oscillation freq |
| **Total Tier 2** | **18,000** | **~15,842** | **88%** | |

#### Combined Dataset
- Phase 1 (Tier 1): 35,000 generated, 16,400 valid
- Phase 2 (Tier 2): 18,000 generated, ~15,842 valid
- **Combined**: 53,000 generated, ~32,242 valid, ~161,000 augmented (5×)
- Tokenizer expanded: 676 → 686 tokens (new TOPO_* and SPEC_* tokens)

#### Combined Model Retraining
- Fresh training (no resume — vocab changed from 676→686)
- Same config: small model, 100 epochs, batch 64, lr 3e-4, cosine decay
- **Best val_loss: 1.237** (epoch 64) — improved from 1.279 (Phase 2, Tier 1 only)
- `--valid-only` flag: trained on valid-only samples (32K instead of 53K) to improve quality

#### RL v2 (Adaptive KL + Structural Bonus)
- Fixed Phase 3 RL's validity degradation problem (22% at end of training)
- Key changes:
  - `struct_bonus=0.5` — extra reward for structurally valid circuits
  - Adaptive KL: coefficient increases if KL divergence exceeds target
  - Result: **96-100% structural validity** maintained throughout training
- Best reward: improved from earlier RL
- Checkpoints: `checkpoints/arcs_rl_v2/best_rl_model.pt`

---

## Phase 5: Paper & Evaluation

### Status: ✅ COMPLETE (code & experiments; paper writing remains)

#### Baselines (160 conditioned specs, all 16 topologies)

| Method | Sims/Design | Sim Success | Sim Valid | Avg Reward |
|--------|-------------|-------------|-----------|------------|
| Random Search (N=200) | 200 | 100.0% | 81.2% | 7.282 |
| Genetic Algorithm (pop=30, 20 gens) | 630 | 100.0% | 80.0% | 7.477 |
| **ARCS (supervised)** | **1** | 66.9% | 43.8% | 3.423 |
| **ARCS + RL** | **1** | 71.2% | 55.0% | 3.644 |

**Key insight**: ARCS trades per-design optimality for **amortized speed** — a single
20ms forward pass vs. 200-630 SPICE simulations (1-5 min).

#### Ablation Studies (160 specs each)

| Variant | Sim Valid | Avg Reward | Δ vs Full |
|---------|-----------|------------|-----------|
| ARCS + RL (full) | 52.5% | 3.488 | — |
| Supervised only (no RL) | 46.9% | 3.240 | −7.1% |
| No spec conditioning | 38.8% | 2.597 | −25.5% |
| Tier 1 only (7 topologies) | 20.6% | 3.774 | lower valid, higher per-design reward |

#### Demo CLI (`arcs.demo`)
- Interactive mode: `--interactive`
- Single design: `--topology buck --vin 12 --vout 5 --iout 1 --simulate`
- Multi-candidate: `-n 5 --simulate` ranks by SPICE reward
- All 16 topologies supported

#### pytest Suite
- 96 tests across 6 test files (test_model, test_model_enhanced, test_model_pytest, test_rl, test_simulate, test_tokenizer, test_evaluate)
- Full CI green: `PYTHONPATH=src python -m pytest tests/ -v`
- Coverage: tokenizer, model, evaluation, RL, simulation, enhanced architectures

---

## Phase 5b: Enhanced Architectures

### Status: ✅ COMPLETE (code + tests; training pending)

#### Motivation
Original README planned two architecture upgrades beyond the baseline GPT:
1. **Two-Head Architecture**: Decouple structure prediction from value regression
2. **Graph Transformer**: Inject circuit topology structure into attention

After inspecting AnalogGenie's codebase (cloned to `reference/AnalogGenie/`), key insight:
AnalogGenie uses **pin-level tokens** where adjacency IS the walk sequence. ARCS uses
**component-level tokens**, so adjacency must be INJECTED as structural attention bias.
This is actually an advantage — explicit inductive bias from known topology graphs.

#### TwoHeadARCSModel (6.8M params)

| Component | Params | Notes |
|-----------|--------|-------|
| Shared backbone | 6,505K | Same as baseline (embeddings + 6 transformer blocks) |
| Structure head | 0 | Weight-tied to token embeddings (same as baseline) |
| Value projection | 131K | SiLU MLP (d_model → d_model) + residual |
| Value head | 176K | Independent linear (d_model → vocab_size) |

- **Training**: `value_mask` routes loss — structural tokens use structure head, value tokens use value head
- **Generation**: Component tokens sampled from structure head; value tokens from value head
- **Rationale**: Structure prediction and value regression have fundamentally different loss landscapes — one is categorical (finite set of component types), the other is quasi-continuous (500 value bins). Decoupling allows independent capacity allocation.

#### GraphTransformerARCSModel (6.8M params)

| Component | Params | Notes |
|-----------|--------|-------|
| Walk position embedding | 8K | 32 positions for circuit body traversal order |
| Graph attention bias | 432 | `adj_bias` (4 scalars) + `edge_type_bias` (17×4 embedding) per layer |
| Two-head output | 307K | Inherited from TwoHeadARCSModel |
| Rest (backbone) | 6,505K | Same structure as baseline |

- **TOPOLOGY_ADJACENCY**: Hardcoded for all 16 topologies — e.g., buck: L↔MOS, L↔CAP, CAP↔ESR
  - Derived from actual SPICE schematics, not heuristic position-based adjacency
  - Indexed by component order in `templates.py` bounds dicts
- **GraphAwareCausalAttention**: Standard causal attention + learned `adj_bias` (per-head scalar for circuit-adjacent pairs) + `edge_type_bias` (component-type pair → per-head bias)
- **Rationale**: AnalogGenie bakes adjacency into sequence structure via Eulerian walks. ARCS generates in component order with values, so topology-aware attention bias is the cleaner inductive bias.

#### Model Factory
- `create_model(model_type, config)` — instantiates baseline / two_head / graph_transformer
- `load_model(ckpt_path, device)` — returns `(model, config, model_type)` with auto-detection
- All entry points unified: `--model-type` flag in train.py, rl.py, evaluate.py, demo.py

#### Tests (30 new, 96 total)
- `TestTopologyAdjacency`: All 16 topologies present, valid tuple pairs, indices in range
- `TestTwoHeadModel`: Instantiation, param groups, forward ± targets, value_mask routing, generate
- `TestGraphTransformerModel`: Forward ± graph features, `compute_graph_features`, adjacency correctness for buck, generate ± tokenizer
- `TestModelFactory`: create all 3 types, save/load roundtrip, backward-compatible loading
- `TestGraphFeaturesEdgeCases`: Empty sequence, unknown topology, mixed-topology batch

---

## Next Steps (Priority Order)
1. ~~Phase 1: Data generation~~ ✅
2. ~~Phase 2: Model training~~ ✅ (100 epochs, best val_loss=1.279)
3. ~~Phase 3: RL~~ ✅ (5000 steps, best reward 7.02/8.0)
4. ~~Phase 4: Tier 2 expansion~~ ✅ (9 new topologies, combined val_loss=1.237)
5. ~~Phase 5: Baselines, ablations, demo~~ ✅
6. ~~Phase 5b: Enhanced architectures~~ ✅ (code + 96 tests)
7. **Phase 6: Train & evaluate enhanced models**
   - Train two_head on combined data (100 epochs)
   - Train graph_transformer on combined data (100 epochs)
   - Compare all 3 architectures on 160-spec evaluation
   - RL fine-tune the best enhanced model
8. **Phase 7: Paper**
   - Architecture comparison table (3 models × metrics)
   - Final paper targeting DAC / ICCAD / NeurIPS workshop
