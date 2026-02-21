# ARCS: Decision Log & Progress Tracker

> Last updated: 2026-02-20

---

## Project Timeline

| Commit | Date | Milestone |
|--------|------|-----------|
| `0b49bc6` | — | Initial commit: README with full research plan |
| `187926e` | — | Phase 1 infrastructure: SPICE pipeline, tokenizer, Euler repr |
| `3f141ef` | — | Fix all 7 topology templates (circuit physics, SW model) |
| `f5468a1` | — | Phase 2: GPT decoder model, dataset, training loop, evaluation |
| `15103f3` | — | Add DECISIONS.md, update README checkboxes |
| `TBD` | 2026-02-20 | Phase 2 training complete, evaluation results, Phase 3 RL begun |

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

### Phase 1: Data Generation & Proof of Concept (Weeks 1-3)
- [x] Build parameterized SPICE templates for 7 power converter topologies
- [x] Write data generation pipeline (random sweep + simulate + extract metrics)
- [x] Generate ~14K circuit samples with performance labels *(35K generated, 16.4K valid)*
- [x] Design tokenizer vocabulary (components + values + pins + specs)
- [x] Implement Eulerian circuit representation + augmentation

### Phase 2: Model Training (Weeks 3-5)
- [x] Implement GPT-style decoder model with circuit tokenizer — `model.py`
- [x] Train on all circuit sequences with spec conditioning *(100 epochs, 175K samples)*
- [x] Add spec-conditioning (spec prefix tokens) — built into model + train.py
- [x] Evaluate: validity rate, spec compliance, diversity
  - Conditioned: 100% validity, 5.3 avg components, 29 unique combos
  - Unconditioned: 77.1% validity, all 7 topologies represented

### Phase 3: SPICE-in-the-Loop RL (Weeks 5-7)
- [🔄] Implement reward function from SPICE simulation metrics — **IN PROGRESS**
- [ ] RL fine-tuning (PPO or GRPO)
- [ ] Compare: pre-trained only vs. RL-refined

### Phase 4-5: Not started

---

## Next Steps (Priority Order)
1. ~~Wait for data gen to complete~~ ✅
2. ~~Launch training~~ ✅ (100 epochs, 27 hours, converged at epoch 68)
3. ~~Evaluate trained model~~ ✅ (100% conditioned validity, 77.1% unconditioned)
4. **Begin Phase 3**: SPICE-in-the-loop RL using best_model.pt as initialization
   - Implement `rl.py` with reward function (SPICE simulation → reward signal)
   - Decode generated tokens → SPICE netlist → simulate → extract metrics
   - Reward = f(vout_error, efficiency, ripple, stability)
   - PPO or GRPO fine-tuning on top of pre-trained model
   - Target: improve value accuracy from 66% and reduce SPICE failure rate
