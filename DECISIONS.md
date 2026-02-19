# ARCS: Decision Log & Progress Tracker

> Last updated: 2026-02-18

---

## Project Timeline

| Commit | Date | Milestone |
|--------|------|-----------|
| `0b49bc6` | — | Initial commit: README with full research plan |
| `187926e` | — | Phase 1 infrastructure: SPICE pipeline, tokenizer, Euler repr |
| `3f141ef` | — | Fix all 7 topology templates (circuit physics, SW model) |
| `f5468a1` | — | Phase 2: GPT decoder model, dataset, training loop, evaluation |

---

## Phase 1: Data Generation & Infrastructure

### Status: ~80% complete (data gen running)

#### Completed
- **7 SPICE templates** for DC-DC power converters: buck, boost, buck-boost, Ćuk, SEPIC, flyback, forward
- **NGSpice runner** (`spice.py`): batch-mode simulation with output file capture, temp file cleanup
- **Data generation pipeline** (`datagen.py`): random parameter sweep with E-series snapping, SPICE simulation, metric extraction, JSONL persistence
- **Tokenizer** (`tokenizer.py`): 676-token vocabulary (components, values, pins, nets, specs, topologies)
- **Eulerian representation** (`euler.py`): Circuit graph → Eulerian walk for data augmentation

#### In Progress
- **Phase 1 dataset generation**: 5,000 samples × 7 topologies = 35,000 total
  - Buck: ✅ 5,000 done (3,811 valid = 76.2% yield)
  - Boost: ✅ 5,000 done (3,156 valid = 63.1%)
  - Buck-Boost: ✅ 5,000 done (2,033 valid = 40.7%)
  - Ćuk: ✅ 5,000 done (2,791 valid = 55.8%)
  - SEPIC: 🔄 Running (~2% done)
  - Flyback: ⏳ Queued
  - Forward: ⏳ Queued
  - **Running total: 20,000 generated, 11,791 valid (59.0% avg yield)**
  - **ETA: ~2 hours remaining for SEPIC + flyback + forward**

#### Not Started
- Eulerian augmentation for training (code exists, not yet applied to dataset)

---

## Phase 2: Model Training

### Status: Code complete, awaiting data

#### Completed Code
- **`model.py`** — ARCSModel: GPT-style decoder-only transformer
- **`dataset.py`** — CircuitDataset + EulerianAugmentedDataset
- **`train.py`** — Full training pipeline with CLI
- **`evaluate.py`** — Generation + evaluation (validity, diversity, specs)
- **`scripts/train_model.sh`** — Launch script
- All smoke-tested with synthetic data ✅

#### Pre-Existing Models (from `circuitgenie/` module — DIFFERENT architecture)
These are from earlier explorations using a simpler parameter-prediction approach:
- `checkpoints/` (v1): vocab=157, d=128, layers=4 — flat param model
- `checkpoints_v2/`: Same arch + RL fine-tuning (5K RL steps with SPICE reward)
- `checkpoints_v3/`: vocab=161, d=256, layers=6 — Eulerian walk model

**Decision: These are NOT compatible with the new ARCS architecture (676-token vocab, SwiGLU, RMSNorm). They will be kept as baselines for comparison but are not used going forward.**

#### Training Plan (once data gen completes)
```bash
bash scripts/train_model.sh small  # 6.5M params, ~30 min on MPS
```
Then evaluate:
```bash
PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/small/best_model.pt
```

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
- [🔄] Generate ~14K circuit samples with performance labels
  - Currently at 20K generated / 11.8K valid — will exceed 14K target once done
  - README called for 2K per topology; we're doing 5K (more diverse training data)
- [x] Design tokenizer vocabulary (components + values + pins + specs)
- [x] Implement Eulerian circuit representation + augmentation
  - `euler.py` exists; shuffle augmentation in dataset.py; full Eulerian walk TBD

### Phase 2: Model Training (Weeks 3-5)
- [x] Implement GPT-style decoder model with circuit tokenizer — `model.py`
- [⏳] Pre-train on unconditional next-token prediction — code ready, awaiting data
- [x] Add spec-conditioning (spec prefix tokens) — built into model + train.py
- [⏳] Fine-tune for spec → circuit generation — same training loop, spec prefix
- [⏳] Evaluate: validity rate, spec compliance, diversity — `evaluate.py` ready

### Phase 3: SPICE-in-the-Loop RL (Weeks 5-7)
- [ ] Implement reward function from SPICE simulation metrics
  - Note: old `checkpoints_v2/` has RL checkpoints with 5K steps — prior art to build on
- [ ] RL fine-tuning (PPO or GRPO)
- [ ] Compare: pre-trained only vs. RL-refined

### Phase 4-5: Not started

---

## Next Steps (Priority Order)
1. **Wait for data gen to complete** (~2 hours remaining)
2. **Launch training**: `bash scripts/train_model.sh small`
3. **Evaluate trained model**: validity rate, spec compliance, diversity
4. **Begin Phase 3**: SPICE-in-the-loop RL using trained model as initialization
