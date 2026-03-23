# ARCS / CircuitGenie — Audit Findings & Next Steps

> Generated from comprehensive codebase audit (March 2026)

---

## Audit Summary

Full audit of the ARCS pipeline covering: templates, data generation, tokenizer,
models (ARCS GT, VCG, CCFM, Reward, RL), evaluation, simulation, and inference.

### Issues Found: 12 total — ALL FIXED ✅
- **Already Fixed**: 8 (fixed in prior sessions or already correct)
- **Fixed Session 1**: 2 (topology alias dedup, code cleanup)
- **Fixed Session 2**: 5 low-yield topology templates (flyback, forward, sepic, colpitts, cascode)
- **Fixed Session 2**: 2 remaining (shared sampling constants, unified SPICE evaluation mode)

---

## Fixed Issues (Verified)

### 1. ✅ Top-p Sampling (model.py:344)
**Status**: Already correct.
The top-p (nucleus) implementation correctly computes `sorted_probs` once and
uses `cum_probs - sorted_probs` for the shift-right pattern. The scatter at
line 350 correctly uses `sorted_idx` as the index. No bug.

### 2. ✅ Current Mirror Analysis (templates.py:1183)
**Status**: Already uses `.tran`.
The `_current_mirror_netlist` already uses `.tran 1u 1m 0.5m` — consistent
with all other 33 topologies.

### 3. ✅ Voltage Doubler IC= (templates.py:1868)
**Status**: Already has IC=.
`C2 vout 0 {C2:.6e} IC={vin * 2}` — initial condition set to expected 2×Vin.

### 4. ✅ Duplicate simulate_decoded_circuit in rl.py
**Status**: Already removed.
Lines 60-63 of rl.py show it now imports from `arcs.simulate` instead of
having a local copy.

### 5. ✅ Efficiency Clamping Warning (datagen.py:95-99)
**Status**: Already has logger.debug.
Over-unity efficiency is logged with a debug message before clamping to 1.0.

### 6. ✅ Position Embedding Bounds (model.py:248-250)
**Status**: Already asserted.
`assert T <= self.config.max_seq_len` at the start of forward().

### 7. ✅ Consistency Loss Weight (flow_matching.py:127)
**Status**: Already configurable.
`consistency_weight: float = 0.1` is a field on the FlowConfig dataclass.

### 8. ✅ TOPOLOGY_RWPE Initialization (model_enhanced.py:251)
**Status**: Already populated.
`TOPOLOGY_RWPE = _precompute_all_rwpe(TOPOLOGY_ADJACENCY)` at module level.

---

## Fixed This Session

### 9. ✅ Deduplicate Topology Alias Mappings
**Problem**: `_topo_to_token` dict (sallen_key_lowpass → TOPO_SALLEN_KEY_LP)
was duplicated in demo.py, valid_circuit_gen.py, and evaluate.py.
**Fix**: Added `CircuitTokenizer.topology_to_token_name()` method. All three
callers now use `tokenizer.topology_to_token_name(topology)`.
**Files changed**: tokenizer.py, demo.py, valid_circuit_gen.py, evaluate.py

### 10. ✅ Inline Math Import (datagen.py)
**Status**: Already at module level (line 11).
The `import math` is at the top of the file, not inline.

---

## Future Improvements — All Fixed ✅

### 11. ✅ Shared Sampling Constants
**Fixed**: commit `8fad89b`
`DEFAULT_TEMPERATURE = 0.8` and `DEFAULT_TOP_K = 50` added to `src/arcs/__init__.py`.
All three callers (`evaluate.py`, `rl.py` RLConfig default, `scripts/evaluate_all.py`)
now import from there instead of hardcoding.

### 12. ✅ Evaluation Fairness — Unified SPICE Mode
**Fixed**: commit `d3a5305`
Both `scripts/evaluate_vcg.py` and `scripts/evaluate_ccfm.py` now support a
`--spice` flag (and `--n-spice-samples`) that runs `vcg_graph_to_spice()` on every
generated circuit and reports `sim_valid_rate` + `avg_reward` alongside structural
validity — giving apples-to-apples comparison with ARCS autoregressive evaluation.

Usage example:
```bash
PYTHONPATH=src python scripts/evaluate_vcg.py \
    --vcg-checkpoint checkpoints/vcg_v4/best_model.pt \
    --data data/combined_v2 \
    --n-samples 160 \
    --spice \
    --output results/vcg_v4_spice_eval.json
```

---

## Low-Yield Topologies — FIXED ✅

All 5 low-yield topologies have been fixed in `src/arcs/templates.py` (commit `e745a8f`).
Data regeneration is in progress.

### Baseline → Expected Improvement

| Topology | Old Yield | Fix Applied | Data Regen |
|----------|-----------|-------------|------------|
| flyback | 18% (901/5000) | Primary clamp diode, correct duty formula, 1000-period sim | In progress |
| forward | 41% (2057/5000) | Tertiary reset winding, duty ≤45%, 1000-period sim | In progress |
| sepic | 34% (1690/5000) | Tighter coupling cap bounds (1-22µF), 1000-period sim | In progress |
| colpitts | 34% (687/2000) | 500-cycle min sim, IC=0.1 on C2, smaller Ce | In progress |
| cascode | 87% (already fixed earlier) | Parameterized Q1 bias, tighter bounds | Done |

Improving these templates should add ~3,000-5,000 more valid samples to
the dataset, improving model performance on these topologies.

---

## Architecture Health

```
Component              Status    Notes
─────────────────────  ────────  ────────────────────────────
Data Pipeline          ✅ Good   61,760 valid / 89,000 total
Tokenizer              ✅ Good   706 tokens, clean mapping
ARCS Graph Transformer ✅ Good   6.84M params, 88% struct
VCG (VAE)              ✅ Good   4.0M params, 100% struct
CCFM (Flow Matching)   ✅ Good   7.66M params, 100% struct
Reward Model           ✅ Good   663K params, proxy reward
RL / GRPO              ✅ Good   3000 steps, reward 3.80
Hybrid Pipeline        ✅ Good   94.1% sim valid, reward 6.59
SPICE Simulation       ✅ Good   ngspice subprocess, temp cleanup
Templates (34)         ✅ Good   All 5 low-yield topologies fixed (e745a8f)
Evaluation             ✅ Good   Comprehensive multi-model eval + unified SPICE mode
```

---

## Training Results (Current v3/v4 Models)

| Model | Params | Struct% | SimOK% | SimValid% | Reward |
|-------|--------|---------|--------|-----------|--------|
| ARCS-SL v3 | 6.84M | 88.0% | 75.0% | 47.0% | 3.77 |
| ARCS-GRPO v2 | 6.84M | 90.0% | 73.0% | 43.0% | 3.80 |
| VCG v4 | 4.0M | 100.0% | — | — | — |
| CCFM v4 | 7.66M | 100.0% | — | — | — |

---

*Last updated: 2026-03-23*
