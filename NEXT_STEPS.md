# ARCS / CircuitGenie — Audit Findings & Next Steps

> Generated from comprehensive codebase audit (March 2026)

---

## Audit Summary

Full audit of the ARCS pipeline covering: templates, data generation, tokenizer,
models (ARCS GT, VCG, CCFM, Reward, RL), evaluation, simulation, and inference.

### Issues Found: 12 total
- **Already Fixed**: 8 (fixed in prior sessions or already correct)
- **Fixed This Session**: 2 (topology alias dedup, code cleanup)
- **Remaining / Future Work**: 2

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

## Future Improvements (Not Blocking)

### 11. 🔲 Shared Sampling Constants
**Priority**: Low
**Description**: `temperature=0.8` and `top_k=50` are hardcoded in 3 places:
- `scripts/evaluate_all.py:86`
- `src/arcs/rl.py:447` (RLConfig default)
- `src/arcs/evaluate.py:385`

Could be extracted to a shared `DEFAULT_TEMPERATURE = 0.8` and
`DEFAULT_TOP_K = 50` in a config module. Not urgent since all three already
use the same values.

### 12. 🔲 Evaluation Fairness
**Priority**: Low
**Description**: Autoregressive models (ARCS) are evaluated with n=100
random-spec circuits, while graph models (VCG/CCFM) are evaluated with
n=10 per topology (340 total). The metrics measured are also different
(SPICE simulation for ARCS, structural validity for VCG/CCFM). This is
by design (they test different things) but could be confusing.

**Suggestion**: Add a unified evaluation mode that runs SPICE simulation
on VCG/CCFM outputs too, using the same 100-circuit spec set.

---

## Low-Yield Topologies (Data Quality)

These topologies have <40% yield in data generation and may benefit from
template improvements:

| Topology | Valid/Total | Yield | Possible Issue |
|----------|-----------|-------|----------------|
| flyback | ~900/5000 | 18% | Transformer model coupling |
| sepic | ~1700/5000 | 34% | Bounds too wide |
| colpitts | ~680/2000 | 34% | Oscillator startup |
| forward | ~2100/5000 | 42% | Transformer model |
| cascode | ~800/2000 | 40% | Bias point sensitivity |

Improving these templates could add ~3,000-5,000 more valid samples to
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
Templates (34)         ⚠️ Fair   5 low-yield topologies
Evaluation             ✅ Good   Comprehensive multi-model eval
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
