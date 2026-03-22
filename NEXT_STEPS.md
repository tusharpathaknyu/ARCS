# ARCS/CircuitGenie — Audit Findings & Fixes

Post-retrain comprehensive audit (2026-03-22). All items below have been verified against the actual codebase.

## Status Key
- [x] Fixed
- [ ] Pending

---

## HIGH Priority

### 1. [x] Top-p nucleus sampling bug (`src/arcs/model.py:341-347`)
**Bug**: The scatter at line 347 uses `sorted_logits` as both the source AND index arg, which is wrong — it should scatter the modified sorted values back using `sorted_idx` as the index.
**Impact**: Top-p sampling produces incorrect results. Currently mitigated because all evals use `top_k=50` without top-p, but future use would be broken.
**Fix**: Correct the scatter call and simplify the cumulative probability logic.

### 2. [x] current_mirror uses `.dc` instead of `.tran` (`src/arcs/templates.py:1180-1186`)
**Bug**: current_mirror is the ONLY topology that uses `.dc` analysis with `AVG` measurement (which is meaningless for DC). All other 33 topologies use `.tran`.
**Impact**: Inconsistent simulation semantics. The DC sweep `Vdummy 0 1 1` is a hack that works but doesn't match the transient framework used everywhere else.
**Fix**: Convert to `.tran` analysis with proper measurement window, matching the pattern used by all other topologies.

### 3. [x] Duplicate `simulate_decoded_circuit` in rl.py (`src/arcs/rl.py:66-168`)
**Bug**: Full copy of `simulate.py:simulate_decoded_circuit()` plus a local `SimulationOutcome` dataclass. The rl.py version lacks `normalize_topology()` call that the simulate.py version has (line 394).
**Impact**: Code drift — rl.py version misses topology normalization, so sallen_key_lp etc. won't resolve correctly during RL training.
**Fix**: Delete the duplicate and import from `arcs.simulate`.

---

## MEDIUM Priority

### 4. [x] voltage_doubler C1 missing `IC=` (`src/arcs/templates.py:1861`)
**Bug**: C2 has `IC={vin * 2}` but C1 has no initial condition. For a Villard doubler, C1 charges to Vin on the negative half-cycle, so `IC=Vin` would help convergence.
**Impact**: Slower simulation convergence, potentially lower yield for this topology.
**Fix**: Add `IC={vin}` to C1.

### 5. [x] Efficiency >1.0 silently clamped (`src/arcs/datagen.py:88-90`)
**Bug**: `min(eff, 1.0)` clamps over-unity efficiency without any warning. Over-unity efficiency indicates a netlist or measurement error that should be flagged.
**Impact**: Masks bugs in netlist templates. Training data could contain circuits with silently wrong metrics.
**Fix**: Log a warning when efficiency > 1.0 before clamping.

### 6. [x] Flow matching consistency loss weight hardcoded (`src/arcs/flow_matching.py:604`)
**Bug**: `consistency_loss = F.mse_loss(...) * 0.1` — the 0.1 weight is a magic number.
**Impact**: Not configurable without code change. Makes hyperparameter sweeps harder.
**Fix**: Add `consistency_weight` to CCFMConfig dataclass.

### 7. [x] Topology alias mapping duplicated (`src/arcs/evaluate.py:436-441`)
**Bug**: `_topo_to_token` dict in evaluate.py duplicates `_TOPO_ALIASES` from simulate.py. If one changes, the other won't.
**Impact**: Silent breakage if tokenizer naming conventions change.
**Fix**: Import and use `normalize_topology` from simulate.py instead.

---

## LOW Priority

### 8. [x] Unused imports
- `src/arcs/evaluate.py:34`: `generate_from_specs` imported but never used (removed)
- `src/arcs/rl.py:49`: `OPERATING_CONDITIONS` imported but never used (removed)
- `src/arcs/datagen.py:149,164`: inline `import math` moved to module level
**Fix**: Removed unused imports, moved inline imports to module level.

### 9. Hardcoded temperature/top_k in multiple places (deferred)
- All 20+ callsites use `temperature=0.8, top_k=50` consistently as function parameter defaults.
- Extracting to shared constants would touch 8 files with minimal practical benefit since they're already consistent.
- **Status**: Deferred — risk/reward not worth it for a refactor-only change.

### 10. [x] Inline `import math` in datagen.py
- `src/arcs/datagen.py:149,164`: `import math` inside function body
**Fix**: Move to module-level import.

---

## NOT Issues (False Alarms from Audit)

- **TOPOLOGY_RWPE empty**: Actually populated at line 251 via `_precompute_all_rwpe(TOPOLOGY_ADJACENCY)`
- **RL resume runs extra steps**: Actually correct — `range(start_step, n_steps+1)` resumes from saved step, doesn't add extra
- **Position embedding bounds**: Already asserted at `model.py:248`
- **Loss double-weighting**: Investigated — the per-token cross-entropy is averaged correctly by PyTorch's `F.cross_entropy` with default `reduction='mean'`
