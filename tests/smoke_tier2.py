"""Smoke test: verify Tier 2 circuits simulate correctly in ngspice."""

from arcs.templates import get_topology
from arcs.spice import NGSpiceRunner
from arcs.datagen import compute_derived_metrics, is_valid_result
import numpy as np

runner = NGSpiceRunner(timeout=30)
rng = np.random.default_rng(42)

CIRCUITS = [
    "inverting_amp", "noninverting_amp", "differential_amp",
    "sallen_key_lowpass", "sallen_key_highpass", "sallen_key_bandpass",
    "wien_bridge", "colpitts",
    # instrumentation_amp tested separately — complex 3-opamp
]

results_summary = []

for name in CIRCUITS:
    t = get_topology(name)
    params = t.sample_parameters(rng)
    netlist = t.generate_netlist(params)
    result = runner.run(netlist)

    print(f"--- {name} ---")
    print(f"  success={result.success}, time={result.sim_time_seconds:.2f}s")

    if result.error_message:
        err = result.error_message[:200].replace("\n", " ")
        print(f"  error: {err}")

    if result.metrics:
        metrics = compute_derived_metrics(
            result.metrics, t.operating_conditions, name
        )
        valid = is_valid_result(metrics, t.operating_conditions, name)
        # Show up to 6 key metrics
        shown = {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in list(metrics.items())[:6]}
        print(f"  metrics: {shown}")
        print(f"  valid={valid}")
        results_summary.append((name, True, valid))
    else:
        print(f"  NO METRICS extracted")
        results_summary.append((name, False, False))

    print()

# Summary
print("=" * 50)
print("SUMMARY")
print("=" * 50)
ok = sum(1 for _, s, _ in results_summary if s)
valid = sum(1 for _, _, v in results_summary if v)
print(f"Simulated: {ok}/{len(CIRCUITS)}, Valid: {valid}/{len(CIRCUITS)}")
for name, success, v in results_summary:
    status = "OK+valid" if v else ("OK" if success else "FAIL")
    print(f"  {name:25s} {status}")
