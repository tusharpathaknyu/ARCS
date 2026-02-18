"""Debug failing topologies."""
import sys
sys.path.insert(0, "src")
import numpy as np
from arcs.templates import get_topology
from arcs.spice import NGSpiceRunner
from arcs.datagen import compute_derived_metrics, is_valid_result

# Test with known-good design parameters
good_params = {
    "buck": {"inductance": 22e-6, "capacitance": 470e-6, "esr": 0.01, "r_dson": 0.05},
    "boost": {"inductance": 100e-6, "capacitance": 100e-6, "esr": 0.01, "r_dson": 0.05},
    "buck_boost": {"inductance": 100e-6, "capacitance": 220e-6, "esr": 0.01, "r_dson": 0.05},
    "cuk": {"inductance_1": 220e-6, "inductance_2": 220e-6, "cap_coupling": 10e-6, "capacitance": 220e-6, "esr": 0.01, "r_dson": 0.05},
    "sepic": {"inductance_1": 220e-6, "inductance_2": 220e-6, "cap_coupling": 10e-6, "capacitance": 220e-6, "esr": 0.01, "r_dson": 0.05},
    "flyback": {"inductance_primary": 500e-6, "turns_ratio": 2.0, "capacitance": 100e-6, "esr": 0.01, "r_dson": 0.05},
    "forward": {"inductance_primary": 500e-6, "turns_ratio": 4.0, "inductance_output": 100e-6, "capacitance": 100e-6, "esr": 0.01, "r_dson": 0.05},
}

for name in ["buck", "boost", "buck_boost", "cuk", "sepic", "flyback", "forward"]:
    t = get_topology(name)
    params = good_params[name]
    netlist = t.generate_netlist(params)
    runner = NGSpiceRunner()
    result = runner.run(netlist, t.metric_names)
    derived = compute_derived_metrics(result.metrics, t.operating_conditions, name)
    all_m = {**result.metrics, **derived}
    valid = is_valid_result(all_m, t.operating_conditions)

    eff = derived.get("efficiency", 0) * 100
    vout = result.metrics.get("vout_avg", 0)
    verr = derived.get("vout_error_pct", 100)
    rip = derived.get("ripple_ratio", 1) * 100

    status = "✅" if valid else "❌"
    print(f"{status} {name:12s} | Vout={vout:8.3f}V (target={t.operating_conditions['vout']}V) | err={verr:.1f}% | eff={eff:.1f}% | ripple={rip:.1f}%")
