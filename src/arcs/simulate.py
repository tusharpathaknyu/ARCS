"""ARCS simulation module: decode → netlist → SPICE → metrics.

Shared between rl.py (RL training) and evaluate.py (evaluation).
Supports both Tier 1 (power converters) and Tier 2 (signal circuits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from arcs.datagen import compute_derived_metrics, is_valid_result
from arcs.templates import (
    get_topology,
    POWER_CONVERTER_BOUNDS,
    SIGNAL_CIRCUIT_BOUNDS,
    _TIER1_NAMES,
    _TIER2_NAMES,
)
from arcs.spice import NGSpiceRunner


# ---------------------------------------------------------------------------
# Topology name normalization
# ---------------------------------------------------------------------------

_TOPO_ALIASES: dict[str, str] = {
    # Sallen-Key tokenizer abbreviations → template names
    "sallen_key_lp": "sallen_key_lowpass",
    "sallen_key_hp": "sallen_key_highpass",
    "sallen_key_bp": "sallen_key_bandpass",
}


def normalize_topology(name: str) -> str:
    """Map abbreviated topology names to canonical template names."""
    return _TOPO_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Component-to-parameter mapping (inverse of tokenizer._params_to_components)
# ---------------------------------------------------------------------------

COMPONENT_TO_PARAM: dict[str, list[tuple[str, str]]] = {
    # ---- Tier 1: Power converters ----
    "buck": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "boost": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "buck_boost": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "cuk": [
        ("INDUCTOR", "inductance_1"),
        ("INDUCTOR", "inductance_2"),
        ("CAPACITOR", "cap_coupling"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "sepic": [
        ("INDUCTOR", "inductance_1"),
        ("INDUCTOR", "inductance_2"),
        ("CAPACITOR", "cap_coupling"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "flyback": [
        ("INDUCTOR", "inductance_primary"),
        ("TRANSFORMER", "turns_ratio"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    "forward": [
        ("INDUCTOR", "inductance_primary"),
        ("TRANSFORMER", "turns_ratio"),
        ("INDUCTOR", "inductance_output"),
        ("CAPACITOR", "capacitance"),
        ("RESISTOR", "esr"),
        ("MOSFET_N", "r_dson"),
    ],
    # ---- Tier 2: Amplifiers ----
    "inverting_amp": [
        ("RESISTOR", "r_input"),
        ("RESISTOR", "r_feedback"),
    ],
    "noninverting_amp": [
        ("RESISTOR", "r_ground"),
        ("RESISTOR", "r_feedback"),
    ],
    "instrumentation_amp": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r_gain"),
        ("RESISTOR", "r2"),
        ("RESISTOR", "r3"),
    ],
    "differential_amp": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
    ],
    # ---- Tier 2: Filters ----
    "sallen_key_lowpass": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
    ],
    "sallen_key_highpass": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
    ],
    "sallen_key_bandpass": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("RESISTOR", "r3"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
    ],
    # ---- Tier 2: Oscillators ----
    "wien_bridge": [
        ("RESISTOR", "r_freq"),
        ("CAPACITOR", "c_freq"),
        ("RESISTOR", "r_feedback"),
        ("RESISTOR", "r_ground"),
    ],
    "colpitts": [
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
        ("RESISTOR", "r_bias_1"),
        ("RESISTOR", "r_bias_2"),
        ("RESISTOR", "r_emitter"),
        ("RESISTOR", "r_collector"),
    ],
    # ---- Additional Filters ----
    "twin_t_notch": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("RESISTOR", "r3"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
        ("CAPACITOR", "c3"),
    ],
    "state_variable_filter": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("RESISTOR", "r3"),
        ("RESISTOR", "r4"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
    ],
    # ---- Additional Oscillators ----
    "hartley": [
        ("INDUCTOR", "inductance_1"),
        ("INDUCTOR", "inductance_2"),
        ("CAPACITOR", "c1"),
        ("RESISTOR", "r_bias_1"),
        ("RESISTOR", "r_bias_2"),
    ],
    "phase_shift": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("RESISTOR", "r3"),
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
        ("CAPACITOR", "c3"),
    ],
    # ---- Regulators ----
    "shunt_regulator": [
        ("RESISTOR", "r_series"),
        ("RESISTOR", "r_load"),
    ],
    "series_regulator": [
        ("RESISTOR", "r1"),
        ("RESISTOR", "r2"),
        ("RESISTOR", "r_load"),
    ],
    # ---- Additional Amplifiers ----
    "inverting_summing_amp": [
        ("RESISTOR", "r_input1"),
        ("RESISTOR", "r_input2"),
        ("RESISTOR", "r_feedback"),
    ],
    "transimpedance_amp": [
        ("RESISTOR", "r_feedback"),
        ("CAPACITOR", "c_feedback"),
    ],
    # ---- Tier 3: BJT Amplifiers ----
    "common_emitter": [
        ("RESISTOR", "r_collector"),
        ("RESISTOR", "r_base"),
        ("RESISTOR", "r_emitter"),
        ("CAPACITOR", "c_bypass"),
    ],
    "common_collector": [
        ("RESISTOR", "r_base"),
        ("RESISTOR", "r_emitter"),
    ],
    "common_base": [
        ("RESISTOR", "r_collector"),
        ("RESISTOR", "r_emitter"),
    ],
    "cascode": [
        ("RESISTOR", "r_collector"),
        ("RESISTOR", "r_bias1"),
        ("RESISTOR", "r_bias2"),
        ("RESISTOR", "r_emitter"),
    ],
    "current_mirror": [
        ("RESISTOR", "r_ref"),
        ("RESISTOR", "r_emitter"),
    ],
    # ---- Power/Misc Topologies ----
    "half_bridge": [
        ("MOSFET_N", "r_dson_high"),
        ("MOSFET_N", "r_dson_low"),
        ("INDUCTOR", "inductance"),
        ("CAPACITOR", "capacitance"),
    ],
    "push_pull": [
        ("TRANSFORMER", "turns_ratio"),
        ("MOSFET_N", "r_dson"),
        ("CAPACITOR", "capacitance"),
        ("INDUCTOR", "inductance"),
    ],
    "charge_pump": [
        ("CAPACITOR", "c_flying"),
        ("CAPACITOR", "c_output"),
        ("RESISTOR", "r_load"),
        ("RESISTOR", "r_esr"),
    ],
    "voltage_doubler": [
        ("CAPACITOR", "c1"),
        ("CAPACITOR", "c2"),
        ("RESISTOR", "r_diode1"),
        ("RESISTOR", "r_diode2"),
    ],
    "zeta_converter": [
        ("INDUCTOR", "inductance_1"),
        ("INDUCTOR", "inductance_2"),
        ("CAPACITOR", "cap_coupling"),
        ("CAPACITOR", "capacitance"),
        ("MOSFET_N", "r_dson"),
    ],
}

# Merge both bounds dicts for fallback defaults
_ALL_BOUNDS = {**POWER_CONVERTER_BOUNDS, **SIGNAL_CIRCUIT_BOUNDS}


def components_to_params(
    topology: str,
    components: list[tuple[str, float]],
) -> dict[str, float] | None:
    """Inverse of tokenizer._params_to_components().

    Maps a list of (component_type, value) pairs back to a parameter dict
    suitable for TopologyTemplate.generate_netlist().

    Returns None if the component list doesn't match the expected topology
    template (wrong types or insufficient params).
    """
    topology = normalize_topology(topology)
    expected = COMPONENT_TO_PARAM.get(topology)
    if expected is None:
        return None

    # Minimum params required: at least half of expected
    min_params = max(2, len(expected) // 2)

    params: dict[str, float] = {}
    components = list(components)  # copy to allow mutation
    comp_idx = 0

    for expected_type, param_name in expected:
        if comp_idx >= len(components):
            break
        comp_type, comp_val = components[comp_idx]
        if comp_type.upper() == expected_type:
            params[param_name] = comp_val
            comp_idx += 1
        else:
            # Type mismatch — search remaining for a match
            found = False
            for j in range(comp_idx, len(components)):
                if components[j][0].upper() == expected_type:
                    params[param_name] = components[j][1]
                    components = components[:j] + components[j + 1:]
                    found = True
                    break
            if not found:
                # Use geometric mean of bounds as fallback
                bounds = _ALL_BOUNDS.get(topology)
                if bounds:
                    for b in bounds:
                        if b.name == param_name:
                            params[param_name] = math.sqrt(b.min_val * b.max_val)
                            break

    return params if len(params) >= min_params else None


# ---------------------------------------------------------------------------
# Simulation outcome
# ---------------------------------------------------------------------------

@dataclass
class SimulationOutcome:
    """Result of simulating a decoded circuit."""

    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    valid: bool = False
    error: str = ""
    sim_time: float = 0.0


# ---------------------------------------------------------------------------
# Spec-to-conditions mapping
# ---------------------------------------------------------------------------

# Tier 1 specs
_SPEC_TO_COND_POWER = {
    "vin": "vin",
    "vout": "vout",
    "iout": "iout",
    "fsw": "fsw",
}

# Tier 2 specs (from tokenizer _OC_SPEC + _METRIC_SPEC)
_SPEC_TO_COND_SIGNAL = {
    "vin": "vin_amp",       # SPEC_VIN reused for vin_amp in signal circuits
    "cutoff_freq": "freq_test",
    "bandwidth": "freq_test",
    "center_freq": "freq_test",
    "osc_freq": "freq_test",
}


def _get_spec_to_cond(topology: str) -> dict[str, str]:
    """Get the spec→condition mapping for a given topology."""
    if topology in _TIER1_NAMES:
        return _SPEC_TO_COND_POWER
    return _SPEC_TO_COND_SIGNAL


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate_decoded_circuit(
    decoded: Any,  # DecodedCircuit from evaluate.py
    runner: NGSpiceRunner | None = None,
    custom_conditions: dict[str, float] | None = None,
) -> SimulationOutcome:
    """Full pipeline: DecodedCircuit → SPICE netlist → simulate → metrics.

    Works for both Tier 1 power converters and Tier 2 signal circuits.

    Args:
        decoded: DecodedCircuit with topology, specs, and components
        runner: NGSpiceRunner instance (creates one if None)
        custom_conditions: Override operating conditions

    Returns:
        SimulationOutcome with metrics and validity flag
    """
    if not decoded.valid_structure or not decoded.topology:
        return SimulationOutcome(success=False, error="Invalid structure")

    # Normalize topology name (sallen_key_lp → sallen_key_lowpass, etc.)
    topology = normalize_topology(decoded.topology)

    # Get topology template
    try:
        template = get_topology(topology)
    except ValueError as e:
        return SimulationOutcome(success=False, error=str(e))

    # Map components back to parameters
    params = components_to_params(topology, list(decoded.components))
    if params is None:
        return SimulationOutcome(
            success=False, error="Could not map components to params"
        )

    # Determine operating conditions
    conditions = dict(template.operating_conditions)
    # Override with decoded specs if available
    spec_to_cond = _get_spec_to_cond(topology)
    if decoded.specs:
        for spec_key, cond_key in spec_to_cond.items():
            if spec_key in decoded.specs:
                conditions[cond_key] = decoded.specs[spec_key]
    if custom_conditions:
        conditions.update(custom_conditions)

    # Build netlist (temporarily override operating conditions)
    old_conds = template.operating_conditions
    template.operating_conditions = conditions
    try:
        netlist = template.generate_netlist(params)
    except Exception as e:
        template.operating_conditions = old_conds
        return SimulationOutcome(success=False, error=f"Netlist error: {e}")
    finally:
        template.operating_conditions = old_conds

    # Simulate
    if runner is None:
        runner = NGSpiceRunner()
    try:
        sim_result = runner.run(netlist, template.metric_names)
    except Exception as e:
        return SimulationOutcome(success=False, error=f"Sim error: {e}")

    if not sim_result.success:
        return SimulationOutcome(
            success=False,
            error=sim_result.error_message or "Simulation failed",
            sim_time=sim_result.sim_time_seconds,
        )

    # Compute derived metrics
    try:
        metrics = compute_derived_metrics(
            sim_result.metrics, conditions, topology
        )
        valid = is_valid_result(metrics, conditions, topology)
    except Exception as e:
        return SimulationOutcome(
            success=True,
            metrics=sim_result.metrics,
            valid=False,
            error=f"Metric error: {e}",
            sim_time=sim_result.sim_time_seconds,
        )

    return SimulationOutcome(
        success=True,
        metrics=metrics,
        valid=valid,
        sim_time=sim_result.sim_time_seconds,
    )


# ---------------------------------------------------------------------------
# Reward functions (domain-aware)
# ---------------------------------------------------------------------------

def compute_reward(
    decoded: Any,  # DecodedCircuit
    outcome: SimulationOutcome,
    target_specs: dict[str, float] | None = None,
    struct_bonus: float = 1.0,
) -> float:
    """Compute scalar reward from simulation outcome (domain-aware).

    Dispatches to power-converter or signal-circuit reward based on topology.

    Returns:
        Reward in [0, max_reward]
    """
    topology = normalize_topology(decoded.topology) if decoded.topology else ""

    reward = 0.0

    # +struct_bonus for valid structure
    if decoded.valid_structure:
        reward += struct_bonus

    if not outcome.success:
        return reward

    # +1.0 for sim convergence
    reward += 1.0

    _POWER_TOPOS = set(_TIER1_NAMES) | {
        "half_bridge", "push_pull", "charge_pump", "voltage_doubler", "zeta_converter",
    }
    _REGULATOR_TOPOS = {"shunt_regulator", "series_regulator"}
    _MIRROR_TOPOS = {"current_mirror"}

    if topology in _POWER_TOPOS:
        reward += _power_reward(outcome, target_specs)
    elif topology in _REGULATOR_TOPOS:
        reward += _regulator_reward(outcome, topology, target_specs)
    elif topology in _MIRROR_TOPOS:
        reward += _current_mirror_reward(outcome)
    else:
        reward += _signal_reward(outcome, topology)

    return reward


def _power_reward(
    outcome: SimulationOutcome,
    target_specs: dict[str, float] | None = None,
) -> float:
    """Power converter reward: vout accuracy + efficiency + low ripple (max 6.0)."""
    reward = 0.0
    m = outcome.metrics

    # Vout accuracy: 3.0 × max(0, 1 - error/10)
    verr = m.get("vout_error_pct", 100)
    reward += 3.0 * max(0.0, 1.0 - verr / 10.0)

    # Efficiency: 2.0 × efficiency
    eff = m.get("efficiency", 0)
    reward += 2.0 * max(0.0, min(1.0, eff))

    # Low ripple: 1.0 × max(0, 1 - ripple×10)
    rip = m.get("ripple_ratio", 1.0)
    reward += 1.0 * max(0.0, 1.0 - rip * 10.0)

    return reward


def _signal_reward(
    outcome: SimulationOutcome,
    topology: str,
) -> float:
    """Signal circuit reward based on domain (max 6.0).

    Amplifiers: reasonable gain + valid measurement
    Filters:    gain + bandwidth detection
    Oscillators: oscillation amplitude
    """
    reward = 0.0
    m = outcome.metrics

    amp_types = {"inverting_amp", "noninverting_amp", "instrumentation_amp", "differential_amp",
                  "common_emitter", "common_collector", "common_base", "cascode",
                  "inverting_summing_amp", "transimpedance_amp"}
    filter_types = {"sallen_key_lowpass", "sallen_key_highpass", "sallen_key_bandpass",
                    "twin_t_notch", "state_variable_filter"}
    osc_types = {"wien_bridge", "colpitts", "hartley", "phase_shift"}

    if topology in amp_types:
        # Gain exists and reasonable → 3.0
        gain_db = m.get("gain_db", m.get("gain_dc"))
        if gain_db is not None and abs(gain_db) <= 120:
            reward += 3.0
            # Gain magnitude bonus (higher gain within reason → better)
            if abs(gain_db) > 0:
                reward += min(2.0, abs(gain_db) / 30.0)
            # Has bandwidth measurement → +1.0
            if m.get("bw_3db") is not None and m["bw_3db"] > 0:
                reward += 1.0

    elif topology in filter_types:
        # Has gain measurement → 2.0
        gain_dc = m.get("gain_dc")
        if gain_dc is not None:
            reward += 2.0
        # Has bandwidth / cutoff detected → 3.0
        bw = m.get("bw_3db")
        if bw is not None and bw > 0:
            reward += 3.0
        # Reasonable passband gain (not too much attenuation) → 1.0
        if gain_dc is not None and gain_dc > -6:
            reward += 1.0

    elif topology in osc_types:
        # Oscillation detected → 3.0
        vosc = m.get("vosc_pp", 0)
        if vosc >= 0.01:
            reward += 3.0
        # Reasonable amplitude (0.1-20V peak-to-peak) → 2.0
        if 0.1 <= vosc <= 20:
            reward += 2.0
        # Frequency detected → 1.0
        if m.get("f_peak") is not None and m["f_peak"] > 0:
            reward += 1.0

    return reward


def _regulator_reward(
    outcome: SimulationOutcome,
    topology: str,
    target_specs: dict[str, float] | None = None,
) -> float:
    """Voltage regulator reward: output regulation + low ripple (max 6.0).

    shunt_regulator targets v_zener, series_regulator targets v_ref.
    Both measure vout_avg and vout_ripple.
    """
    reward = 0.0
    m = outcome.metrics

    vout_avg = abs(m.get("vout_avg", 0))

    # Determine target voltage from topology or specs
    vtarget = 0.0
    if target_specs:
        vtarget = target_specs.get("v_zener", target_specs.get("v_ref", 0))
    if vtarget <= 0 and topology == "shunt_regulator":
        vtarget = 5.1  # default zener voltage
    elif vtarget <= 0 and topology == "series_regulator":
        vtarget = 2.5  # default ref voltage

    # Vout regulation accuracy: 3.0 × max(0, 1 - error/10)
    if vtarget > 0.01:
        verr_pct = abs(vout_avg - vtarget) / vtarget * 100
        reward += 3.0 * max(0.0, 1.0 - verr_pct / 10.0)
    elif vout_avg > 0.1:
        # At least producing some output
        reward += 1.0

    # Low ripple: 2.0 × max(0, 1 - ripple/vout × 10)
    ripple = abs(m.get("vout_ripple", 0))
    if vout_avg > 0.01:
        rip_ratio = ripple / vout_avg
        reward += 2.0 * max(0.0, 1.0 - rip_ratio * 10.0)

    # Current sourcing: 1.0 if meaningful output current
    iout = abs(m.get("iout_avg", 0))
    if iout > 1e-6:
        reward += 1.0

    return reward


def _current_mirror_reward(
    outcome: SimulationOutcome,
) -> float:
    """Current mirror reward: output/reference current matching (max 6.0).

    Measures iref and iout; a perfect mirror has iout/iref = 1.
    """
    reward = 0.0
    m = outcome.metrics

    iref = abs(m.get("iref", 0))
    iout = abs(m.get("iout", 0))

    if iref < 1e-9:
        return reward  # no reference current → can't evaluate

    # Current ratio accuracy: 3.0 × max(0, 1 - |1 - iout/iref| × 5)
    ratio = iout / iref
    ratio_error = abs(1.0 - ratio)
    reward += 3.0 * max(0.0, 1.0 - ratio_error * 5.0)

    # Both currents in reasonable range (1µA - 100mA): 2.0
    if 1e-6 < iref < 0.1 and 1e-6 < iout < 0.1:
        reward += 2.0

    # Tight matching (< 5% error): bonus 1.0
    if ratio_error < 0.05:
        reward += 1.0

    return reward


# ---------------------------------------------------------------------------
# Topology-aware test specs for evaluation
# ---------------------------------------------------------------------------

# Tier 1 test specs (power converters)
TIER1_TEST_SPECS = [
    ("buck", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
    ("boost", {"vin": 5.0, "vout": 12.0, "iout": 0.5, "fsw": 100000}),
    ("buck_boost", {"vin": 12.0, "vout": 9.0, "iout": 1.0, "fsw": 100000}),
    ("cuk", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
    ("sepic", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
    ("flyback", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
    ("forward", {"vin": 48.0, "vout": 12.0, "iout": 1.0, "fsw": 100000}),
]

# Tier 2 test specs (signal circuits)
TIER2_TEST_SPECS = [
    ("inverting_amp", {"vin": 0.1, "cutoff_freq": 1000}),
    ("noninverting_amp", {"vin": 0.1, "cutoff_freq": 1000}),
    ("instrumentation_amp", {"vin": 0.01, "cutoff_freq": 1000}),
    ("differential_amp", {"vin": 0.1, "cutoff_freq": 1000}),
    ("sallen_key_lowpass", {"cutoff_freq": 1000}),
    ("sallen_key_highpass", {"cutoff_freq": 1000}),
    ("sallen_key_bandpass", {"cutoff_freq": 1000}),
    ("wien_bridge", {}),
    ("colpitts", {"vin": 12.0}),
    ("common_emitter", {"vin": 0.1, "cutoff_freq": 1000}),
    ("common_collector", {"vin": 0.1, "cutoff_freq": 1000}),
    ("common_base", {"vin": 0.1, "cutoff_freq": 1000}),
    ("cascode", {"vin": 0.1, "cutoff_freq": 1000}),
    ("current_mirror", {"vin": 12.0}),
    ("twin_t_notch", {"cutoff_freq": 1000}),
    ("state_variable_filter", {"cutoff_freq": 1000}),
    ("hartley", {"vin": 12.0}),
    ("phase_shift", {}),
    ("shunt_regulator", {"vin": 12.0}),
    ("series_regulator", {"vin": 12.0}),
    ("inverting_summing_amp", {"vin": 0.1, "cutoff_freq": 1000}),
    ("transimpedance_amp", {"cutoff_freq": 1000}),
    ("half_bridge", {"vin": 48.0, "vout": 24.0, "iout": 2.0, "fsw": 100000}),
    ("push_pull", {"vin": 48.0, "vout": 12.0, "iout": 2.0, "fsw": 100000}),
    ("charge_pump", {"vin": 5.0, "vout": 10.0, "iout": 0.1, "fsw": 100000}),
    ("voltage_doubler", {"vin": 12.0, "vout": 24.0, "iout": 0.1, "fsw": 50000}),
    ("zeta_converter", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
]

ALL_TEST_SPECS = TIER1_TEST_SPECS + TIER2_TEST_SPECS
