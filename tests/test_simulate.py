"""Tests for circuit simulation utilities (no SPICE needed)."""

import pytest
from arcs.simulate import components_to_params, normalize_topology, compute_reward
from arcs.evaluate import DecodedCircuit


class TestNormalizeTopology:
    """Topology name normalization."""

    @pytest.mark.parametrize("input_name,expected", [
        ("buck", "buck"),
        ("buck_boost", "buck_boost"),
        ("inverting_amp", "inverting_amp"),
        ("sallen_key_lowpass", "sallen_key_lowpass"),
    ])
    def test_normalize(self, input_name, expected):
        assert normalize_topology(input_name) == expected


class TestComponentsToParams:
    """Component list → parameter dict conversion."""

    def test_buck(self):
        comps = [
            ("INDUCTOR", 100e-6),
            ("CAPACITOR", 47e-6),
            ("RESISTOR", 0.01),
            ("MOSFET_N", 0.05),
        ]
        params = components_to_params("buck", comps)
        assert params is not None
        assert "inductance" in params
        assert "capacitance" in params
        assert abs(params["inductance"] - 100e-6) < 1e-10

    def test_boost(self):
        comps = [
            ("INDUCTOR", 47e-6),
            ("CAPACITOR", 220e-6),
            ("RESISTOR", 0.02),
            ("MOSFET_N", 0.05),
        ]
        params = components_to_params("boost", comps)
        assert params is not None
        assert "inductance" in params

    def test_cuk_six_components(self):
        comps = [
            ("INDUCTOR", 50e-6),
            ("INDUCTOR", 30e-6),
            ("CAPACITOR", 1e-6),
            ("CAPACITOR", 47e-6),
            ("RESISTOR", 0.01),
            ("MOSFET_N", 0.03),
        ]
        params = components_to_params("cuk", comps)
        assert params is not None
        assert "inductance_1" in params
        assert "inductance_2" in params
        assert "cap_coupling" in params

    def test_flyback_with_transformer(self):
        comps = [
            ("INDUCTOR", 500e-6),
            ("TRANSFORMER", 2.0),
            ("CAPACITOR", 100e-6),
            ("RESISTOR", 0.02),
            ("MOSFET_N", 0.04),
        ]
        params = components_to_params("flyback", comps)
        assert params is not None
        assert "turns_ratio" in params
        assert abs(params["turns_ratio"] - 2.0) < 1e-10

    def test_inverting_amp(self):
        comps = [("RESISTOR", 10000), ("RESISTOR", 1000)]
        params = components_to_params("inverting_amp", comps)
        assert params is not None
        assert "r_feedback" in params or "rf" in params or len(params) >= 2

    def test_empty_components(self):
        params = components_to_params("buck", [])
        assert params is None

    def test_wrong_component_count(self):
        comps = [("INDUCTOR", 100e-6)]  # Too few for buck
        params = components_to_params("buck", comps)
        assert params is None


class TestComputeReward:
    """Reward function (without simulation)."""

    def test_invalid_structure_zero(self):
        dec = DecodedCircuit(
            topology=None, specs={}, components=[],
            raw_tokens=[], valid_structure=False, error="bad"
        )
        from arcs.simulate import SimulationOutcome
        outcome = SimulationOutcome(success=False)
        reward = compute_reward(dec, outcome)
        assert reward == 0.0

    def test_valid_structure_min_reward(self):
        """Valid structure but failed sim → at least 1.0 reward."""
        dec = DecodedCircuit(
            topology="buck",
            specs={"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000},
            components=[("INDUCTOR", 100e-6), ("CAPACITOR", 47e-6),
                        ("RESISTOR", 0.01), ("MOSFET_N", 0.05)],
            raw_tokens=[],
            valid_structure=True
        )
        from arcs.simulate import SimulationOutcome
        outcome = SimulationOutcome(success=False, error="sim failed")
        reward = compute_reward(dec, outcome)
        assert reward >= 1.0
