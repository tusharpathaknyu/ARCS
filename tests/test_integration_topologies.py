"""Integration tests covering all 34 topologies end-to-end.

Tests netlist generation, parameter sampling, derived metric computation,
reward computation, and validity checking for every registered topology.
"""

import pytest
import numpy as np

from arcs.templates import (
    get_topology, get_all_topologies,
    _TIER1_NAMES, _TIER2_NAMES,
)
from arcs.datagen import compute_derived_metrics, is_valid_result
from arcs.simulate import compute_reward, normalize_topology


ALL_TOPO_NAMES = _TIER1_NAMES + _TIER2_NAMES


class TestAllTopologiesRegistered:
    """Verify all topologies are properly registered in templates."""

    def test_total_topology_count(self):
        all_topos = get_all_topologies()
        assert len(all_topos) == 34, f"Expected 34 topologies, got {len(all_topos)}"

    def test_tier1_count(self):
        assert len(_TIER1_NAMES) == 7

    def test_tier2_count(self):
        assert len(_TIER2_NAMES) == 27

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_get_topology(self, name):
        topo = get_topology(name)
        assert topo.name == name
        assert len(topo.component_bounds) > 0
        assert len(topo.metric_names) > 0
        assert topo.operating_conditions is not None


class TestNetlistGeneration:
    """Verify every topology can generate a valid SPICE netlist."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_generate_netlist(self, name):
        topo = get_topology(name)
        rng = np.random.default_rng(42)
        params = topo.sample_parameters(rng)

        netlist = topo.generate_netlist(params)
        assert isinstance(netlist, str)
        assert len(netlist) > 50, f"Netlist too short for {name}"
        # All netlists should have .end
        assert ".end" in netlist.lower(), f"Missing .end in {name} netlist"


class TestParameterSampling:
    """Verify parameter sampling produces valid ranges."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_sample_parameters(self, name):
        topo = get_topology(name)
        rng = np.random.default_rng(42)

        for _ in range(5):
            params = topo.sample_parameters(rng)
            assert isinstance(params, dict)
            assert len(params) == len(topo.component_bounds)
            for bound in topo.component_bounds:
                assert bound.name in params, f"Missing param {bound.name} for {name}"
                val = params[bound.name]
                assert bound.min_val <= val <= bound.max_val, (
                    f"{name}.{bound.name}: {val} not in [{bound.min_val}, {bound.max_val}]"
                )


class TestESeriesSnapping:
    """Verify E-series snapping works for all topologies."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_snap_to_e24(self, name):
        topo = get_topology(name)
        rng = np.random.default_rng(42)
        params = topo.sample_parameters(rng)

        for bound in topo.component_bounds:
            if bound.unit in ("H", "F", "Ω"):
                snapped = bound.snap_to_e_series(params[bound.name], series=24)
                assert bound.min_val * 0.5 <= snapped <= bound.max_val * 2.0


class TestDerivedMetrics:
    """Verify derived metric computation doesn't crash for any topology."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_compute_derived_metrics_no_crash(self, name):
        topo = get_topology(name)
        # Simulate with dummy raw metrics
        raw = {m: 1.0 for m in topo.metric_names}
        derived = compute_derived_metrics(raw, topo.operating_conditions, name)
        assert isinstance(derived, dict)
        assert len(derived) >= len(raw)


class TestRewardComputation:
    """Verify reward computation works for all topologies."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_reward_no_crash(self, name):
        from arcs.simulate import SimulationOutcome
        from arcs.evaluate import DecodedCircuit
        topo = get_topology(name)
        # Create minimal decoded circuit + simulation outcome
        decoded = DecodedCircuit(
            topology=name,
            specs=topo.operating_conditions,
            components=[(b.unit, b.min_val) for b in topo.component_bounds],
            raw_tokens=[],
            valid_structure=True,
        )
        outcome = SimulationOutcome(
            success=True,
            metrics={m: 1.0 for m in topo.metric_names},
            valid=True,
        )
        reward = compute_reward(decoded, outcome, target_specs=topo.operating_conditions)
        assert isinstance(reward, (int, float))
        assert reward >= 0


class TestValidityChecks:
    """Verify validity checking covers all topologies."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_is_valid_result_no_crash(self, name):
        topo = get_topology(name)
        metrics = {m: 1.0 for m in topo.metric_names}
        derived = compute_derived_metrics(metrics, topo.operating_conditions, name)
        result = is_valid_result(derived, topo.operating_conditions, name)
        assert isinstance(result, bool)


class TestTopologyNormalization:
    """Verify all topology names normalize to themselves."""

    @pytest.mark.parametrize("name", ALL_TOPO_NAMES)
    def test_canonical_names_unchanged(self, name):
        assert normalize_topology(name) == name


class TestVCGTopologyCoverage:
    """Verify VCG knows about all topologies."""

    def test_all_topologies_in_vcg(self):
        from arcs.valid_circuit_gen import ALL_TOPOLOGIES, TOPOLOGY_TO_IDX
        for name in ALL_TOPO_NAMES:
            assert name in ALL_TOPOLOGIES, f"{name} missing from VCG ALL_TOPOLOGIES"
            assert name in TOPOLOGY_TO_IDX, f"{name} missing from VCG TOPOLOGY_TO_IDX"

    def test_all_topologies_have_adjacency(self):
        from arcs.valid_circuit_gen import TOPOLOGY_ADJACENCY
        for name in ALL_TOPO_NAMES:
            assert name in TOPOLOGY_ADJACENCY, f"{name} missing from TOPOLOGY_ADJACENCY"

    def test_expected_components_computed(self):
        from arcs.valid_circuit_gen import TOPOLOGY_EXPECTED_COMPONENTS
        for name in ALL_TOPO_NAMES:
            assert name in TOPOLOGY_EXPECTED_COMPONENTS, (
                f"{name} missing from TOPOLOGY_EXPECTED_COMPONENTS"
            )
            assert TOPOLOGY_EXPECTED_COMPONENTS[name] >= 1
