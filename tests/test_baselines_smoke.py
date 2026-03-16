"""Smoke tests for arcs.baselines — instantiation and single-trial checks."""
from __future__ import annotations

import pytest
from arcs.baselines import (
    TrialResult,
    _specs_to_conditions,
    _encode_log,
    _decode_log,
    BaselineResults,
)
from arcs.simulate import SimulationOutcome
from arcs.templates import get_topology
import numpy as np


class TestTrialResult:
    def test_dataclass_creation(self):
        outcome = SimulationOutcome(success=True, metrics={}, valid=True)
        tr = TrialResult(
            topology="buck",
            specs={"vin": 12.0},
            params={"inductance": 1e-4},
            outcome=outcome,
            reward=5.0,
        )
        assert tr.topology == "buck"
        assert tr.reward == 5.0
        assert tr.outcome.success


class TestSpecConversion:
    def test_power_topology(self):
        topo = get_topology("buck")
        conds = _specs_to_conditions(
            "buck",
            {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100e3},
            topo,
        )
        assert isinstance(conds, dict)
        assert len(conds) > 0

    def test_signal_topology(self):
        topo = get_topology("inverting_amp")
        conds = _specs_to_conditions(
            "inverting_amp",
            {"gain": -10.0, "bandwidth": 1e6},
            topo,
        )
        assert isinstance(conds, dict)


class TestGeneEncoding:
    def test_encode_decode_roundtrip(self):
        topo = get_topology("buck")
        bounds = topo.component_bounds
        # Create params using bound names (not units)
        params = {}
        for b in bounds:
            mid = np.sqrt(b.min_val * b.max_val)
            params[b.name] = mid
        genes = _encode_log(params, bounds)
        decoded = _decode_log(genes, bounds)
        for b in bounds:
            assert b.name in decoded
            assert b.min_val <= decoded[b.name] <= b.max_val


class TestBaselineResults:
    def test_empty_results(self):
        br = BaselineResults(
            method="random_search",
            n_specs=0,
            trials_per_spec=0,
        )
        assert br.method == "random_search"
        assert br.sim_success_rate == 0.0
