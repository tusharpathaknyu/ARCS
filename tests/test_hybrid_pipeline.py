"""Tests for hybrid circuit generation pipeline."""

import torch
import pytest
from arcs.hybrid_pipeline import (
    HybridGenerator,
    GeneratedCircuit,
    vcg_graph_to_spice,
    _prepare_vcg_input,
    evaluate_generator,
    summarize_eval_results,
    EvalResult,
)
from arcs.valid_circuit_gen import (
    VCGConfig,
    ValidCircuitGenModel,
    CircuitGraph,
    check_circuit_validity,
    TOPOLOGY_TO_IDX,
)
from arcs.tokenizer import CircuitTokenizer
from arcs.simulate import ALL_TEST_SPECS


class TestPrepareVCGInput:
    def test_buck_specs(self):
        config = VCGConfig()
        batch = _prepare_vcg_input(
            "buck",
            {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 1e5},
            config,
            torch.device("cpu"),
        )
        assert batch["topology_idx"].shape == (1,)
        assert batch["active_mask"].shape == (1, config.max_nodes)
        assert batch["spec_mask"].sum().item() >= 2  # at least vin, vout
        assert batch["value_bounds_min"].shape == (1, config.max_nodes)

    def test_unknown_topology(self):
        config = VCGConfig()
        batch = _prepare_vcg_input(
            "nonexistent_topo",
            {"vin": 12.0},
            config,
            torch.device("cpu"),
        )
        # Should still produce valid tensors (with 0 active nodes)
        assert batch["topology_idx"].item() == 0
        assert batch["active_mask"].sum().item() == 0


class TestVCGGraphToSpice:
    def test_valid_graph(self):
        """A VCG graph can be converted to SPICE format."""
        from arcs.spice import NGSpiceRunner
        runner = NGSpiceRunner()
        tokenizer = CircuitTokenizer()

        # Create a simple buck-like graph
        graph = CircuitGraph(
            topology="buck",
            n_components=4,
            node_types=torch.tensor([3, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
            adjacency=torch.zeros(12, 12),
            values=torch.tensor([-4.0, -4.0, -1.0, -1.5, 0, 0, 0, 0, 0, 0, 0, 0]),
            active_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            spec_types=torch.zeros(8, dtype=torch.long),
            spec_values=torch.zeros(8),
            spec_mask=torch.zeros(8),
            value_bounds_min=torch.full((12,), -12.0),
            value_bounds_max=torch.full((12,), 6.0),
        )

        decoded, outcome, reward = vcg_graph_to_spice(graph, runner, tokenizer)
        # Should at least decode without error
        assert decoded is not None
        assert isinstance(reward, float)


class TestHybridGenerator:
    def test_empty_generator(self):
        """HybridGenerator with no models returns empty."""
        gen = HybridGenerator()
        result = gen.generate_best("buck", {"vin": 12, "vout": 5})
        assert result.source == "none"
        assert result.reward == 0.0

    def test_vcg_generator(self):
        """HybridGenerator with VCG model generates circuits."""
        config = VCGConfig(
            latent_dim=32, d_model=64, n_encoder_layers=2,
            n_heads=2, d_ff=128, n_decoder_layers=2, decoder_hidden=128,
        )
        vcg = ValidCircuitGenModel(config)

        gen = HybridGenerator(vcg_model=vcg)
        circuits = gen.generate_from_vcg(
            "buck",
            {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 1e5},
            n_candidates=2,
        )
        assert len(circuits) == 2
        assert all(c.source == "vcg" for c in circuits)
        assert all(c.topology == "buck" for c in circuits)

    def test_generate_best_with_prerank_limits_simulations(self, monkeypatch):
        """Pre-ranking should simulate only top-k shortlisted candidates."""
        gen = HybridGenerator()

        fake_candidates = [
            GeneratedCircuit(source="vcg", topology="buck", gen_time_ms=float(i))
            for i in range(5)
        ]

        monkeypatch.setattr(
            gen,
            "generate_from_vcg",
            lambda topology, specs, n_candidates, simulate=True: list(fake_candidates),
        )

        score_map = {id(c): c.gen_time_ms for c in fake_candidates}
        monkeypatch.setattr(
            gen,
            "_score_candidate_proxy",
            lambda c: score_map[id(c)],
        )

        simulated_ids = []

        def _fake_simulate(candidate, target_specs=None):
            simulated_ids.append(id(candidate))
            candidate.reward = candidate.gen_time_ms
            return candidate

        monkeypatch.setattr(gen, "_simulate_candidate", _fake_simulate)

        best = gen.generate_best(
            "buck",
            {"vin": 12.0, "vout": 5.0},
            n_candidates_per_source=5,
            sources=["vcg"],
            pre_rank_top_k=2,
        )

        assert len(simulated_ids) == 2
        # Highest proxy scores are gen_time_ms 4 and 3; best should be 4.
        assert best.reward == 4.0


class TestEvalResult:
    def test_summarize(self):
        results = {
            "buck": EvalResult(
                topology="buck", n_generated=10,
                n_struct_valid=8, n_sim_success=6, n_sim_valid=4,
                mean_reward=3.5,
            ),
            "boost": EvalResult(
                topology="boost", n_generated=10,
                n_struct_valid=9, n_sim_success=7, n_sim_valid=5,
                mean_reward=4.0,
            ),
        }
        summary = summarize_eval_results(results)
        assert summary["n_topologies"] == 2
        assert summary["n_generated"] == 20
        assert 0.8 <= summary["struct_valid_rate"] <= 0.9
        assert abs(summary["mean_reward"] - 3.75) < 0.01

    def test_evaluate_generator_accepts_list_specs(self):
        def generator_fn(topology, specs):
            return [GeneratedCircuit(source="mock", topology=topology, reward=1.0)]

        results = evaluate_generator(generator_fn, test_specs=ALL_TEST_SPECS, label="mock")
        assert isinstance(results, dict)
        assert "buck" in results
        assert results["buck"].n_generated >= 1


class TestGeneratedCircuit:
    def test_dataclass(self):
        gc = GeneratedCircuit(source="vcg", topology="buck", reward=5.0)
        assert gc.source == "vcg"
        assert gc.reward == 5.0
        assert gc.decoded is None
