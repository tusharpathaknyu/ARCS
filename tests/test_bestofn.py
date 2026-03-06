"""Tests for Best-of-N inference-time compute scaling."""

from __future__ import annotations

import torch
import pytest

from arcs.bestofn import (
    BestOfNGenerator,
    BestOfNResult,
    ScoredCandidate,
    rank_candidates,
    compute_diversity,
    run_scaling_experiment,
)
from arcs.constrained import ConstraintLevel
from arcs.model import ARCSConfig, ARCSModel
from arcs.tokenizer import CircuitTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return CircuitTokenizer()


@pytest.fixture
def small_model(tokenizer):
    config = ARCSConfig.small()
    config.vocab_size = tokenizer.vocab_size
    model = ARCSModel(config)
    model.eval()
    return model


@pytest.fixture
def generator(small_model, tokenizer):
    return BestOfNGenerator(
        small_model, tokenizer,
        constraint_level=ConstraintLevel.TOPOLOGY,
        ranking_method="confidence",
    )


@pytest.fixture
def generator_unconstrained(small_model, tokenizer):
    return BestOfNGenerator(
        small_model, tokenizer,
        constraint_level=ConstraintLevel.NONE,
        ranking_method="confidence",
    )


# ---------------------------------------------------------------------------
# Basic generation tests
# ---------------------------------------------------------------------------


class TestBestOfNGeneration:
    """Test core Best-of-N generation."""

    def test_generate_n1(self, generator):
        """N=1 produces a single candidate."""
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=1)
        assert isinstance(result, BestOfNResult)
        assert result.n_candidates == 1
        assert len(result.candidates) == 1
        assert result.best is result.candidates[0]

    def test_generate_n5(self, generator):
        """N=5 produces 5 candidates."""
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=5)
        assert len(result.candidates) == 5
        assert result.best.rank == 0

    def test_generate_n10(self, generator):
        """N=10 produces 10 ranked candidates."""
        result = generator.generate_n(
            "boost", {"vin": 5, "vout": 12, "iout": 0.5}, n=10
        )
        assert len(result.candidates) == 10
        # Ranks should be 0..9
        ranks = [c.rank for c in result.candidates]
        assert sorted(ranks) == list(range(10))

    def test_best_has_highest_confidence(self, generator):
        """Best candidate has highest mean_log_prob."""
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=10)
        best_conf = result.best.mean_log_prob
        for c in result.candidates[1:]:
            assert c.mean_log_prob <= best_conf + 1e-6

    def test_all_candidates_valid_structure(self, generator):
        """With TOPOLOGY constraint, all candidates should be valid."""
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=10)
        for c in result.candidates:
            assert c.decoded.valid_structure, f"Rank {c.rank} invalid"

    def test_candidate_has_scores(self, generator):
        """Each candidate has confidence scores."""
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=3)
        for c in result.candidates:
            assert isinstance(c.mean_log_prob, float)
            assert isinstance(c.total_log_prob, float)
            assert isinstance(c.mean_entropy, float)
            assert c.gen_time_ms > 0

    def test_total_time_positive(self, generator):
        """Total time is positive and reasonable."""
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=5)
        assert result.total_time_ms > 0
        # Should be at least N * ~5ms (even a fast model)
        assert result.total_time_ms >= 1.0

    def test_unconstrained_generation(self, generator_unconstrained):
        """Unconstrained mode also works."""
        result = generator_unconstrained.generate_n(
            "buck", {"vin": 12, "vout": 5}, n=3
        )
        assert len(result.candidates) == 3


# ---------------------------------------------------------------------------
# Multi-topology tests
# ---------------------------------------------------------------------------


class TestMultiTopology:
    """Test across different topologies."""

    @pytest.mark.parametrize(
        "topo,specs",
        [
            ("buck", {"vin": 12, "vout": 5, "iout": 1.0}),
            ("boost", {"vin": 5, "vout": 12, "iout": 0.5}),
            ("inverting_amp", {"vin": 0.1, "gain": 10}),
            ("sallen_key_lowpass", {"fc": 1000, "q": 0.707}),
        ],
    )
    def test_topology_generates(self, generator, topo, specs):
        """Each topology produces valid candidates."""
        result = generator.generate_n(topo, specs, n=3)
        assert len(result.candidates) == 3
        assert result.topology == topo
        assert result.best.decoded.valid_structure


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------


class TestRanking:
    """Test ranking methods."""

    def _make_candidates(self) -> list[ScoredCandidate]:
        """Create mock candidates with known scores."""
        from arcs.evaluate import DecodedCircuit

        candidates = []
        for i, (mlp, ent, valid) in enumerate([
            (-1.5, 2.0, True),
            (-0.5, 1.0, True),
            (-2.0, 3.0, False),
            (-1.0, 1.5, True),
        ]):
            candidates.append(ScoredCandidate(
                tokens=[1, 2, 3],
                decoded=DecodedCircuit(
                    topology="buck", specs={}, components=[("INDUCTOR", 1e-5)],
                    raw_tokens=[1, 2, 3], valid_structure=valid,
                ),
                log_probs=torch.tensor([mlp]),
                entropies=torch.tensor([ent]),
                mean_log_prob=mlp,
                total_log_prob=mlp,
                mean_entropy=ent,
                gen_time_ms=10.0,
            ))
        return candidates

    def test_rank_by_confidence(self):
        """Confidence ranking orders by mean_log_prob descending."""
        cands = self._make_candidates()
        ranked = rank_candidates(cands, method="confidence")
        assert ranked[0].mean_log_prob == -0.5  # highest

    def test_rank_by_entropy(self):
        """Entropy ranking orders by mean_entropy ascending."""
        cands = self._make_candidates()
        ranked = rank_candidates(cands, method="entropy")
        assert ranked[0].mean_entropy == 1.0  # lowest

    def test_rank_by_valid_first(self):
        """Valid-first ranking puts valid candidates first."""
        cands = self._make_candidates()
        ranked = rank_candidates(cands, method="valid_first")
        # Invalid candidate (index 2) should be last
        assert ranked[-1].decoded.valid_structure is False

    def test_rank_assigns_indices(self):
        """Ranking assigns rank 0, 1, 2, 3."""
        cands = self._make_candidates()
        ranked = rank_candidates(cands, method="confidence")
        for i, c in enumerate(ranked):
            assert c.rank == i


# ---------------------------------------------------------------------------
# Diversity tests
# ---------------------------------------------------------------------------


class TestDiversity:
    """Test diversity computation."""

    def test_single_candidate(self):
        """Single candidate has diversity 1.0."""
        from arcs.evaluate import DecodedCircuit

        cands = [ScoredCandidate(
            tokens=[1], decoded=DecodedCircuit(
                topology="buck", specs={},
                components=[("INDUCTOR", 1e-5)],
                raw_tokens=[1], valid_structure=True,
            ),
            log_probs=torch.tensor([-1.0]),
            entropies=torch.tensor([1.0]),
            mean_log_prob=-1.0, total_log_prob=-1.0,
            mean_entropy=1.0, gen_time_ms=10.0,
        )]
        assert compute_diversity(cands) == 1.0

    def test_identical_candidates(self):
        """Identical candidates have diversity 0.0."""
        from arcs.evaluate import DecodedCircuit

        base = ScoredCandidate(
            tokens=[1], decoded=DecodedCircuit(
                topology="buck", specs={},
                components=[("INDUCTOR", 1e-5)],
                raw_tokens=[1], valid_structure=True,
            ),
            log_probs=torch.tensor([-1.0]),
            entropies=torch.tensor([1.0]),
            mean_log_prob=-1.0, total_log_prob=-1.0,
            mean_entropy=1.0, gen_time_ms=10.0,
        )
        cands = [base, base, base]
        assert compute_diversity(cands) == 0.0

    def test_diverse_candidates(self):
        """Different candidates have diversity > 0."""
        from arcs.evaluate import DecodedCircuit

        cands = []
        for val in [1e-5, 2e-5, 3e-5]:
            cands.append(ScoredCandidate(
                tokens=[1], decoded=DecodedCircuit(
                    topology="buck", specs={},
                    components=[("INDUCTOR", val)],
                    raw_tokens=[1], valid_structure=True,
                ),
                log_probs=torch.tensor([-1.0]),
                entropies=torch.tensor([1.0]),
                mean_log_prob=-1.0, total_log_prob=-1.0,
                mean_entropy=1.0, gen_time_ms=10.0,
            ))
        assert compute_diversity(cands) == 1.0  # All different components


# ---------------------------------------------------------------------------
# BestOfNResult properties tests
# ---------------------------------------------------------------------------


class TestBestOfNResult:
    """Test result dataclass properties."""

    def test_best_confidence(self, generator):
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=5)
        assert result.best_confidence == result.best.mean_log_prob

    def test_avg_confidence(self, generator):
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=5)
        expected = sum(c.mean_log_prob for c in result.candidates) / 5
        assert abs(result.avg_confidence - expected) < 1e-6

    def test_confidence_spread(self, generator):
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=10)
        # Spread should be non-negative
        assert result.confidence_spread >= 0

    def test_diversity_in_range(self, generator):
        result = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=10)
        assert 0.0 <= result.diversity <= 1.0


# ---------------------------------------------------------------------------
# Scaling experiment tests
# ---------------------------------------------------------------------------


class TestScalingExperiment:
    """Test the scaling experiment runner."""

    def test_scaling_runs(self, small_model, tokenizer):
        """Scaling experiment completes with small params."""
        results = run_scaling_experiment(
            small_model, tokenizer,
            device=torch.device("cpu"),
            n_values=[1, 3],
            n_specs=4,
            constraint_level=ConstraintLevel.TOPOLOGY,
            verbose=False,
        )
        assert 1 in results
        assert 3 in results
        assert "validity_rate" in results[1]
        assert "avg_confidence" in results[1]

    def test_scaling_monotonic_time(self, small_model, tokenizer):
        """Higher N should take more time."""
        results = run_scaling_experiment(
            small_model, tokenizer,
            device=torch.device("cpu"),
            n_values=[1, 5],
            n_specs=4,
            constraint_level=ConstraintLevel.TOPOLOGY,
            verbose=False,
        )
        # N=5 should take ~5x the time of N=1 (approximately)
        assert results[5]["avg_time_ms"] > results[1]["avg_time_ms"]

    def test_scaling_has_all_metrics(self, small_model, tokenizer):
        """All expected metric keys are present."""
        results = run_scaling_experiment(
            small_model, tokenizer,
            device=torch.device("cpu"),
            n_values=[1],
            n_specs=2,
            constraint_level=ConstraintLevel.TOPOLOGY,
            verbose=False,
        )
        expected_keys = [
            "validity_rate", "avg_confidence", "avg_entropy",
            "avg_diversity", "avg_time_ms", "total_candidates",
        ]
        for k in expected_keys:
            assert k in results[1], f"Missing key: {k}"


# ---------------------------------------------------------------------------
# Statistical property tests
# ---------------------------------------------------------------------------


class TestStatisticalProperties:
    """Test that Best-of-N improves confidence with N."""

    def test_best_of_n_improves_confidence(self, generator):
        """Best-of-10 should have >= confidence of Best-of-1 on average."""
        torch.manual_seed(42)
        n_trials = 20
        conf_1 = []
        conf_10 = []

        for i in range(n_trials):
            r1 = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=1)
            conf_1.append(r1.best.mean_log_prob)
            r10 = generator.generate_n("buck", {"vin": 12, "vout": 5}, n=10)
            conf_10.append(r10.best.mean_log_prob)

        avg_1 = sum(conf_1) / len(conf_1)
        avg_10 = sum(conf_10) / len(conf_10)
        # Best-of-10 should be at least as confident on average
        assert avg_10 >= avg_1 - 0.5, (
            f"Expected Best-of-10 ({avg_10:.3f}) >= Best-of-1 ({avg_1:.3f})"
        )

    def test_all_topologies_100pct_valid(self, generator):
        """With TOPOLOGY constraints, all Best-of-N candidates are valid."""
        from arcs.simulate import COMPONENT_TO_PARAM

        topologies = list(COMPONENT_TO_PARAM.keys())
        for topo in topologies[:8]:  # Test subset for speed
            result = generator.generate_n(topo, {"vin": 12, "vout": 5}, n=5)
            for c in result.candidates:
                assert c.decoded.valid_structure, (
                    f"Topology {topo}, rank {c.rank} invalid"
                )
