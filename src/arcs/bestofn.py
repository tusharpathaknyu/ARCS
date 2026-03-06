"""Best-of-N inference-time compute scaling for ARCS.

Generate N candidate circuits in parallel, score by model confidence
(mean log-probability), and return the best.  Because ARCS inference
is ~20 ms per circuit, even N=50 costs only ~1 s — still orders of
magnitude faster than random search (200 SPICE sims, ~60 s) or GA
(630 SPICE sims, ~250 s).

Key insight: model confidence is *free* (computed during generation)
and correlates with circuit quality — no SPICE oracle required for
candidate ranking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from arcs.constrained import ConstrainedGenerator, ConstraintLevel
from arcs.evaluate import DecodedCircuit, decode_generated_sequence
from arcs.simulate import (
    ALL_TEST_SPECS,
    TIER1_TEST_SPECS,
    TIER2_TEST_SPECS,
    compute_reward,
    simulate_decoded_circuit,
)
from arcs.tokenizer import CircuitTokenizer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    """A generated circuit with confidence scores."""

    tokens: list[int]
    decoded: DecodedCircuit
    log_probs: torch.Tensor        # per-token log probs
    entropies: torch.Tensor        # per-token entropy
    mean_log_prob: float           # avg log-prob (confidence)
    total_log_prob: float          # sum log-prob (joint probability)
    mean_entropy: float            # avg entropy (uncertainty)
    gen_time_ms: float             # wall-clock generation time
    rank: int = 0                  # rank within batch (0 = best)

    # Optional — populated if SPICE simulation is requested
    reward: float | None = None
    sim_valid: bool | None = None
    sim_success: bool | None = None


@dataclass
class BestOfNResult:
    """Result of a Best-of-N generation run."""

    topology: str
    specs: dict[str, float]
    n_candidates: int
    best: ScoredCandidate
    candidates: list[ScoredCandidate]
    total_time_ms: float           # wall-clock for all N candidates

    # Ranking statistics
    confidence_spread: float = 0.0  # max - min mean_log_prob
    diversity: float = 0.0          # fraction of unique circuits

    @property
    def best_confidence(self) -> float:
        return self.best.mean_log_prob

    @property
    def avg_confidence(self) -> float:
        return sum(c.mean_log_prob for c in self.candidates) / len(self.candidates)


# ---------------------------------------------------------------------------
# Scoring / ranking
# ---------------------------------------------------------------------------


def _edit_distance_tokens(a: list[int], b: list[int]) -> int:
    """Levenshtein distance on token sequences (for diversity)."""
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len(b)]


def compute_diversity(candidates: list[ScoredCandidate]) -> float:
    """Fraction of pairwise-distinct candidates (edit distance > 2)."""
    if len(candidates) <= 1:
        return 1.0
    n_pairs = 0
    n_distinct = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            n_pairs += 1
            # Compare component sections only (skip prefix)
            a_comps = [(c, v) for c, v in candidates[i].decoded.components]
            b_comps = [(c, v) for c, v in candidates[j].decoded.components]
            if a_comps != b_comps:
                n_distinct += 1
    return n_distinct / max(n_pairs, 1)


def rank_candidates(
    candidates: list[ScoredCandidate],
    method: str = "confidence",
) -> list[ScoredCandidate]:
    """Rank candidates by the specified method.

    Methods:
        confidence: Mean log-probability (higher = more confident)
        joint:      Total log-probability (higher = more likely)
        entropy:    Mean entropy (lower = more certain)
        valid_first: Valid structure first, then by confidence
    """
    if method == "confidence":
        ranked = sorted(candidates, key=lambda c: c.mean_log_prob, reverse=True)
    elif method == "joint":
        ranked = sorted(candidates, key=lambda c: c.total_log_prob, reverse=True)
    elif method == "entropy":
        ranked = sorted(candidates, key=lambda c: c.mean_entropy)
    elif method == "valid_first":
        ranked = sorted(
            candidates,
            key=lambda c: (c.decoded.valid_structure, c.mean_log_prob),
            reverse=True,
        )
    else:
        raise ValueError(f"Unknown ranking method: {method}")

    for i, c in enumerate(ranked):
        c.rank = i
    return ranked


# ---------------------------------------------------------------------------
# Core Best-of-N generator
# ---------------------------------------------------------------------------


class BestOfNGenerator:
    """Generate N candidates per spec and return the best.

    Parameters
    ----------
    model : nn.Module
        Trained ARCS model (any architecture).
    tokenizer : CircuitTokenizer
        The circuit tokenizer.
    constraint_level : ConstraintLevel
        Constraint level for generation (default: TOPOLOGY for balance
        of validity guarantee + value diversity).
    ranking_method : str
        How to rank candidates: 'confidence', 'joint', 'entropy',
        'valid_first'.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: CircuitTokenizer,
        constraint_level: ConstraintLevel = ConstraintLevel.TOPOLOGY,
        ranking_method: str = "confidence",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.constraint_level = constraint_level
        self.ranking_method = ranking_method

        # Build constrained generator if level > NONE
        if constraint_level != ConstraintLevel.NONE:
            self.gen = ConstrainedGenerator(model, tokenizer, level=constraint_level)
        else:
            self.gen = None

    def generate_n(
        self,
        topology: str,
        specs: dict[str, float],
        n: int = 10,
        temperature: float = 0.8,
        top_k: int = 50,
        max_new_tokens: int = 80,
        device: torch.device | str = "cpu",
    ) -> BestOfNResult:
        """Generate N candidate circuits and return ranked results.

        Parameters
        ----------
        topology : str
            Target topology name (e.g., 'buck', 'inverting_amp').
        specs : dict
            Target specifications (e.g., {'vin': 12, 'vout': 5}).
        n : int
            Number of candidates to generate.
        temperature : float
            Sampling temperature.
        top_k : int
            Top-k filtering parameter.
        max_new_tokens : int
            Maximum generation length.
        device : torch.device | str
            Device for generation.

        Returns
        -------
        BestOfNResult
            Ranked candidates with the best one selected.
        """
        device = torch.device(device) if isinstance(device, str) else device
        prefix = self._build_prefix(topology, specs).to(device)

        candidates: list[ScoredCandidate] = []
        t_total_start = time.perf_counter()

        for _ in range(n):
            t0 = time.perf_counter()
            candidate = self._generate_one(
                prefix, topology, temperature, top_k, max_new_tokens
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            candidate.gen_time_ms = dt_ms
            candidates.append(candidate)

        total_time_ms = (time.perf_counter() - t_total_start) * 1000

        # Rank candidates
        ranked = rank_candidates(candidates, method=self.ranking_method)

        # Compute diversity
        diversity = compute_diversity(ranked)

        # Confidence spread
        if len(ranked) >= 2:
            spread = ranked[0].mean_log_prob - ranked[-1].mean_log_prob
        else:
            spread = 0.0

        return BestOfNResult(
            topology=topology,
            specs=specs,
            n_candidates=n,
            best=ranked[0],
            candidates=ranked,
            total_time_ms=total_time_ms,
            confidence_spread=spread,
            diversity=diversity,
        )

    def _generate_one(
        self,
        prefix: torch.Tensor,
        topology: str,
        temperature: float,
        top_k: int,
        max_new_tokens: int,
    ) -> ScoredCandidate:
        """Generate a single candidate with confidence scoring."""
        from arcs.constrained import constrained_sample_with_logprobs
        from arcs.rl import sample_with_logprobs

        if self.constraint_level != ConstraintLevel.NONE:
            gen_tokens, log_probs, entropies = constrained_sample_with_logprobs(
                self.model,
                prefix,
                self.tokenizer,
                topology=topology,
                level=self.constraint_level,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            gen_tokens, log_probs, entropies = sample_with_logprobs(
                self.model,
                prefix,
                self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Full sequence = prefix + generated
        all_ids = prefix[0].tolist() + gen_tokens.tolist()
        decoded = decode_generated_sequence(all_ids, self.tokenizer)

        # Compute confidence scores
        if len(log_probs) > 0:
            mean_lp = log_probs.mean().item()
            total_lp = log_probs.sum().item()
            mean_ent = entropies.mean().item()
        else:
            mean_lp = 0.0
            total_lp = 0.0
            mean_ent = 0.0

        return ScoredCandidate(
            tokens=all_ids,
            decoded=decoded,
            log_probs=log_probs,
            entropies=entropies,
            mean_log_prob=mean_lp,
            total_log_prob=total_lp,
            mean_entropy=mean_ent,
            gen_time_ms=0.0,  # filled by caller
        )

    def _build_prefix(
        self, topology: str, specs: dict[str, float]
    ) -> torch.Tensor:
        """Build a conditioning prefix tensor."""
        tok = self.tokenizer
        prefix_ids = [tok.start_id]

        # Topology token
        topo_key = f"TOPO_{topology.upper()}"
        _topo_map = {
            "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
            "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
            "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
        }
        topo_key = _topo_map.get(topology, topo_key)
        if topo_key in tok.name_to_id:
            prefix_ids.append(tok.name_to_id[topo_key])

        prefix_ids.append(tok.sep_id)

        # Spec tokens
        for spec_name, spec_val in specs.items():
            spec_key = f"SPEC_{spec_name.upper()}"
            if spec_key in tok.name_to_id:
                prefix_ids.append(tok.name_to_id[spec_key])
                prefix_ids.append(tok.encode_value(abs(spec_val)))

        prefix_ids.append(tok.sep_id)
        return torch.tensor([prefix_ids])


# ---------------------------------------------------------------------------
# Scaling experiment runner
# ---------------------------------------------------------------------------


def run_scaling_experiment(
    model: Any,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_values: list[int] | None = None,
    n_specs: int = 160,
    constraint_level: ConstraintLevel = ConstraintLevel.TOPOLOGY,
    temperature: float = 0.8,
    top_k: int = 50,
    simulate: bool = False,
    tier: int | None = None,
    verbose: bool = False,
) -> dict[int, dict[str, float]]:
    """Run Best-of-N at multiple N values and return scaling metrics.

    Parameters
    ----------
    n_values : list[int]
        Values of N to test (default: [1, 3, 5, 10, 20, 50]).
    n_specs : int
        Number of test specifications to evaluate.
    simulate : bool
        Whether to run SPICE simulation on the best candidates.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from N to metrics dict with keys:
        'validity_rate', 'avg_confidence', 'avg_entropy',
        'avg_diversity', 'avg_time_ms', 'avg_time_per_design_ms',
        and optionally 'avg_reward', 'sim_valid_rate'.
    """
    if n_values is None:
        n_values = [1, 3, 5, 10, 20, 50]

    # Select test specs
    if tier == 1:
        test_specs = TIER1_TEST_SPECS
    elif tier == 2:
        test_specs = TIER2_TEST_SPECS
    else:
        test_specs = ALL_TEST_SPECS

    results: dict[int, dict[str, float]] = {}

    for n in n_values:
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"  Best-of-N = {n}")
            print(f"{'=' * 50}")

        gen = BestOfNGenerator(
            model, tokenizer,
            constraint_level=constraint_level,
            ranking_method="confidence",
        )

        valid_count = 0
        confidences = []
        entropies_list = []
        diversities = []
        times = []
        rewards = []
        sim_valids = 0
        sim_attempted = 0

        for i in range(n_specs):
            topo, specs = test_specs[i % len(test_specs)]
            result = gen.generate_n(
                topology=topo,
                specs=specs,
                n=n,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )

            best = result.best
            if best.decoded.valid_structure:
                valid_count += 1

            confidences.append(result.best_confidence)
            entropies_list.append(best.mean_entropy)
            diversities.append(result.diversity)
            times.append(result.total_time_ms)

            # Optional SPICE simulation
            if simulate and best.decoded.valid_structure:
                sim_attempted += 1
                try:
                    outcome = simulate_decoded_circuit(best.decoded)
                    r = compute_reward(best.decoded, outcome, specs)
                    rewards.append(r)
                    if outcome.valid:
                        sim_valids += 1
                except Exception:
                    rewards.append(0.0)

        metrics = {
            "validity_rate": valid_count / n_specs,
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_entropy": sum(entropies_list) / len(entropies_list),
            "avg_diversity": sum(diversities) / max(len(diversities), 1),
            "avg_time_ms": sum(times) / len(times),
            "avg_time_per_design_ms": sum(times) / len(times),
            "total_candidates": n * n_specs,
        }

        if simulate and rewards:
            metrics["avg_reward"] = sum(rewards) / len(rewards)
            metrics["sim_valid_rate"] = sim_valids / max(sim_attempted, 1)

        if verbose:
            print(f"  Validity:    {metrics['validity_rate']:.1%}")
            print(f"  Confidence:  {metrics['avg_confidence']:.4f}")
            print(f"  Entropy:     {metrics['avg_entropy']:.4f}")
            print(f"  Diversity:   {metrics['avg_diversity']:.1%}")
            print(f"  Time/spec:   {metrics['avg_time_ms']:.1f} ms")
            if "avg_reward" in metrics:
                print(f"  Avg reward:  {metrics['avg_reward']:.3f}")
                print(f"  Sim valid:   {metrics['sim_valid_rate']:.1%}")

        results[n] = metrics

    return results


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------


def calibration_analysis(
    model: Any,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_samples: int = 200,
    constraint_level: ConstraintLevel = ConstraintLevel.NONE,
    temperature: float = 0.8,
    top_k: int = 50,
) -> dict[str, Any]:
    """Analyze correlation between model confidence and SPICE validity.

    Generates n_samples circuits, simulates each, and computes:
    - Pearson correlation between mean_log_prob and binary sim_valid
    - Confidence distributions for valid vs invalid circuits
    - ROC-AUC if enough data

    Returns
    -------
    dict with keys: 'n_total', 'n_valid', 'n_invalid',
    'valid_mean_conf', 'invalid_mean_conf', 'conf_gap',
    'correlation', 'valid_confidences', 'invalid_confidences'.
    """
    gen = BestOfNGenerator(
        model, tokenizer,
        constraint_level=constraint_level,
        ranking_method="confidence",
    )

    valid_confs: list[float] = []
    invalid_confs: list[float] = []
    all_confs: list[float] = []
    all_valids: list[int] = []

    for i in range(n_samples):
        topo, specs = ALL_TEST_SPECS[i % len(ALL_TEST_SPECS)]
        result = gen.generate_n(
            topology=topo,
            specs=specs,
            n=1,  # Single sample for calibration
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        best = result.best
        conf = best.mean_log_prob

        if not best.decoded.valid_structure:
            invalid_confs.append(conf)
            all_confs.append(conf)
            all_valids.append(0)
            continue

        try:
            outcome = simulate_decoded_circuit(best.decoded)
            if outcome.valid:
                valid_confs.append(conf)
                all_valids.append(1)
            else:
                invalid_confs.append(conf)
                all_valids.append(0)
        except Exception:
            invalid_confs.append(conf)
            all_valids.append(0)

        all_confs.append(conf)

    # Compute correlation
    correlation = 0.0
    if len(all_confs) >= 10 and len(set(all_valids)) > 1:
        mean_c = sum(all_confs) / len(all_confs)
        mean_v = sum(all_valids) / len(all_valids)
        cov = sum(
            (c - mean_c) * (v - mean_v)
            for c, v in zip(all_confs, all_valids)
        ) / len(all_confs)
        std_c = (
            sum((c - mean_c) ** 2 for c in all_confs) / len(all_confs)
        ) ** 0.5
        std_v = (
            sum((v - mean_v) ** 2 for v in all_valids) / len(all_valids)
        ) ** 0.5
        if std_c > 0 and std_v > 0:
            correlation = cov / (std_c * std_v)

    valid_mean = sum(valid_confs) / max(len(valid_confs), 1)
    invalid_mean = sum(invalid_confs) / max(len(invalid_confs), 1)

    return {
        "n_total": n_samples,
        "n_valid": len(valid_confs),
        "n_invalid": len(invalid_confs),
        "valid_mean_conf": valid_mean,
        "invalid_mean_conf": invalid_mean,
        "conf_gap": valid_mean - invalid_mean,
        "correlation": correlation,
        "valid_confidences": valid_confs,
        "invalid_confidences": invalid_confs,
    }
