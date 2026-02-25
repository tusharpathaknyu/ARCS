"""ARCS baselines: Random Search and Genetic Algorithm for circuit design.

Provides naive search-based methods that operate directly in parameter space
(no learned model). Used as paper comparison baselines for the ARCS model.

Both methods:
1. Sample/evolve component values within the same ComponentBounds as training
2. Build SPICE netlists via the template system
3. Simulate with ngspice and compute domain-aware reward
4. Report the same metrics as ARCS evaluation for fair comparison
"""

from __future__ import annotations

import json
import math
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from arcs.templates import get_topology, TopologyTemplate, ComponentBounds
from arcs.datagen import compute_derived_metrics, is_valid_result
from arcs.spice import NGSpiceRunner
from arcs.simulate import (
    ALL_TEST_SPECS,
    TIER1_TEST_SPECS,
    TIER2_TEST_SPECS,
    _TIER1_NAMES,
    _SPEC_TO_COND_POWER,
    _SPEC_TO_COND_SIGNAL,
    _power_reward,
    _signal_reward,
    SimulationOutcome,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Result of a single parameter trial."""
    topology: str
    specs: dict[str, float]
    params: dict[str, float]
    outcome: SimulationOutcome
    reward: float


def _specs_to_conditions(topology: str, specs: dict[str, float],
                         template: TopologyTemplate) -> dict[str, float]:
    """Merge test specs into operating conditions for simulation."""
    conditions = dict(template.operating_conditions)
    mapping = _SPEC_TO_COND_POWER if topology in _TIER1_NAMES else _SPEC_TO_COND_SIGNAL
    for spec_key, cond_key in mapping.items():
        if spec_key in specs:
            conditions[cond_key] = specs[spec_key]
    return conditions


def _simulate_params(
    topology: str,
    params: dict[str, float],
    specs: dict[str, float],
    template: TopologyTemplate,
    runner: NGSpiceRunner,
) -> tuple[SimulationOutcome, float]:
    """Simulate a parameter set and compute reward.

    Returns (outcome, reward). Reward range: [0, 8.0].
    """
    conditions = _specs_to_conditions(topology, specs, template)

    # Build netlist
    old_conds = template.operating_conditions
    template.operating_conditions = conditions
    try:
        netlist = template.generate_netlist(params)
    except Exception as e:
        template.operating_conditions = old_conds
        return SimulationOutcome(success=False, error=f"Netlist error: {e}"), 0.0
    finally:
        template.operating_conditions = old_conds

    # Simulate
    try:
        sim_result = runner.run(netlist, template.metric_names)
    except Exception as e:
        return SimulationOutcome(success=False, error=f"Sim error: {e}"), 0.0

    if not sim_result.success:
        return (
            SimulationOutcome(
                success=False,
                error=sim_result.error_message or "Simulation failed",
                sim_time=sim_result.sim_time_seconds,
            ),
            0.0,  # no struct_bonus for baselines (params always valid)
        )

    # Derive metrics
    try:
        metrics = compute_derived_metrics(
            sim_result.metrics, conditions, topology
        )
        valid = is_valid_result(metrics, conditions, topology)
    except Exception as e:
        outcome = SimulationOutcome(
            success=True,
            metrics=sim_result.metrics,
            valid=False,
            error=f"Metric error: {e}",
            sim_time=sim_result.sim_time_seconds,
        )
        return outcome, 1.0  # sim converged but metric error

    outcome = SimulationOutcome(
        success=True,
        metrics=metrics,
        valid=valid,
        sim_time=sim_result.sim_time_seconds,
    )

    # Compute reward: struct_bonus(1) + sim_converge(1) + domain(6) = max 8.0
    # Baselines always have valid structure since we sample within bounds
    reward = 2.0  # struct(1) + sim_converge(1)
    if topology in _TIER1_NAMES:
        reward += _power_reward(outcome, specs)
    else:
        reward += _signal_reward(outcome, topology)

    return outcome, reward


# ---------------------------------------------------------------------------
# Random Search
# ---------------------------------------------------------------------------

def random_search(
    topology: str,
    specs: dict[str, float],
    n_trials: int = 200,
    seed: int | None = None,
    runner: NGSpiceRunner | None = None,
) -> TrialResult:
    """Random search baseline: sample N parameter sets, return the best.

    Samples component values log-uniformly within ComponentBounds (same
    distribution as training data generation).

    Args:
        topology: Topology name (e.g., "buck", "inverting_amp")
        specs: Target specifications
        n_trials: Number of random trials
        seed: Random seed for reproducibility
        runner: NGSpiceRunner instance (creates one if None)

    Returns:
        TrialResult for the best trial found
    """
    template = get_topology(topology)
    rng = np.random.default_rng(seed)
    if runner is None:
        runner = NGSpiceRunner()

    best: TrialResult | None = None

    for i in range(n_trials):
        params = template.sample_parameters(rng)
        outcome, reward = _simulate_params(topology, params, specs, template, runner)

        if best is None or reward > best.reward:
            best = TrialResult(
                topology=topology,
                specs=specs,
                params=dict(params),
                outcome=outcome,
                reward=reward,
            )

    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

def _encode_log(params: dict[str, float], bounds: list[ComponentBounds]) -> np.ndarray:
    """Encode parameters to log-space representation for GA operations."""
    genes = []
    for b in bounds:
        val = params.get(b.name, math.sqrt(b.min_val * b.max_val))
        val = max(b.min_val, min(b.max_val, val))
        genes.append(math.log(val) if b.log_scale else val)
    return np.array(genes)


def _decode_log(genes: np.ndarray, bounds: list[ComponentBounds]) -> dict[str, float]:
    """Decode log-space genes back to parameter dict."""
    params = {}
    for i, b in enumerate(bounds):
        val = math.exp(genes[i]) if b.log_scale else genes[i]
        val = max(b.min_val, min(b.max_val, val))
        params[b.name] = val
    return params


def _tournament_select(
    population: list[np.ndarray],
    fitnesses: list[float],
    rng: np.random.Generator,
    k: int = 3,
) -> np.ndarray:
    """Tournament selection: pick k individuals, return the fittest."""
    indices = rng.choice(len(population), size=k, replace=False)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx].copy()


def _blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """BLX-alpha crossover for real-valued genes."""
    diff = np.abs(parent1 - parent2)
    low = np.minimum(parent1, parent2) - alpha * diff
    high = np.maximum(parent1, parent2) + alpha * diff
    child1 = rng.uniform(low, high)
    child2 = rng.uniform(low, high)
    return child1, child2


def _gaussian_mutate(
    genes: np.ndarray,
    bounds: list[ComponentBounds],
    rng: np.random.Generator,
    mutation_rate: float = 0.2,
    mutation_scale: float = 0.1,
) -> np.ndarray:
    """Gaussian mutation in log-space with per-gene probability."""
    mutated = genes.copy()
    for i, b in enumerate(bounds):
        if rng.random() < mutation_rate:
            # Scale relative to parameter range in log-space
            if b.log_scale:
                span = math.log(b.max_val) - math.log(b.min_val)
            else:
                span = b.max_val - b.min_val
            sigma = mutation_scale * span
            mutated[i] += rng.normal(0, sigma)
            # Clamp to bounds in the encoded space
            if b.log_scale:
                mutated[i] = max(math.log(b.min_val), min(math.log(b.max_val), mutated[i]))
            else:
                mutated[i] = max(b.min_val, min(b.max_val, mutated[i]))
    return mutated


def genetic_algorithm(
    topology: str,
    specs: dict[str, float],
    pop_size: int = 30,
    n_generations: int = 20,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.2,
    mutation_scale: float = 0.1,
    elitism: int = 2,
    seed: int | None = None,
    runner: NGSpiceRunner | None = None,
) -> TrialResult:
    """Genetic algorithm baseline: evolve component values for max reward.

    Uses real-valued GA in log-space with BLX-alpha crossover and
    Gaussian mutation. Selection via tournament.

    Total simulations = pop_size + n_generations × pop_size = pop_size × (1 + n_generations).

    Args:
        topology: Topology name
        specs: Target specifications
        pop_size: Population size per generation
        n_generations: Number of generations
        crossover_rate: Probability of crossover (per pair)
        mutation_rate: Per-gene mutation probability
        mutation_scale: Mutation std as fraction of parameter log-range
        elitism: Number of top individuals to carry forward unchanged
        seed: Random seed
        runner: NGSpiceRunner instance

    Returns:
        TrialResult for the best individual found across all generations
    """
    template = get_topology(topology)
    bounds = template.component_bounds
    rng = np.random.default_rng(seed)
    if runner is None:
        runner = NGSpiceRunner()

    # Initialize population
    population: list[np.ndarray] = []
    for _ in range(pop_size):
        params = template.sample_parameters(rng)
        population.append(_encode_log(params, bounds))

    # Evaluate initial population
    fitnesses = []
    best_overall: TrialResult | None = None
    for genes in population:
        params = _decode_log(genes, bounds)
        outcome, reward = _simulate_params(topology, params, specs, template, runner)
        fitnesses.append(reward)
        if best_overall is None or reward > best_overall.reward:
            best_overall = TrialResult(
                topology=topology, specs=specs, params=dict(params),
                outcome=outcome, reward=reward,
            )

    # Evolve
    for gen in range(n_generations):
        # Sort by fitness (descending)
        order = np.argsort(fitnesses)[::-1]
        sorted_pop = [population[i] for i in order]
        sorted_fit = [fitnesses[i] for i in order]

        new_pop: list[np.ndarray] = []
        new_fit: list[float] = []

        # Elitism: carry top individuals unchanged
        for i in range(min(elitism, pop_size)):
            new_pop.append(sorted_pop[i].copy())
            new_fit.append(sorted_fit[i])

        # Fill rest via crossover + mutation
        while len(new_pop) < pop_size:
            p1 = _tournament_select(population, fitnesses, rng)
            p2 = _tournament_select(population, fitnesses, rng)

            if rng.random() < crossover_rate:
                c1, c2 = _blend_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = _gaussian_mutate(c1, bounds, rng, mutation_rate, mutation_scale)
            c2 = _gaussian_mutate(c2, bounds, rng, mutation_rate, mutation_scale)

            for child_genes in [c1, c2]:
                if len(new_pop) >= pop_size:
                    break
                params = _decode_log(child_genes, bounds)
                outcome, reward = _simulate_params(
                    topology, params, specs, template, runner
                )
                new_pop.append(child_genes)
                new_fit.append(reward)

                if reward > best_overall.reward:  # type: ignore
                    best_overall = TrialResult(
                        topology=topology, specs=specs, params=dict(params),
                        outcome=outcome, reward=reward,
                    )

        population = new_pop
        fitnesses = new_fit

        logger.debug(
            f"GA {topology} gen {gen+1}/{n_generations}: "
            f"best={max(fitnesses):.3f} avg={np.mean(fitnesses):.3f}"
        )

    assert best_overall is not None
    return best_overall


# ---------------------------------------------------------------------------
# Full evaluation: run baseline across all topologies
# ---------------------------------------------------------------------------

@dataclass
class BaselineResults:
    """Aggregated results from a baseline run."""
    method: str                                # "random_search" or "ga"
    n_specs: int                               # number of topology-spec combos
    trials_per_spec: int                       # random_search: n_trials, GA: pop×(1+gen)
    results: list[TrialResult] = field(default_factory=list)
    wall_time: float = 0.0

    # Aggregate metrics
    @property
    def sim_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.outcome.success) / len(self.results)

    @property
    def sim_valid_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.outcome.valid) / len(self.results)

    @property
    def avg_reward(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.reward for r in self.results) / len(self.results)

    def per_topology_stats(self) -> dict[str, dict[str, float]]:
        """Per-topology breakdown."""
        by_topo: dict[str, list[TrialResult]] = {}
        for r in self.results:
            by_topo.setdefault(r.topology, []).append(r)

        stats = {}
        for topo, trials in sorted(by_topo.items()):
            n = len(trials)
            stats[topo] = {
                "n": n,
                "sim_success": sum(1 for t in trials if t.outcome.success) / n,
                "sim_valid": sum(1 for t in trials if t.outcome.valid) / n,
                "avg_reward": sum(t.reward for t in trials) / n,
                "best_reward": max(t.reward for t in trials),
            }
        return stats

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_specs": self.n_specs,
            "trials_per_spec": self.trials_per_spec,
            "wall_time": self.wall_time,
            "sim_success_rate": self.sim_success_rate,
            "sim_valid_rate": self.sim_valid_rate,
            "avg_reward": self.avg_reward,
            "per_topology": self.per_topology_stats(),
            "results": [
                {
                    "topology": r.topology,
                    "specs": r.specs,
                    "reward": r.reward,
                    "sim_success": r.outcome.success,
                    "sim_valid": r.outcome.valid,
                    "metrics": r.outcome.metrics,
                }
                for r in self.results
            ],
        }


def run_baseline(
    method: str = "random_search",
    test_specs: list[tuple[str, dict[str, float]]] | None = None,
    n_repeats: int = 10,
    # Random search params
    rs_trials: int = 200,
    # GA params
    ga_pop_size: int = 30,
    ga_generations: int = 20,
    seed: int = 42,
    tier: int | None = None,
) -> BaselineResults:
    """Run a baseline method across all test specs.

    Args:
        method: "random_search" or "ga"
        test_specs: List of (topology, specs) tuples. Defaults to ALL_TEST_SPECS.
        n_repeats: Number of independent runs per spec (to get variance)
        rs_trials: Number of random trials (random_search only)
        ga_pop_size: Population size (GA only)
        ga_generations: Number of generations (GA only)
        seed: Base random seed
        tier: Filter to tier 1 or 2 only

    Returns:
        BaselineResults with aggregated metrics
    """
    if test_specs is None:
        if tier == 1:
            test_specs = TIER1_TEST_SPECS
        elif tier == 2:
            test_specs = TIER2_TEST_SPECS
        else:
            test_specs = ALL_TEST_SPECS

    runner = NGSpiceRunner()
    all_results: list[TrialResult] = []

    if method == "random_search":
        trials_per = rs_trials
    else:
        trials_per = ga_pop_size * (1 + ga_generations)

    total_evals = len(test_specs) * n_repeats
    logger.info(
        f"Running {method} baseline: {len(test_specs)} specs × "
        f"{n_repeats} repeats = {total_evals} evaluations"
    )

    t0 = time.time()

    for spec_idx, (topology, specs) in enumerate(test_specs):
        for rep in range(n_repeats):
            rep_seed = seed + spec_idx * 1000 + rep

            if method == "random_search":
                result = random_search(
                    topology, specs,
                    n_trials=rs_trials,
                    seed=rep_seed,
                    runner=runner,
                )
            elif method == "ga":
                result = genetic_algorithm(
                    topology, specs,
                    pop_size=ga_pop_size,
                    n_generations=ga_generations,
                    seed=rep_seed,
                    runner=runner,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            all_results.append(result)

        n_done = (spec_idx + 1) * n_repeats
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (total_evals - n_done) / rate if rate > 0 else 0
        logger.info(
            f"  [{n_done}/{total_evals}] {topology}: "
            f"best_reward={all_results[-1].reward:.3f}  "
            f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
        )

    wall_time = time.time() - t0

    return BaselineResults(
        method=method,
        n_specs=len(test_specs) * n_repeats,
        trials_per_spec=trials_per,
        results=all_results,
        wall_time=wall_time,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ARCS baseline methods (random search / genetic algorithm)"
    )
    parser.add_argument(
        "--method", choices=["random_search", "ga", "both"],
        default="both", help="Baseline method to run"
    )
    parser.add_argument(
        "--tier", type=int, choices=[1, 2], default=None,
        help="Run only tier 1 or tier 2 topologies"
    )
    parser.add_argument(
        "--n-repeats", type=int, default=10,
        help="Independent repeats per spec"
    )
    parser.add_argument(
        "--rs-trials", type=int, default=200,
        help="Random search: trials per spec"
    )
    parser.add_argument(
        "--ga-pop", type=int, default=30,
        help="GA: population size"
    )
    parser.add_argument(
        "--ga-gens", type=int, default=20,
        help="GA: number of generations"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: results/baseline_{method}.json)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    methods = ["random_search", "ga"] if args.method == "both" else [args.method]
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {method} baseline")
        logger.info(f"{'='*60}")

        results = run_baseline(
            method=method,
            n_repeats=args.n_repeats,
            rs_trials=args.rs_trials,
            ga_pop_size=args.ga_pop,
            ga_generations=args.ga_gens,
            seed=args.seed,
            tier=args.tier,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"{method.upper()} RESULTS")
        print(f"{'='*60}")
        print(f"Specs evaluated:    {results.n_specs}")
        print(f"Trials per spec:    {results.trials_per_spec}")
        print(f"Wall time:          {results.wall_time:.1f}s")
        print(f"Sim success rate:   {results.sim_success_rate:.1%}")
        print(f"Sim valid rate:     {results.sim_valid_rate:.1%}")
        print(f"Avg reward:         {results.avg_reward:.3f}")
        print()

        print("Per-topology breakdown:")
        print(f"  {'Topology':<25s} {'N':>4s} {'SimOK':>7s} {'Valid':>7s} {'AvgR':>7s} {'BestR':>7s}")
        print(f"  {'-'*25} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for topo, stats in results.per_topology_stats().items():
            print(
                f"  {topo:<25s} {int(stats['n']):>4d} "
                f"{stats['sim_success']:>6.1%} {stats['sim_valid']:>6.1%} "
                f"{stats['avg_reward']:>7.3f} {stats['best_reward']:>7.3f}"
            )

        # Save results
        outpath = args.output or str(results_dir / f"baseline_{method}.json")
        if args.tier:
            outpath = outpath.replace(".json", f"_t{args.tier}.json")
        with open(outpath, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Results saved to {outpath}")

    # If both methods run, print comparison table
    if len(methods) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        # Re-load and compare…  (results are already in memory from the loop)
        print("(See individual JSON files for detailed results)")


if __name__ == "__main__":
    main()
