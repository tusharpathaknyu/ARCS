#!/usr/bin/env python3
"""Warm-start experiment: ARCS generates initial design, then GA refines it.

This tests the "best of both worlds" approach mentioned in the paper:
- ARCS provides a fast initial design (~20ms)
- GA refines it from that starting point (much fewer sims needed)

Compares:
1. GA from scratch (pop=30, 20 gens, ~630 sims)
2. GA warm-started from ARCS (pop=30, 10 gens, ~330 sims)
3. ARCS only (1 sim)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import os

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import decode_generated_sequence
from arcs.simulate import (
    components_to_params, ALL_TEST_SPECS, _TIER1_NAMES,
    SimulationOutcome,
)
from arcs.baselines import (
    genetic_algorithm, _simulate_params, _encode_log, _decode_log,
    _tournament_select, _blend_crossover, _gaussian_mutate,
    TrialResult,
)
from arcs.templates import get_topology
from arcs.spice import NGSpiceRunner


def _build_prefix(tokenizer, topology: str, specs: dict[str, float]) -> list[int]:
    """Build a conditioned spec prefix token sequence."""
    prefix_ids = [tokenizer.start_id]
    topo_key = f"TOPO_{topology.upper()}"
    _topo_to_token = {
        "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
        "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
        "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
    }
    topo_key = _topo_to_token.get(topology, topo_key)
    if topo_key in tokenizer.name_to_id:
        prefix_ids.append(tokenizer.name_to_id[topo_key])
    prefix_ids.append(tokenizer.sep_id)
    for spec_name, spec_val in specs.items():
        spec_key = f"SPEC_{spec_name.upper()}"
        if spec_key in tokenizer.name_to_id:
            prefix_ids.append(tokenizer.name_to_id[spec_key])
            prefix_ids.append(tokenizer.encode_value(abs(spec_val)))
    prefix_ids.append(tokenizer.sep_id)
    return prefix_ids


def generate_arcs_design(
    model: ARCSModel,
    tokenizer: CircuitTokenizer,
    topology: str,
    specs: dict[str, float],
    device: torch.device,
    n_candidates: int = 5,
    temperature: float = 0.8,
) -> tuple[dict[str, float] | None, float]:
    """Generate ARCS design and extract params. Try multiple candidates.

    Returns: (best_params, arcs_reward) or (None, 0.0)
    """
    runner = NGSpiceRunner()
    template = get_topology(topology)

    best_params = None
    best_reward = 0.0

    for _ in range(n_candidates):
        # Build prefix
        prefix = _build_prefix(tokenizer, topology, specs)
        prefix_tensor = torch.tensor([prefix], device=device)

        # Generate
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                prefix_tensor,
                max_new_tokens=40,
                temperature=temperature,
                top_k=50,
            )

        all_ids = prefix + generated[0].tolist()
        decoded = decode_generated_sequence(all_ids, tokenizer)

        if not decoded.valid_structure or not decoded.topology:
            continue

        # Extract params
        params = components_to_params(decoded.topology, decoded.components)
        if params is None:
            continue

        # Simulate to get reward
        outcome, reward = _simulate_params(
            topology, params, specs, template, runner
        )

        if reward > best_reward:
            best_reward = reward
            best_params = params

    return best_params, best_reward


def warm_start_ga(
    topology: str,
    specs: dict[str, float],
    initial_params: dict[str, float],
    pop_size: int = 30,
    n_generations: int = 10,
    seed: int | None = None,
    runner: NGSpiceRunner | None = None,
) -> TrialResult:
    """GA that starts from an ARCS-generated design.

    The initial population is seeded with the ARCS design and mutations
    of it, rather than purely random individuals. This gives the GA
    a "warm start" in a promising region of parameter space.
    """
    template = get_topology(topology)
    bounds = template.component_bounds
    rng = np.random.default_rng(seed)
    if runner is None:
        runner = NGSpiceRunner()

    # Encode the ARCS design
    arcs_genes = _encode_log(initial_params, bounds)

    # Initialize population: 1 exact copy + rest are mutations of it
    # (varying mutation scales for diversity)
    population: list[np.ndarray] = [arcs_genes.copy()]
    for i in range(pop_size - 1):
        scale = 0.05 + 0.3 * (i / (pop_size - 1))  # 0.05 to 0.35
        mutated = _gaussian_mutate(arcs_genes.copy(), bounds, rng,
                                   mutation_rate=0.8, mutation_scale=scale)
        population.append(mutated)

    # Evaluate initial population
    fitnesses: list[float] = []
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

    n_sims = pop_size

    # Evolve
    for gen in range(n_generations):
        order = np.argsort(fitnesses)[::-1]
        sorted_pop = [population[i] for i in order]
        sorted_fit = [fitnesses[i] for i in order]

        new_pop: list[np.ndarray] = []
        new_fit: list[float] = []

        # Elitism
        for i in range(min(2, pop_size)):
            new_pop.append(sorted_pop[i].copy())
            new_fit.append(sorted_fit[i])

        while len(new_pop) < pop_size:
            p1 = _tournament_select(population, fitnesses, rng)
            p2 = _tournament_select(population, fitnesses, rng)

            if rng.random() < 0.7:
                c1, c2 = _blend_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = _gaussian_mutate(c1, bounds, rng, 0.2, 0.1)
            c2 = _gaussian_mutate(c2, bounds, rng, 0.2, 0.1)

            for child in [c1, c2]:
                if len(new_pop) >= pop_size:
                    break
                params = _decode_log(child, bounds)
                outcome, reward = _simulate_params(
                    topology, params, specs, template, runner
                )
                new_pop.append(child)
                new_fit.append(reward)
                n_sims += 1

                if reward > best_overall.reward:
                    best_overall = TrialResult(
                        topology=topology, specs=specs, params=dict(params),
                        outcome=outcome, reward=reward,
                    )

        population = new_pop
        fitnesses = new_fit

    return best_overall, n_sims


def main():
    parser = argparse.ArgumentParser(description="Warm-start experiment")
    parser.add_argument("--checkpoint", default="checkpoints/arcs_rl_v2/best_rl_model.pt")
    parser.add_argument("--n-candidates", type=int, default=5,
                        help="ARCS candidates per spec")
    parser.add_argument("--ga-pop", type=int, default=30)
    parser.add_argument("--ga-gens-cold", type=int, default=20,
                        help="GA generations (cold start)")
    parser.add_argument("--ga-gens-warm", type=int, default=10,
                        help="GA generations (warm start)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/warm_start.json")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ARCSConfig.from_dict(ckpt["config"])
    model = ARCSModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = CircuitTokenizer()
    runner = NGSpiceRunner()

    # Use a subset of test specs for reasonable runtime
    test_specs = ALL_TEST_SPECS
    n_specs = len(test_specs)
    print(f"Running warm-start experiment on {n_specs} specs...")

    results = {
        "arcs_only": [],
        "ga_cold": [],
        "ga_warm": [],
    }

    for i, (topo, specs) in enumerate(test_specs):
        print(f"\n[{i+1}/{n_specs}] {topo}: {specs}")

        # 1. ARCS only
        t0 = time.time()
        arcs_params, arcs_reward = generate_arcs_design(
            model, tokenizer, topo, specs, device,
            n_candidates=args.n_candidates, temperature=0.8,
        )
        arcs_time = time.time() - t0
        print(f"  ARCS: reward={arcs_reward:.3f} ({arcs_time:.1f}s, {args.n_candidates} candidates)")

        results["arcs_only"].append({
            "topology": topo, "reward": arcs_reward,
            "time": arcs_time, "sims": args.n_candidates,
            "has_params": arcs_params is not None,
        })

        # 2. GA cold start
        t0 = time.time()
        cold_result = genetic_algorithm(
            topo, specs, pop_size=args.ga_pop,
            n_generations=args.ga_gens_cold, seed=args.seed, runner=runner,
        )
        cold_time = time.time() - t0
        cold_sims = args.ga_pop * (1 + args.ga_gens_cold)
        print(f"  GA cold: reward={cold_result.reward:.3f} ({cold_time:.1f}s, {cold_sims} sims)")

        results["ga_cold"].append({
            "topology": topo, "reward": cold_result.reward,
            "time": cold_time, "sims": cold_sims,
        })

        # 3. GA warm start (only if ARCS produced valid params)
        if arcs_params is not None:
            t0 = time.time()
            warm_result, warm_sims = warm_start_ga(
                topo, specs, arcs_params,
                pop_size=args.ga_pop, n_generations=args.ga_gens_warm,
                seed=args.seed, runner=runner,
            )
            warm_time = time.time() - t0 + arcs_time  # include ARCS time
            print(f"  GA warm: reward={warm_result.reward:.3f} ({warm_time:.1f}s, {warm_sims}+{args.n_candidates} sims)")

            results["ga_warm"].append({
                "topology": topo, "reward": warm_result.reward,
                "time": warm_time, "sims": warm_sims + args.n_candidates,
            })
        else:
            # Fallback: warm-start failed, use cold GA result
            print(f"  GA warm: SKIPPED (ARCS produced no valid params)")
            results["ga_warm"].append({
                "topology": topo, "reward": cold_result.reward,
                "time": cold_time, "sims": cold_sims,
                "fallback": True,
            })

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WARM-START EXPERIMENT RESULTS")
    print("=" * 80)

    for method in ["arcs_only", "ga_cold", "ga_warm"]:
        rewards = [r["reward"] for r in results[method]]
        times = [r["time"] for r in results[method]]
        sims = [r["sims"] for r in results[method]]
        avg_reward = np.mean(rewards)
        avg_time = np.mean(times)
        avg_sims = np.mean(sims)
        print(f"\n{method:12s}: reward={avg_reward:.3f}  time={avg_time:.1f}s  sims={avg_sims:.0f}")

    # Per-topology comparison
    print(f"\n{'Topology':<15s} {'ARCS':>8s} {'GA cold':>8s} {'GA warm':>8s} {'Δ warm-cold':>12s}")
    print("-" * 55)
    for i in range(n_specs):
        topo = results["arcs_only"][i]["topology"]
        arcs_r = results["arcs_only"][i]["reward"]
        cold_r = results["ga_cold"][i]["reward"]
        warm_r = results["ga_warm"][i]["reward"]
        delta = warm_r - cold_r
        marker = "✓" if delta > 0 else ""
        print(f"{topo:<15s} {arcs_r:8.3f} {cold_r:8.3f} {warm_r:8.3f} {delta:+8.3f}    {marker}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
