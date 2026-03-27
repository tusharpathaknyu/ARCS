#!/usr/bin/env python3
"""Publication-ready evaluation: baselines + ablation + statistical analysis.

Runs 6 methods on 32 topologies (excluding flyback/forward) with N samples
per topology, computing bootstrap confidence intervals and pairwise tests.

Methods (also serves as ablation study for components 3-6):
  1. random_search  — Random parameter search baseline
  2. ga             — Genetic algorithm baseline
  3. vcg_only       — VCG generation alone
  4. ccfm_only      — CCFM generation alone
  5. hybrid_ranked  — VCG + CCFM ranked by SPICE reward
  6. hybrid_reward  — Full hybrid + latent reward pre-ranking

Usage:
    # Quick smoke test (~2 min):
    PYTHONPATH=src python scripts/evaluate_publication.py \\
        --vcg checkpoints/vcg_v5/best_model.pt \\
        --ccfm checkpoints/ccfm_v5/best_ccfm.pt \\
        --methods vcg_only --n-samples 3 --output results/pub_smoke.json

    # Full evaluation (~4-8h):
    PYTHONPATH=src nohup python -u scripts/evaluate_publication.py \\
        --vcg checkpoints/vcg_v5/best_model.pt \\
        --ccfm checkpoints/ccfm_v5/best_ccfm.pt \\
        --reward-model checkpoints/latent_reward_v5/best_reward_predictor.pt \\
        --n-samples 50 --n-baseline-repeats 20 \\
        --output results/publication_eval.json \\
        --seed 42 > /tmp/pub_eval.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from arcs.baselines import run_baseline
from arcs.flow_matching import ConstrainedFlowMatchingModel
from arcs.hybrid_pipeline import HybridGenerator, GeneratedCircuit
from arcs.simulate import ALL_TEST_SPECS, _TIER1_NAMES, compute_reward
from arcs.valid_circuit_gen import ValidCircuitGenModel, VCGConfig
from arcs import DEFAULT_TEMPERATURE, DEFAULT_TOP_K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Exclude topologies with broken SPICE templates
EXCLUDED_TOPOLOGIES = {"flyback", "forward"}
EVAL_SPECS = [(t, s) for t, s in ALL_TEST_SPECS if t not in EXCLUDED_TOPOLOGIES]

# Power topologies for per-spec metrics
_POWER_TOPOS = (set(_TIER1_NAMES) - EXCLUDED_TOPOLOGIES) | {
    "half_bridge", "push_pull", "charge_pump", "voltage_doubler", "zeta_converter",
}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return (v, v)
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def compute_topology_stats(
    rewards: list[float], n_boot: int = 10_000,
) -> dict:
    """Compute per-topology statistics from a list of reward values."""
    arr = np.array(rewards) if rewards else np.array([0.0])
    ci_lo, ci_hi = bootstrap_ci(rewards, n_boot=n_boot)
    return {
        "n": len(rewards),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "median": float(np.median(arr)),
        "ci_95": [ci_lo, ci_hi],
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pairwise_test(
    rewards_a: dict[str, list[float]],
    rewards_b: dict[str, list[float]],
) -> dict:
    """Wilcoxon signed-rank test on per-topology mean rewards."""
    common_topos = sorted(set(rewards_a.keys()) & set(rewards_b.keys()))
    if len(common_topos) < 5:
        return {"p_value": None, "note": "too few topologies"}
    means_a = [np.mean(rewards_a[t]) for t in common_topos]
    means_b = [np.mean(rewards_b[t]) for t in common_topos]
    diff = np.array(means_a) - np.array(means_b)
    # If all differences are zero, skip
    if np.all(diff == 0):
        return {"p_value": 1.0, "mean_diff": 0.0, "n_topologies": len(common_topos)}
    stat, p = scipy_stats.wilcoxon(means_a, means_b, alternative="two-sided")
    return {
        "p_value": float(p),
        "statistic": float(stat),
        "mean_diff": float(np.mean(diff)),
        "n_topologies": len(common_topos),
    }


# ---------------------------------------------------------------------------
# Per-spec accuracy (for power converters)
# ---------------------------------------------------------------------------

def extract_per_spec_metrics(
    topology: str, specs: dict, circuits: list[GeneratedCircuit],
) -> dict:
    """Extract per-spec accuracy from generated circuits."""
    result: dict[str, list[float]] = defaultdict(list)

    for circ in circuits:
        if circ.outcome is None or not circ.outcome.success:
            continue
        m = circ.outcome.metrics

        if topology in _POWER_TOPOS:
            # Vout accuracy
            vout_target = specs.get("vout")
            if vout_target and "vout_avg" in m:
                err = abs(m["vout_avg"] - abs(vout_target)) / abs(vout_target) * 100
                result["vout_error_pct"].append(err)
            # Efficiency
            if "efficiency" in m:
                result["efficiency"].append(m["efficiency"])
            # Ripple
            if "vout_ripple" in m and "vout_avg" in m and m["vout_avg"] != 0:
                result["ripple_pct"].append(abs(m["vout_ripple"] / m["vout_avg"]) * 100)

    # Aggregate
    return {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
        for k, v in result.items()
    }


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def run_baseline_method(
    method: str,
    eval_specs: list,
    n_repeats: int,
    rs_trials: int = 50,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Run baseline and return per-topology reward lists."""
    logger.info(f"Running baseline: {method} ({n_repeats} repeats × {len(eval_specs)} specs)")
    t0 = time.time()
    results = run_baseline(
        method=method,
        test_specs=eval_specs,
        n_repeats=n_repeats,
        rs_trials=rs_trials,
        ga_pop_size=20,
        ga_generations=15,
        seed=seed,
    )
    wall = time.time() - t0
    logger.info(f"  {method} done in {wall:.0f}s: avg_reward={results.avg_reward:.3f}")

    # Group rewards by topology
    per_topo: dict[str, list[float]] = defaultdict(list)
    for r in results.results:
        per_topo[r.topology].append(r.reward)
    return dict(per_topo)


def run_learned_method(
    method_name: str,
    hybrid: HybridGenerator,
    eval_specs: list,
    n_samples: int,
    seed: int = 42,
) -> tuple[dict[str, list[float]], dict[str, dict]]:
    """Run a learned method and return per-topology rewards + per-spec metrics."""
    logger.info(f"Running: {method_name} ({n_samples} samples × {len(eval_specs)} specs)")
    t0 = time.time()

    per_topo_rewards: dict[str, list[float]] = defaultdict(list)
    per_topo_spec_metrics: dict[str, list[dict]] = defaultdict(list)

    for spec_idx, (topology, specs) in enumerate(eval_specs):
        for i in range(n_samples):
            torch.manual_seed(seed + spec_idx * 10000 + i)
            np.random.seed(seed + spec_idx * 10000 + i + 1)

            try:
                if method_name == "vcg_only":
                    circuits = hybrid.generate_from_vcg(topology, specs, n_candidates=1)
                elif method_name == "ccfm_only":
                    circuits = hybrid.generate_from_ccfm(topology, specs, n_candidates=1)
                elif method_name == "hybrid_ranked":
                    best = hybrid.generate_best(
                        topology, specs,
                        n_candidates_per_source=4,
                        sources=["vcg", "ccfm"],
                    )
                    circuits = [best]
                elif method_name == "hybrid_reward":
                    best = hybrid.generate_best(
                        topology, specs,
                        n_candidates_per_source=4,
                        sources=["vcg", "ccfm"],
                        pre_rank_top_k=4,
                    )
                    circuits = [best]
                else:
                    raise ValueError(f"Unknown method: {method_name}")
            except Exception as e:
                logger.warning(f"  {method_name}/{topology} sample {i}: {e}")
                per_topo_rewards[topology].append(0.0)
                continue

            for circ in circuits:
                per_topo_rewards[topology].append(circ.reward)

            # Per-spec metrics
            sm = extract_per_spec_metrics(topology, specs, circuits)
            if sm:
                per_topo_spec_metrics[topology].append(sm)

        # Progress
        elapsed = time.time() - t0
        done = (spec_idx + 1)
        remaining = elapsed / done * (len(eval_specs) - done)
        best_r = max(per_topo_rewards[topology]) if per_topo_rewards[topology] else 0
        logger.info(
            f"  [{done}/{len(eval_specs)}] {topology}: "
            f"mean={np.mean(per_topo_rewards[topology]):.2f} "
            f"max={best_r:.2f}  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
        )

    wall = time.time() - t0
    logger.info(f"  {method_name} done in {wall:.0f}s")

    # Aggregate per-spec metrics
    aggregated_spec: dict[str, dict] = {}
    for topo, metric_list in per_topo_spec_metrics.items():
        merged: dict[str, list[float]] = defaultdict(list)
        for m in metric_list:
            for k, v in m.items():
                if "mean" in v:
                    merged[k].append(v["mean"])
        aggregated_spec[topo] = {
            k: {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
            for k, vals in merged.items()
        }

    return dict(per_topo_rewards), aggregated_spec


# ---------------------------------------------------------------------------
# Autoregressive model evaluation
# ---------------------------------------------------------------------------

def run_autoregressive_method(
    checkpoint: str,
    eval_specs: list,
    n_samples: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[dict[str, list[float]], dict[str, dict]]:
    """Run autoregressive ARCS model on spec-aware evaluation."""
    from arcs.model_enhanced import load_model, create_model, ARCSConfig
    from arcs.tokenizer import CircuitTokenizer
    from arcs.simulate import simulate_decoded_circuit

    logger.info(f"Loading autoregressive model from {checkpoint}")
    tokenizer = CircuitTokenizer()

    # Load model (handle both SL and RL checkpoint formats)
    try:
        model, config, mt = load_model(checkpoint, device=device)
    except RuntimeError:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
        config = ARCSConfig.from_dict(ckpt["config"])
        mt = ckpt.get("model_type", "graph_transformer")
        model = create_model(mt, config)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    per_topo_rewards: dict[str, list[float]] = defaultdict(list)
    per_topo_spec_metrics: dict[str, list[dict]] = defaultdict(list)
    t0 = time.time()

    for spec_idx, (topology, specs) in enumerate(eval_specs):
        for i in range(n_samples):
            torch.manual_seed(seed + spec_idx * 10000 + i)
            np.random.seed(seed + spec_idx * 10000 + i + 1)

            try:
                # Generate with autoregressive model
                from arcs.demo import generate_circuit
                circuit, _dt = generate_circuit(
                    model, tokenizer, device, topology, specs,
                    temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K,
                )
                if circuit is not None and circuit.valid_structure:
                    outcome = simulate_decoded_circuit(circuit, topology)
                    reward = compute_reward(outcome, topology, target_specs=specs)
                    per_topo_rewards[topology].append(reward)
                else:
                    per_topo_rewards[topology].append(0.0)
            except Exception as e:
                logger.warning(f"  arcs_grpo/{topology} sample {i}: {e}")
                per_topo_rewards[topology].append(0.0)

        elapsed = time.time() - t0
        done = spec_idx + 1
        remaining = elapsed / done * (len(eval_specs) - done)
        mean_r = np.mean(per_topo_rewards[topology])
        logger.info(
            f"  [{done}/{len(eval_specs)}] {topology}: mean={mean_r:.2f}"
            f"  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
        )

    wall = time.time() - t0
    logger.info(f"  arcs_grpo done in {wall:.0f}s")
    return dict(per_topo_rewards), {}


# ---------------------------------------------------------------------------
# LaTeX table formatting
# ---------------------------------------------------------------------------

def format_latex_table(all_results: dict[str, dict]) -> str:
    """Format comparison table for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of circuit generation methods across 32 topologies.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Sim Valid\% & Reward & Reward 95\% CI \\",
        r"\midrule",
    ]

    for method_name, data in all_results.items():
        rewards_all = []
        for topo_rewards in data["per_topology"].values():
            rewards_all.extend(topo_rewards)
        n = len(rewards_all)
        if n == 0:
            continue
        mean_r = np.mean(rewards_all)
        sim_valid = sum(1 for r in rewards_all if r > 2.0) / n * 100
        ci_lo, ci_hi = bootstrap_ci(rewards_all)
        display = method_name.replace("_", " ").title()
        lines.append(
            rf"{display} & {sim_valid:.1f} & {mean_r:.2f} & [{ci_lo:.2f}, {ci_hi:.2f}] \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready evaluation with baselines, ablation, and statistical analysis"
    )
    parser.add_argument("--vcg", type=str, default="checkpoints/vcg_v5/best_model.pt")
    parser.add_argument("--ccfm", type=str, default="checkpoints/ccfm_v5/best_ccfm.pt")
    parser.add_argument("--reward-model", type=str, default=None)
    parser.add_argument("--arcs-checkpoint", type=str, default=None,
                        help="Autoregressive ARCS checkpoint for arcs_grpo method")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Samples per topology for learned methods")
    parser.add_argument("--n-baseline-repeats", type=int, default=20,
                        help="Repeats per topology for baselines")
    parser.add_argument("--rs-trials", type=int, default=50,
                        help="Random search trials per repeat")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["random_search", "ga", "arcs_grpo", "vcg_only",
                                 "ccfm_only", "hybrid_ranked", "hybrid_reward"],
                        help="Methods to evaluate")
    parser.add_argument("--output", type=str, default="results/publication_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Eval specs: {len(EVAL_SPECS)} topologies (excluded: {EXCLUDED_TOPOLOGIES})")
    logger.info(f"Methods: {args.methods}")

    # Load models
    hybrid = None
    learned_methods = {"vcg_only", "ccfm_only", "hybrid_ranked", "hybrid_reward"}
    if any(m in args.methods for m in learned_methods):
        logger.info("Loading VCG from %s", args.vcg)
        ckpt = torch.load(args.vcg, map_location="cpu", weights_only=False)
        cfg = VCGConfig.from_dict(ckpt["config"])
        vcg_model = ValidCircuitGenModel(cfg).to(device)
        vcg_model.load_state_dict(ckpt["model"])
        vcg_model.eval()

        logger.info("Loading CCFM from %s", args.ccfm)
        ccfm_model = ConstrainedFlowMatchingModel.load(args.ccfm, device=device)
        ccfm_model.eval()

        hybrid = HybridGenerator(
            vcg_model=vcg_model, ccfm_model=ccfm_model, device=device,
        )
        logger.info("Models loaded")

    # Run each method
    all_results: dict[str, dict] = {}
    t0_total = time.time()

    for method in args.methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Method: {method}")
        logger.info(f"{'='*60}")

        if method in ("random_search", "ga"):
            per_topo = run_baseline_method(
                method=method,
                eval_specs=EVAL_SPECS,
                n_repeats=args.n_baseline_repeats,
                rs_trials=args.rs_trials,
                seed=args.seed,
            )
            per_spec = {}
        elif method == "arcs_grpo":
            arcs_ckpt = args.arcs_checkpoint or "checkpoints/arcs_grpo_extended/best_rl_model.pt"
            per_topo, per_spec = run_autoregressive_method(
                checkpoint=arcs_ckpt,
                eval_specs=EVAL_SPECS,
                n_samples=args.n_samples,
                device=device,
                seed=args.seed,
            )
        else:
            assert hybrid is not None, "Need --vcg and --ccfm for learned methods"
            per_topo, per_spec = run_learned_method(
                method_name=method,
                hybrid=hybrid,
                eval_specs=EVAL_SPECS,
                n_samples=args.n_samples,
                seed=args.seed,
            )

        # Compute stats
        topo_stats = {
            t: compute_topology_stats(rewards)
            for t, rewards in per_topo.items()
        }

        # Aggregate across topologies
        all_rewards = []
        for rewards in per_topo.values():
            all_rewards.extend(rewards)
        overall_stats = compute_topology_stats(all_rewards)
        sim_valid_count = sum(1 for r in all_rewards if r > 2.0)
        overall_stats["sim_valid_rate"] = sim_valid_count / len(all_rewards) if all_rewards else 0.0

        all_results[method] = {
            "overall": overall_stats,
            "per_topology": per_topo,
            "per_topology_stats": topo_stats,
            "per_spec_accuracy": per_spec,
        }

        logger.info(
            f"  → {method}: reward={overall_stats['mean']:.3f} "
            f"[{overall_stats['ci_95'][0]:.3f}, {overall_stats['ci_95'][1]:.3f}] "
            f"sim_valid={overall_stats['sim_valid_rate']:.1%}"
        )

    # Pairwise statistical tests
    logger.info(f"\n{'='*60}")
    logger.info("Pairwise statistical tests")
    logger.info(f"{'='*60}")

    stat_tests = {}
    if "hybrid_ranked" in all_results:
        for other in args.methods:
            if other == "hybrid_ranked":
                continue
            if other in all_results:
                key = f"hybrid_ranked_vs_{other}"
                test = pairwise_test(
                    all_results["hybrid_ranked"]["per_topology"],
                    all_results[other]["per_topology"],
                )
                stat_tests[key] = test
                p = test.get("p_value")
                d = test.get("mean_diff", 0)
                logger.info(f"  hybrid_ranked vs {other}: p={p:.4f}, Δ={d:+.3f}" if p else f"  {key}: skipped")

    # Format LaTeX table
    latex = format_latex_table(all_results)
    logger.info(f"\nLaTeX table:\n{latex}")

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Method':<20} {'Reward':>8} {'95% CI':>18} {'SimValid%':>10}")
    logger.info("-" * 60)
    for method, data in all_results.items():
        s = data["overall"]
        ci = s["ci_95"]
        logger.info(
            f"{method:<20} {s['mean']:>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {s['sim_valid_rate']:>9.1%}"
        )

    # Save results
    total_wall = time.time() - t0_total
    output = {
        "config": {
            "n_samples": args.n_samples,
            "n_baseline_repeats": args.n_baseline_repeats,
            "rs_trials": args.rs_trials,
            "n_topologies": len(EVAL_SPECS),
            "excluded": sorted(EXCLUDED_TOPOLOGIES),
            "seed": args.seed,
            "methods": args.methods,
            "wall_time_sec": total_wall,
        },
        "results": {
            method: {
                "overall": data["overall"],
                "per_topology_stats": data["per_topology_stats"],
                "per_spec_accuracy": data.get("per_spec_accuracy", {}),
            }
            for method, data in all_results.items()
        },
        "statistical_tests": stat_tests,
        "latex_table": latex,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nResults saved to {out_path} ({total_wall:.0f}s total)")


if __name__ == "__main__":
    main()
