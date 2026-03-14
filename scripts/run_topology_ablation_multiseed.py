#!/usr/bin/env python3
"""Run topology ablation evaluation across multiple seeds and summarize mean/std.

Uses existing checkpoints (no training) and calls scripts/evaluate_topology_ablation.py
for each seed. Writes per-seed JSON plus aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    w = idx - lo
    return sorted_values[lo] * (1 - w) + sorted_values[hi] * w


def _bootstrap_ci(values: list[float], n_bootstrap: int, ci: float, rng: random.Random) -> dict:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": _mean(values),
        "low": _percentile(means, alpha),
        "high": _percentile(means, 1.0 - alpha),
    }


def _bootstrap_diff_ci(
    a_values: list[float],
    b_values: list[float],
    n_bootstrap: int,
    ci: float,
    rng: random.Random,
) -> dict:
    if not a_values or not b_values:
        return {"mean_diff": 0.0, "low": 0.0, "high": 0.0, "p_gt_zero": 0.0}
    na, nb = len(a_values), len(b_values)
    diffs: list[float] = []
    gt_zero = 0
    for _ in range(n_bootstrap):
        a_mean = _mean([a_values[rng.randrange(na)] for _ in range(na)])
        b_mean = _mean([b_values[rng.randrange(nb)] for _ in range(nb)])
        d = a_mean - b_mean
        diffs.append(d)
        if d > 0:
            gt_zero += 1
    diffs.sort()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean_diff": _mean(a_values) - _mean(b_values),
        "low": _percentile(diffs, alpha),
        "high": _percentile(diffs, 1.0 - alpha),
        "p_gt_zero": gt_zero / n_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topology ablation eval for multiple seeds")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/arcs_graph_transformer/best_model.pt")
    parser.add_argument("--topo-head-ckpt", type=str, default="checkpoints/arcs_graph_transformer_topo_value_medium/best_model.pt")
    parser.add_argument("--family-moe-ckpt", type=str, default="checkpoints/arcs_graph_transformer_family_moe_medium/best_model.pt")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--output", type=str, default="results/topology_ablation_medium_multiseed.json")
    parser.add_argument("--python", type=str, default=".venv/bin/python")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.bootstrap_seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed_dir = output_path.parent / "topology_ablation_multiseed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    per_seed_files: list[Path] = []
    for seed in args.seeds:
        out_file = per_seed_dir / f"seed_{seed}.json"
        per_seed_files.append(out_file)
        cmd = [
            args.python,
            "scripts/evaluate_topology_ablation.py",
            "--baseline-ckpt",
            args.baseline_ckpt,
            "--topo-head-ckpt",
            args.topo_head_ckpt,
            "--family-moe-ckpt",
            args.family_moe_ckpt,
            "--n-samples",
            str(args.n_samples),
            "--seed",
            str(seed),
            "--output",
            str(out_file),
        ]
        print(f"\n[seed={seed}] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    rows_by_model: dict[str, list[dict]] = {}
    for file_path in per_seed_files:
        rows = json.loads(file_path.read_text())
        for row in rows:
            rows_by_model.setdefault(row["name"], []).append(row)

    summary: list[dict] = []
    raw_metrics_by_model: dict[str, dict[str, list[float]]] = {}
    for model_name, rows in rows_by_model.items():
        sim_valid = [float(r["sim_valid_rate"]) for r in rows]
        struct_valid = [float(r["valid_structure_rate"]) for r in rows]
        reward = [float(r["mean_reward"]) for r in rows]
        raw_metrics_by_model[model_name] = {
            "sim_valid": sim_valid,
            "struct": struct_valid,
            "reward": reward,
        }
        sim_valid_ci = _bootstrap_ci(sim_valid, args.bootstrap_iters, args.ci, rng)
        struct_ci = _bootstrap_ci(struct_valid, args.bootstrap_iters, args.ci, rng)
        reward_ci = _bootstrap_ci(reward, args.bootstrap_iters, args.ci, rng)
        summary.append(
            {
                "name": model_name,
                "n_runs": len(rows),
                "seeds": [int(r.get("seed", -1)) for r in rows],
                "sim_valid_mean": _mean(sim_valid),
                "sim_valid_std": _std(sim_valid),
                "sim_valid_ci": sim_valid_ci,
                "struct_mean": _mean(struct_valid),
                "struct_std": _std(struct_valid),
                "struct_ci": struct_ci,
                "reward_mean": _mean(reward),
                "reward_std": _std(reward),
                "reward_ci": reward_ci,
            }
        )

    baseline_name = "GraphTransformer Baseline"
    pairwise_vs_baseline: dict[str, dict] = {}
    if baseline_name in raw_metrics_by_model:
        base = raw_metrics_by_model[baseline_name]
        for model_name, metrics in raw_metrics_by_model.items():
            if model_name == baseline_name:
                continue
            pairwise_vs_baseline[model_name] = {
                "sim_valid_diff": _bootstrap_diff_ci(
                    metrics["sim_valid"], base["sim_valid"], args.bootstrap_iters, args.ci, rng
                ),
                "reward_diff": _bootstrap_diff_ci(
                    metrics["reward"], base["reward"], args.bootstrap_iters, args.ci, rng
                ),
                "struct_diff": _bootstrap_diff_ci(
                    metrics["struct"], base["struct"], args.bootstrap_iters, args.ci, rng
                ),
            }

    summary.sort(key=lambda x: x["name"])
    payload = {
        "n_samples": args.n_samples,
        "seeds": args.seeds,
        "bootstrap": {
            "iters": args.bootstrap_iters,
            "ci": args.ci,
            "seed": args.bootstrap_seed,
        },
        "per_seed_outputs": [str(p) for p in per_seed_files],
        "summary": summary,
        "pairwise_vs_baseline": pairwise_vs_baseline,
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print("\n--- Multiseed Summary ---")
    for row in summary:
        print(
            f"{row['name']}: "
            f"sim_valid={row['sim_valid_mean']:.1%}±{row['sim_valid_std']:.1%}, "
            f"reward={row['reward_mean']:.3f}±{row['reward_std']:.3f}, "
            f"struct={row['struct_mean']:.1%}±{row['struct_std']:.1%}"
        )

    if pairwise_vs_baseline:
        print("\n--- Pairwise vs Baseline (bootstrap) ---")
        for model_name, diffs in pairwise_vs_baseline.items():
            sv = diffs["sim_valid_diff"]
            rw = diffs["reward_diff"]
            print(
                f"{model_name}: "
                f"sim_valid Δ={sv['mean_diff']:.1%} "
                f"[{sv['low']:.1%}, {sv['high']:.1%}], p(Δ>0)={sv['p_gt_zero']:.3f}; "
                f"reward Δ={rw['mean_diff']:.3f} "
                f"[{rw['low']:.3f}, {rw['high']:.3f}], p(Δ>0)={rw['p_gt_zero']:.3f}"
            )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
