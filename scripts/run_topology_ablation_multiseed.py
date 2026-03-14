#!/usr/bin/env python3
"""Run topology ablation evaluation across multiple seeds and summarize mean/std.

Uses existing checkpoints (no training) and calls scripts/evaluate_topology_ablation.py
for each seed. Writes per-seed JSON plus aggregate JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topology ablation eval for multiple seeds")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/arcs_graph_transformer/best_model.pt")
    parser.add_argument("--topo-head-ckpt", type=str, default="checkpoints/arcs_graph_transformer_topo_value_medium/best_model.pt")
    parser.add_argument("--family-moe-ckpt", type=str, default="checkpoints/arcs_graph_transformer_family_moe_medium/best_model.pt")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--output", type=str, default="results/topology_ablation_medium_multiseed.json")
    parser.add_argument("--python", type=str, default=".venv/bin/python")
    args = parser.parse_args()

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
    for model_name, rows in rows_by_model.items():
        sim_valid = [float(r["sim_valid_rate"]) for r in rows]
        struct_valid = [float(r["valid_structure_rate"]) for r in rows]
        reward = [float(r["mean_reward"]) for r in rows]
        summary.append(
            {
                "name": model_name,
                "n_runs": len(rows),
                "seeds": [int(r.get("seed", -1)) for r in rows],
                "sim_valid_mean": _mean(sim_valid),
                "sim_valid_std": _std(sim_valid),
                "struct_mean": _mean(struct_valid),
                "struct_std": _std(struct_valid),
                "reward_mean": _mean(reward),
                "reward_std": _std(reward),
            }
        )

    summary.sort(key=lambda x: x["name"])
    payload = {
        "n_samples": args.n_samples,
        "seeds": args.seeds,
        "per_seed_outputs": [str(p) for p in per_seed_files],
        "summary": summary,
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

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
