#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pilot training multiseed outputs")
    parser.add_argument("--run-prefix", type=str, default="pilot2")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=str, default="results/topology_training_sweep_pilot2_multiseed.json")
    args = parser.parse_args()

    by_name: dict[str, list[dict]] = {}
    per_seed_files: list[str] = []

    for seed in args.seeds:
        path = Path("results") / f"topology_training_sweep_{args.run_prefix}_seed{seed}_eval.json"
        if not path.exists():
            raise SystemExit(f"Missing per-seed eval: {path}")
        per_seed_files.append(str(path))
        rows = json.loads(path.read_text())
        for row in rows:
            by_name.setdefault(row["name"], []).append(row)

    summary: list[dict] = []
    for name, rows in by_name.items():
        sim_valid = [float(r["sim_valid_rate"]) for r in rows]
        reward = [float(r["mean_reward"]) for r in rows]
        struct = [float(r["valid_structure_rate"]) for r in rows]
        summary.append(
            {
                "name": name,
                "n_runs": len(rows),
                "sim_valid_mean": _mean(sim_valid),
                "sim_valid_std": _std(sim_valid),
                "reward_mean": _mean(reward),
                "reward_std": _std(reward),
                "struct_mean": _mean(struct),
                "struct_std": _std(struct),
            }
        )

    summary.sort(key=lambda x: x["name"])
    baseline = next((row for row in summary if row["name"] == "GraphTransformer Baseline"), None)
    delta_vs_baseline: list[dict] = []
    if baseline is not None:
        for row in summary:
            if row["name"] == "GraphTransformer Baseline":
                continue
            delta_vs_baseline.append(
                {
                    "name": row["name"],
                    "sim_valid_delta": row["sim_valid_mean"] - baseline["sim_valid_mean"],
                    "reward_delta": row["reward_mean"] - baseline["reward_mean"],
                    "struct_delta": row["struct_mean"] - baseline["struct_mean"],
                }
            )

    payload = {
        "run_prefix": args.run_prefix,
        "seeds": args.seeds,
        "per_seed_files": per_seed_files,
        "summary": summary,
        "delta_vs_baseline": delta_vs_baseline,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("--- Pilot Multiseed Summary ---")
    for row in summary:
        print(
            f"{row['name']}: sim_valid={row['sim_valid_mean']:.1%}±{row['sim_valid_std']:.1%}, "
            f"reward={row['reward_mean']:.3f}±{row['reward_std']:.3f}, "
            f"struct={row['struct_mean']:.1%}±{row['struct_std']:.1%}"
        )

    if delta_vs_baseline:
        print("--- Delta vs Baseline ---")
        for row in delta_vs_baseline:
            print(
                f"{row['name']}: Δsim_valid={row['sim_valid_delta']:+.1%}, "
                f"Δreward={row['reward_delta']:+.3f}, Δstruct={row['struct_delta']:+.1%}"
            )

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
