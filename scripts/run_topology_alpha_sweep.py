#!/usr/bin/env python3
"""Sweep topology/family alpha values on existing checkpoints (no retraining).

This script evaluates:
- Baseline checkpoint
- Topology-head checkpoint for a list of topology alpha values
- Family-MoE checkpoint for a list of family alpha values

Across multiple seeds and reports mean/std, then highlights the best setting by
sim_valid (tie-breaker: reward).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from arcs.evaluate import generate_and_evaluate
from arcs.model_enhanced import load_model
from arcs.tokenizer import CircuitTokenizer


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return float(np.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1)))


def _evaluate(
    checkpoint: Path,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_samples: int,
    temperature: float,
    top_k: int,
    topology_alpha: float | None,
    family_alpha: float | None,
    enable_family: bool,
) -> dict:
    model, config, model_type = load_model(str(checkpoint), device=device)

    if topology_alpha is not None:
        setattr(config, "use_topology_value_heads", True)
        setattr(config, "topology_value_head_alpha", float(topology_alpha))
        setattr(model, "use_topology_value_heads", True)
        setattr(model, "topology_value_head_alpha", float(topology_alpha))

    setattr(config, "use_topology_family_moe", bool(enable_family))
    setattr(model, "use_topology_family_moe", bool(enable_family))

    if family_alpha is not None:
        setattr(config, "topology_family_moe_alpha", float(family_alpha))
        setattr(model, "topology_family_moe_alpha", float(family_alpha))

    results = generate_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_samples=n_samples,
        temperature=temperature,
        top_k=top_k,
        conditioned=True,
        simulate=True,
    )

    return {
        "model_type": model_type,
        "valid_structure_rate": float(results.validity_rate),
        "sim_success_rate": float(results.sim_success_rate),
        "sim_valid_rate": float(results.sim_valid_rate),
        "mean_reward": float(results.avg_reward),
        "mean_efficiency": float(results.avg_efficiency),
        "mean_vout_error": float(results.avg_vout_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha sweep for topology/family routing")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/arcs_graph_transformer/best_model.pt")
    parser.add_argument("--topo-head-ckpt", type=str, default="checkpoints/arcs_graph_transformer_topo_value_medium/best_model.pt")
    parser.add_argument("--family-moe-ckpt", type=str, default="checkpoints/arcs_graph_transformer_family_moe_medium/best_model.pt")
    parser.add_argument("--topology-alphas", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--family-alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--family-topology-alpha", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/topology_alpha_sweep.json")
    args = parser.parse_args()

    device = _pick_device()
    tokenizer = CircuitTokenizer()

    baseline_ckpt = Path(args.baseline_ckpt)
    topo_ckpt = Path(args.topo_head_ckpt)
    family_ckpt = Path(args.family_moe_ckpt)

    if not baseline_ckpt.exists():
        raise SystemExit(f"Missing baseline checkpoint: {baseline_ckpt}")
    if not topo_ckpt.exists():
        raise SystemExit(f"Missing topology-head checkpoint: {topo_ckpt}")
    if not family_ckpt.exists():
        raise SystemExit(f"Missing family-MoE checkpoint: {family_ckpt}")

    settings: list[dict] = [
        {
            "name": "baseline",
            "checkpoint": str(baseline_ckpt),
            "topology_alpha": None,
            "family_alpha": None,
            "enable_family": False,
            "label": "Baseline",
        }
    ]
    for alpha in args.topology_alphas:
        settings.append(
            {
                "name": f"topo_alpha_{alpha:.2f}",
                "checkpoint": str(topo_ckpt),
                "topology_alpha": float(alpha),
                "family_alpha": None,
                "enable_family": False,
                "label": f"TopologyHeads α={alpha:.2f}",
            }
        )
    for alpha in args.family_alphas:
        settings.append(
            {
                "name": f"family_alpha_{alpha:.2f}",
                "checkpoint": str(family_ckpt),
                "topology_alpha": float(args.family_topology_alpha),
                "family_alpha": float(alpha),
                "enable_family": True,
                "label": f"FamilyMoE αf={alpha:.2f}, αt={args.family_topology_alpha:.2f}",
            }
        )

    per_setting: dict[str, dict] = {
        s["name"]: {
            **s,
            "per_seed": [],
        }
        for s in settings
    }

    print(f"device: {device}")
    print(f"seeds: {args.seeds}")
    print(f"n_samples: {args.n_samples}")

    for seed in args.seeds:
        _set_seed(seed)
        print(f"\n=== Seed {seed} ===")
        for s in settings:
            print(f"- {s['label']}")
            metrics = _evaluate(
                checkpoint=Path(s["checkpoint"]),
                tokenizer=tokenizer,
                device=device,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                topology_alpha=s["topology_alpha"],
                family_alpha=s["family_alpha"],
                enable_family=s["enable_family"],
            )
            per_setting[s["name"]]["per_seed"].append({"seed": seed, **metrics})
            print(
                f"  struct={metrics['valid_structure_rate']:.1%} "
                f"sim_valid={metrics['sim_valid_rate']:.1%} "
                f"reward={metrics['mean_reward']:.3f}"
            )

    summary: list[dict] = []
    for s in settings:
        rows = per_setting[s["name"]]["per_seed"]
        sim_valid = [r["sim_valid_rate"] for r in rows]
        struct = [r["valid_structure_rate"] for r in rows]
        reward = [r["mean_reward"] for r in rows]
        summary.append(
            {
                "name": s["name"],
                "label": s["label"],
                "checkpoint": s["checkpoint"],
                "topology_alpha": s["topology_alpha"],
                "family_alpha": s["family_alpha"],
                "enable_family": s["enable_family"],
                "n_runs": len(rows),
                "sim_valid_mean": _mean(sim_valid),
                "sim_valid_std": _std(sim_valid),
                "struct_mean": _mean(struct),
                "struct_std": _std(struct),
                "reward_mean": _mean(reward),
                "reward_std": _std(reward),
            }
        )

    summary.sort(key=lambda x: (x["sim_valid_mean"], x["reward_mean"]), reverse=True)
    best = summary[0]

    payload = {
        "seeds": args.seeds,
        "n_samples": args.n_samples,
        "topology_alphas": args.topology_alphas,
        "family_alphas": args.family_alphas,
        "family_topology_alpha": args.family_topology_alpha,
        "settings": per_setting,
        "summary": summary,
        "best": best,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print("\n--- Sweep Summary (sorted) ---")
    for row in summary:
        print(
            f"{row['label']}: sim_valid={row['sim_valid_mean']:.1%}±{row['sim_valid_std']:.1%}, "
            f"reward={row['reward_mean']:.3f}±{row['reward_std']:.3f}, "
            f"struct={row['struct_mean']:.1%}±{row['struct_std']:.1%}"
        )
    print(f"\nBest: {best['label']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
