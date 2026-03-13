#!/usr/bin/env python3
"""Evaluate topology-aware ARCS ablations on a fixed small budget.

Compares checkpoint variants such as:
  - baseline graph transformer
  - + topology-specific value heads
  - + topology-family MoE value routing

Usage:
  PYTHONPATH=src python scripts/evaluate_topology_ablation.py \
      --n-samples 48 --output results/topology_ablation.json

Override checkpoint paths if needed:
  --baseline-ckpt path/to/baseline.pt
  --topo-head-ckpt path/to/topology_heads.pt
  --family-moe-ckpt path/to/family_moe.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def evaluate_checkpoint(
    name: str,
    checkpoint: Path,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_samples: int,
    temperature: float,
    top_k: int,
) -> dict | None:
    if not checkpoint.exists():
        print(f"SKIP {name}: missing checkpoint {checkpoint}")
        return None

    print(f"\n=== {name} ===")
    print(f"checkpoint: {checkpoint}")

    model, config, model_type = load_model(str(checkpoint), device=device)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
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
    dt = time.time() - t0

    out = {
        "name": name,
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "n_params": n_params,
        "config": {
            "use_topology_value_heads": bool(getattr(config, "use_topology_value_heads", False)),
            "topology_value_head_alpha": float(getattr(config, "topology_value_head_alpha", 0.0)),
            "use_topology_family_moe": bool(getattr(config, "use_topology_family_moe", False)),
            "topology_family_moe_alpha": float(getattr(config, "topology_family_moe_alpha", 0.0)),
        },
        "n_samples": n_samples,
        "eval_time_s": round(dt, 1),
        "valid_structure_rate": results.validity_rate,
        "sim_success_rate": results.sim_success_rate,
        "sim_valid_rate": results.sim_valid_rate,
        "mean_reward": results.avg_reward,
        "mean_efficiency": results.avg_efficiency,
        "mean_vout_error": results.avg_vout_error,
    }

    print(
        f"struct={out['valid_structure_rate']:.1%} "
        f"sim_ok={out['sim_success_rate']:.1%} "
        f"sim_valid={out['sim_valid_rate']:.1%} "
        f"reward={out['mean_reward']:.3f} "
        f"time={out['eval_time_s']:.1f}s"
    )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate topology-aware model ablations")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/arcs_graph_transformer/best_model.pt")
    parser.add_argument("--topo-head-ckpt", type=str, default="checkpoints/arcs_graph_transformer_topo_value/best_model.pt")
    parser.add_argument("--family-moe-ckpt", type=str, default="checkpoints/arcs_graph_transformer_family_moe/best_model.pt")
    parser.add_argument("--n-samples", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/topology_ablation.json")
    args = parser.parse_args()

    device = _pick_device()
    print(f"device: {device}")

    tokenizer = CircuitTokenizer()
    variants = [
        ("GraphTransformer Baseline", Path(args.baseline_ckpt)),
        ("GraphTransformer + Topology Value Heads", Path(args.topo_head_ckpt)),
        ("GraphTransformer + Family MoE", Path(args.family_moe_ckpt)),
    ]

    rows = []
    for name, ckpt in variants:
        row = evaluate_checkpoint(
            name=name,
            checkpoint=ckpt,
            tokenizer=tokenizer,
            device=device,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No checkpoints found to evaluate.")

    print("\n--- Summary ---")
    for r in rows:
        print(
            f"{r['name']}: sim_valid={r['sim_valid_rate']:.1%}, "
            f"reward={r['mean_reward']:.3f}, struct={r['valid_structure_rate']:.1%}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
