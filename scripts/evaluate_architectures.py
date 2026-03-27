#!/usr/bin/env python3
"""Unified architecture comparison: evaluate ALL autoregressive variants.

Evaluates all models on the SAME protocol (n_samples per topology, same test
specs, same seed) to produce a single authoritative results file for the paper.

Usage:
    PYTHONPATH=src python scripts/evaluate_architectures.py \
        --n-samples 10 --seed 42 --output results/arch_comparison_v2.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from arcs.model_enhanced import load_model, create_model, ARCSConfig
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import generate_and_evaluate
from arcs import DEFAULT_TEMPERATURE, DEFAULT_TOP_K


# All checkpoints to evaluate
MODELS = [
    {
        "name": "Baseline GPT",
        "checkpoint": "checkpoints/arcs_combined/best_model.pt",
        "model_type": "baseline",
    },
    {
        "name": "Two-Head",
        "checkpoint": "checkpoints/arcs_two_head/best_model.pt",
        "model_type": "two_head",
    },
    {
        "name": "Graph Transformer SL",
        "checkpoint": "checkpoints/arcs_graph_transformer/best_model.pt",
        "model_type": "graph_transformer",
    },
    {
        "name": "GT + REINFORCE (5000)",
        "checkpoint": "checkpoints/arcs_rl_v2/best_rl_model.pt",
        "model_type": "graph_transformer",
    },
    {
        "name": "GT + GRPO (500)",
        "checkpoint": "checkpoints/arcs_grpo/best_rl_model.pt",
        "model_type": "graph_transformer",
    },
    {
        "name": "GT + GRPO (3500)",
        "checkpoint": "checkpoints/arcs_grpo_extended/best_rl_model.pt",
        "model_type": "graph_transformer",
    },
]


def load_arcs_model(checkpoint: str, device: torch.device):
    """Load an ARCS autoregressive model, handling both SL and RL formats."""
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return None, None, None

    try:
        model, config, mt = load_model(checkpoint, device=device)
        return model, config, mt
    except RuntimeError:
        pass

    # RL checkpoint format
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        return None, None, None

    config = ARCSConfig.from_dict(ckpt["config"])
    mt = ckpt.get("model_type", "graph_transformer")
    model = create_model(mt, config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config, mt


def evaluate_model(
    entry: dict,
    n_samples: int,
    device: torch.device,
    tokenizer: CircuitTokenizer,
) -> dict | None:
    """Evaluate a single autoregressive model."""
    name = entry["name"]
    checkpoint = entry["checkpoint"]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*60}")

    model, config, mt = load_arcs_model(checkpoint, device)
    if model is None:
        print(f"  SKIP: checkpoint not found or failed to load")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    results = generate_and_evaluate(
        model, tokenizer, device,
        n_samples=n_samples,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        conditioned=True,
        simulate=True,
    )
    dt = time.time() - t0

    out = {
        "name": name,
        "checkpoint": checkpoint,
        "model_type": entry["model_type"],
        "n_params": n_params,
        "n_samples": n_samples,
        "eval_time_s": round(dt, 1),
        "struct_valid_rate": results.validity_rate,
        "sim_success_rate": results.sim_success_rate,
        "sim_valid_rate": results.sim_valid_rate,
        "avg_reward": results.avg_reward,
        "avg_efficiency": results.avg_efficiency,
        "avg_vout_error": results.avg_vout_error,
        "per_topology": {},
    }

    # Extract per-topology breakdown
    if hasattr(results, "per_topology_sim") and results.per_topology_sim:
        out["per_topology"] = results.per_topology_sim
    elif hasattr(results, "per_topology") and results.per_topology:
        out["per_topology"] = results.per_topology

    print(f"  Results:")
    print(f"    Structure valid: {out['struct_valid_rate']:.1%}")
    print(f"    Sim success:     {out['sim_success_rate']:.1%}")
    print(f"    Sim valid:       {out['sim_valid_rate']:.1%}")
    print(f"    Mean reward:     {out['avg_reward']:.3f}")
    print(f"    Time:            {dt:.1f}s")

    return out


def main():
    parser = argparse.ArgumentParser(description="Unified architecture comparison")
    parser.add_argument("--n-samples", type=int, default=160,
                        help="Total samples (10 per topology × 16 topologies)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/arch_comparison_v2.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = CircuitTokenizer()

    results = []
    for entry in MODELS:
        result = evaluate_model(entry, args.n_samples, device, tokenizer)
        if result is not None:
            results.append(result)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  UNIFIED ARCHITECTURE COMPARISON ({args.n_samples} samples, seed={args.seed})")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'Params':>8} {'Struct':>8} {'SimValid':>10} {'Reward':>8}")
    print("-" * 70)
    for r in results:
        params_m = r["n_params"] / 1e6
        print(
            f"{r['name']:<30} {params_m:>7.2f}M "
            f"{r['struct_valid_rate']:>7.1%} "
            f"{r['sim_valid_rate']:>9.1%} "
            f"{r['avg_reward']:>7.2f}"
        )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
