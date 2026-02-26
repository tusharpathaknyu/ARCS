"""ARCS Phase 6: Compare all 3 model architectures.

Evaluates baseline, two_head, and graph_transformer on the same 160 conditioned
specs (10 per topology × 16 topologies) with SPICE simulation.

Prerequisites:
  - checkpoints/arcs_combined/best_model.pt       (baseline, from Phase 4)
  - checkpoints/arcs_two_head/best_model.pt        (from Phase 6 training)
  - checkpoints/arcs_graph_transformer/best_model.pt (from Phase 6 training)

Usage:
    PYTHONPATH=src python scripts/compare_architectures.py \
        [--n-samples 160] [--output results/arch_comparison.json] [-v]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from arcs.model_enhanced import load_model
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import generate_and_evaluate


MODEL_CONFIGS = [
    {
        "name": "Baseline GPT",
        "checkpoint": "checkpoints/arcs_combined/best_model.pt",
        "description": "6.5M param baseline (SwiGLU + RMSNorm, weight-tied LM head)",
    },
    {
        "name": "Two-Head",
        "checkpoint": "checkpoints/arcs_two_head/best_model.pt",
        "description": "6.8M param, separate structure head (weight-tied) + value head (SiLU MLP)",
    },
    {
        "name": "Graph Transformer",
        "checkpoint": "checkpoints/arcs_graph_transformer/best_model.pt",
        "description": "6.8M param, topology-aware causal attention + two-head output",
    },
]


def run_model_eval(
    config: dict,
    n_samples: int,
    device: torch.device,
    tokenizer: CircuitTokenizer,
    verbose: bool = False,
) -> dict | None:
    """Evaluate a single model checkpoint. Returns None if checkpoint missing."""
    name = config["name"]
    ckpt_path = config["checkpoint"]

    if not Path(ckpt_path).exists():
        print(f"\n  SKIP: {name} — checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model, model_config, model_type = load_model(ckpt_path, device=device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"  Type: {model_type}, Epoch: {epoch}, Val loss: {val_loss}")
    print(f"  Params: {model.count_parameters():,}")

    t0 = time.time()
    results = generate_and_evaluate(
        model,
        tokenizer,
        device,
        n_samples=n_samples,
        temperature=0.8,
        top_k=50,
        conditioned=True,
        simulate=True,
    )
    wall_time = time.time() - t0

    summary = {
        "name": name,
        "model_type": model_type,
        "description": config["description"],
        "checkpoint": ckpt_path,
        "epoch": epoch,
        "val_loss": val_loss if isinstance(val_loss, float) else None,
        "n_params": model.count_parameters(),
        "n_samples": n_samples,
        "wall_time": round(wall_time, 1),
        "struct_valid_rate": results.validity_rate,
        "sim_success_rate": results.sim_success_rate,
        "sim_valid_rate": results.sim_valid_rate,
        "avg_reward": results.avg_reward,
        "per_topology": {},
    }

    # Per-topology breakdown
    for topo, topo_results in results.per_topology.items():
        n = len(topo_results)
        if n == 0:
            continue
        sim_ok = sum(1 for r in topo_results if r.sim_outcome and r.sim_outcome.success)
        sim_valid = sum(1 for r in topo_results if r.sim_outcome and r.sim_outcome.valid)
        rewards = [r.reward for r in topo_results if r.reward is not None]
        summary["per_topology"][topo] = {
            "n": n,
            "sim_success": sim_ok,
            "sim_valid": sim_valid,
            "avg_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0,
        }

    # Print summary
    print(f"\n  Results ({n_samples} samples, {wall_time:.1f}s):")
    print(f"    Struct valid:  {results.validity_rate:.1%}")
    print(f"    Sim success:   {results.sim_success_rate:.1%}")
    print(f"    Sim valid:     {results.sim_valid_rate:.1%}")
    print(f"    Avg reward:    {results.avg_reward:.3f}/8.0")

    if verbose:
        print(f"\n  Per-topology:")
        for topo, d in sorted(summary["per_topology"].items()):
            print(f"    {topo:<25s} n={d['n']:>3d} sim_ok={d['sim_success']:>3d} "
                  f"sim_valid={d['sim_valid']:>3d} reward={d['avg_reward']:.3f}")

    return summary


def print_comparison_table(results: list[dict]) -> None:
    """Print markdown-style comparison table."""
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*80}\n")

    header = f"| {'Model':<20s} | {'Params':>8s} | {'Val Loss':>8s} | {'Struct':>7s} | {'Sim OK':>7s} | {'Sim Valid':>9s} | {'Reward':>7s} |"
    sep = f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*11}|{'-'*9}|"
    print(header)
    print(sep)

    for r in results:
        vl = f"{r['val_loss']:.4f}" if r.get("val_loss") else "?"
        print(
            f"| {r['name']:<20s} | {r['n_params']/1e6:>6.1f}M | {vl:>8s} | "
            f"{r['struct_valid_rate']:>6.1%} | {r['sim_success_rate']:>6.1%} | "
            f"{r['sim_valid_rate']:>8.1%} | {r['avg_reward']:>6.3f} |"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare ARCS model architectures")
    parser.add_argument("--n-samples", type=int, default=160, help="Samples per eval (default 160 = 10 per topo)")
    parser.add_argument("--output", type=str, default="results/arch_comparison.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-topology breakdown")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = CircuitTokenizer()
    results = []

    for config in MODEL_CONFIGS:
        result = run_model_eval(
            config, args.n_samples, device, tokenizer, verbose=args.verbose,
        )
        if result is not None:
            results.append(result)

    if len(results) >= 2:
        print_comparison_table(results)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
