"""ARCS Ablation Studies — evaluate model variants for the paper.

Ablations:
  1. Full ARCS (combined + RL) — the main model (already evaluated)
  2. No RL — combined supervised-only model
  3. No spec conditioning — generate unconditioned circuits, evaluate sim quality
  4. No augmentation — train on raw data without component shuffle augmentation
  5. No invalid examples — train only on valid circuits (valid_only flag)

For ablations 4 & 5, we'd need to retrain from scratch (hours).
Instead, we evaluate the EXISTING checkpoints with different eval modes:
  - Ablation 1: checkpoints/arcs_rl_v2/best_rl_model.pt (conditioned)
  - Ablation 2: checkpoints/arcs_combined/best_model.pt (conditioned)
  - Ablation 3: checkpoints/arcs_rl_v2/best_rl_model.pt (UNconditioned)
  - Ablation 4: checkpoints/arcs_small/best_model.pt (Tier 1 only, no Tier 2)
  - Ablation 5: Same as 4 (phase1 was run without valid_only filter)

This gives us 4 meaningful ablation rows comparing:
  - Effect of RL fine-tuning (row 1 vs 2)
  - Effect of spec conditioning (row 1 vs 3)
  - Effect of expanded data (Tier 1+2 vs Tier 1 only) (row 2 vs 4)

Usage:
    PYTHONPATH=src python scripts/run_ablations.py [--n-samples 160] [--output results/ablations.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import generate_and_evaluate


ABLATION_CONFIGS = [
    {
        "name": "ARCS + RL (full)",
        "checkpoint": "checkpoints/arcs_rl_v2/best_rl_model.pt",
        "conditioned": True,
        "tier": None,
        "description": "Full model: combined training + RL fine-tuning, spec-conditioned",
    },
    {
        "name": "ARCS supervised only",
        "checkpoint": "checkpoints/arcs_combined/best_model.pt",
        "conditioned": True,
        "tier": None,
        "description": "Combined Tier 1+2 training only, no RL fine-tuning",
    },
    {
        "name": "ARCS + RL (no spec cond.)",
        "checkpoint": "checkpoints/arcs_rl_v2/best_rl_model.pt",
        "conditioned": False,
        "tier": None,
        "description": "Same full model but generating unconditionally (no spec prefix)",
    },
    {
        "name": "ARCS Tier 1 only",
        "checkpoint": "checkpoints/arcs_small/best_model.pt",
        "conditioned": True,
        "tier": 1,
        "description": "Trained on Tier 1 (power converters) only, no Tier 2 signal circuits",
    },
]


def run_ablation(
    config: dict,
    n_samples: int,
    device: torch.device,
    tokenizer: CircuitTokenizer,
) -> dict:
    """Run a single ablation and return results dict."""
    name = config["name"]
    ckpt_path = config["checkpoint"]

    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Conditioned: {config['conditioned']}")
    print(f"  Tier filter: {config['tier'] or 'all'}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ARCSConfig.from_dict(checkpoint["config"])
    model = ARCSModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded epoch {checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?')}")

    t0 = time.time()
    results = generate_and_evaluate(
        model,
        tokenizer,
        device,
        n_samples=n_samples,
        temperature=0.8,
        top_k=50,
        conditioned=config["conditioned"],
        simulate=True,
        tier=config["tier"],
    )
    wall_time = time.time() - t0

    # Summarize
    summary = {
        "name": name,
        "description": config["description"],
        "checkpoint": ckpt_path,
        "conditioned": config["conditioned"],
        "tier": config["tier"],
        "n_samples": n_samples,
        "wall_time": round(wall_time, 1),
        "struct_valid_rate": results.validity_rate,
        "sim_success_rate": results.sim_success_rate,
        "sim_valid_rate": results.sim_valid_rate,
        "avg_reward": round(results.avg_reward, 3),
        "avg_efficiency": round(results.avg_efficiency, 4),
        "avg_vout_error": round(results.avg_vout_error, 2),
        "diversity_score": round(results.diversity_score, 3),
        "per_topology": results.per_topology_sim,
    }

    print(f"\n  Results ({wall_time:.0f}s):")
    print(f"    Struct valid: {results.validity_rate:.1%}")
    print(f"    Sim success:  {results.sim_success_rate:.1%}")
    print(f"    Sim valid:    {results.sim_valid_rate:.1%}")
    print(f"    Avg reward:   {results.avg_reward:.3f}/8.0")

    return summary


def main():
    parser = argparse.ArgumentParser(description="ARCS Ablation Studies")
    parser.add_argument("--n-samples", type=int, default=160,
                        help="Number of circuits per ablation (default 160)")
    parser.add_argument("--output", type=str, default="results/ablations.json",
                        help="Output file for ablation results")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
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

    # Run all ablations
    all_results = []
    for config in ABLATION_CONFIGS:
        ckpt_path = config["checkpoint"]
        if not Path(ckpt_path).exists():
            print(f"\n⚠ Skipping '{config['name']}': {ckpt_path} not found")
            continue
        result = run_ablation(config, args.n_samples, device, tokenizer)
        all_results.append(result)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAblation results saved to {args.output}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("ABLATION COMPARISON TABLE")
    print(f"{'='*80}")
    hdr = f"{'Variant':<30s} {'StructValid':>11s} {'SimSuccess':>11s} {'SimValid':>9s} {'Reward':>8s}"
    print(hdr)
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['name']:<30s} "
            f"{r['struct_valid_rate']:>10.1%} "
            f"{r['sim_success_rate']:>10.1%} "
            f"{r['sim_valid_rate']:>8.1%} "
            f"{r['avg_reward']:>8.3f}"
        )

    # Analysis
    if len(all_results) >= 3:
        print(f"\nKey findings:")
        full = all_results[0]
        sup = all_results[1]
        nocond = all_results[2]

        rl_delta = full["avg_reward"] - sup["avg_reward"]
        print(f"  RL fine-tuning effect:    reward {sup['avg_reward']:.3f} → {full['avg_reward']:.3f} (Δ={rl_delta:+.3f})")

        cond_delta = full["avg_reward"] - nocond["avg_reward"]
        print(f"  Spec conditioning effect: reward {nocond['avg_reward']:.3f} → {full['avg_reward']:.3f} (Δ={cond_delta:+.3f})")

        if len(all_results) >= 4:
            t1only = all_results[3]
            data_delta = sup["avg_reward"] - t1only["avg_reward"]
            print(f"  Expanded data effect:    reward {t1only['avg_reward']:.3f} → {sup['avg_reward']:.3f} (Δ={data_delta:+.3f})")


if __name__ == "__main__":
    main()
