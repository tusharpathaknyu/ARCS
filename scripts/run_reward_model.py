#!/usr/bin/env python3
"""Train a learned reward model and compare ranking strategies for Best-of-N.

Usage:
    # Train reward model on all data
    python scripts/run_reward_model.py --train

    # Train with embedding transfer from generator
    python scripts/run_reward_model.py --train --generator-checkpoint checkpoints/arcs_combined/best_model.pt

    # Evaluate trained model
    python scripts/run_reward_model.py --evaluate --reward-checkpoint checkpoints/reward_model/best_reward_model.pt

    # Compare ranking methods (confidence vs reward_model) with SPICE
    python scripts/run_reward_model.py --compare --simulate \
        --reward-checkpoint checkpoints/reward_model/best_reward_model.pt \
        --generator-checkpoint checkpoints/arcs_rl_v2/best_rl_model.pt

    # Full pipeline: train + compare
    python scripts/run_reward_model.py --train --compare --simulate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from arcs.bestofn import (
    BestOfNGenerator,
    rank_candidates,
    run_scaling_experiment,
)
from arcs.constrained import ConstraintLevel
from arcs.evaluate import decode_generated_sequence
from arcs.model import ARCSConfig, ARCSModel
from arcs.reward_model import (
    CircuitRewardDataset,
    CircuitRewardModel,
    RewardModelConfig,
    RewardModelRanker,
    RewardModelTrainer,
    evaluate_reward_model,
)
from arcs.simulate import (
    ALL_TEST_SPECS,
    TIER1_TEST_SPECS,
    TIER2_TEST_SPECS,
    compute_reward,
    simulate_decoded_circuit,
)
from arcs.tokenizer import CircuitTokenizer


def train_reward_model(
    data_dir: str = "data/combined",
    generator_checkpoint: str | None = None,
    config_size: str = "small",
    output_dir: str = "checkpoints/reward_model",
    device: str = "cpu",
) -> tuple[CircuitRewardModel, dict]:
    """Train the reward model on SPICE simulation data."""
    print("\n" + "=" * 60)
    print("  ARCS Learned Reward Model — Training")
    print("=" * 60)

    tokenizer = CircuitTokenizer()

    # Config
    configs = {
        "tiny": RewardModelConfig.tiny,
        "small": RewardModelConfig.small,
        "medium": RewardModelConfig.medium,
    }
    config = configs.get(config_size, RewardModelConfig.small)()
    print(f"\n  Config: {config_size}")
    print(f"  d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}")

    # Load data
    print(f"\n  Loading data from {data_dir}...")
    t0 = time.perf_counter()
    dataset = CircuitRewardDataset(data_dir, tokenizer, max_seq_len=config.max_seq_len)
    dt = time.perf_counter() - t0
    print(f"  Loaded {len(dataset):,} samples in {dt:.1f}s")

    stats = dataset.reward_stats
    print(f"  Reward stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
          f"min={stats['min']:.2f}, max={stats['max']:.2f}")
    print(f"  Zero-reward fraction: {stats['frac_zero']:.1%}")

    # Train
    print(f"\n  Training on device={device}...")
    gen_ckpt = generator_checkpoint if generator_checkpoint else None
    trainer = RewardModelTrainer(config, device=device, generator_checkpoint=gen_ckpt)
    model = trainer.train(dataset, verbose=True)

    # Save
    output_path = Path(output_dir) / "best_reward_model.pt"
    trainer.save_checkpoint(output_path)
    print(f"\n  Saved to {output_path}")

    # Final eval
    print("\n  Final evaluation on full dataset:")
    metrics = evaluate_reward_model(model, dataset, device=device)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    return model, metrics


def evaluate_model(
    reward_checkpoint: str,
    data_dir: str = "data/combined",
    device: str = "cpu",
) -> dict:
    """Evaluate a trained reward model."""
    print("\n" + "=" * 60)
    print("  ARCS Reward Model — Evaluation")
    print("=" * 60)

    tokenizer = CircuitTokenizer()

    # Load model
    trainer, model = RewardModelTrainer.load_checkpoint(reward_checkpoint, device)
    print(f"  Loaded model: {model.count_parameters():,} params")
    print(f"  Training history: {len(trainer.history)} epochs")

    # Load data
    dataset = CircuitRewardDataset(
        data_dir, tokenizer, max_seq_len=model.config.max_seq_len
    )
    print(f"  Dataset: {len(dataset):,} samples")

    metrics = evaluate_reward_model(model, dataset, device=device)
    print("\n  Evaluation Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    return metrics


def compare_ranking_methods(
    generator_checkpoint: str,
    reward_checkpoint: str,
    simulate: bool = False,
    n_values: list[int] | None = None,
    n_specs: int = 16,
    device: str = "cpu",
) -> dict:
    """Compare confidence-based vs reward-model-based ranking.

    For each N value, generate candidates and rank them two ways:
    1. By model confidence (log-prob) — free, no oracle
    2. By learned reward model — cheap oracle proxy (~1ms per batch)

    If --simulate, also get ground truth SPICE rewards for each ranking.
    """
    print("\n" + "=" * 60)
    print("  ARCS Ranking Method Comparison")
    print("=" * 60)

    if n_values is None:
        n_values = [1, 3, 5, 10, 20]

    tokenizer = CircuitTokenizer()

    # Load generator
    ckpt = torch.load(generator_checkpoint, map_location=device, weights_only=False)
    gen_config = ARCSConfig.from_dict(ckpt["config"])
    gen_model = ARCSModel(gen_config)
    gen_model.load_state_dict(ckpt["model_state_dict"])
    gen_model.to(device)
    gen_model.eval()
    print(f"  Generator: {gen_model.count_parameters():,} params")

    # Load reward model
    _, reward_model = RewardModelTrainer.load_checkpoint(reward_checkpoint, device)
    print(f"  Reward model: {reward_model.count_parameters():,} params")

    ranker = RewardModelRanker(reward_model, tokenizer, device=device)

    results = {}

    for n in n_values:
        print(f"\n{'─' * 50}")
        print(f"  N = {n}")
        print(f"{'─' * 50}")

        conf_rewards = []
        rm_rewards = []
        conf_valids = 0
        rm_valids = 0
        total = 0

        test_specs = ALL_TEST_SPECS[:n_specs]

        for topo, specs in test_specs:
            # Generate N candidates
            gen = BestOfNGenerator(
                gen_model, tokenizer,
                constraint_level=ConstraintLevel.TOPOLOGY,
                ranking_method="confidence",
            )
            result = gen.generate_n(
                topo, specs, n=n, temperature=0.8, top_k=50,
                device=torch.device(device),
            )
            candidates = result.candidates

            # Rank by confidence (already done by generate_n)
            conf_ranked = rank_candidates(candidates, method="confidence")
            conf_best = conf_ranked[0]

            # Rank by reward model
            rm_ranked = rank_candidates(
                candidates, method="reward_model", reward_ranker=ranker
            )
            rm_best = rm_ranked[0]

            total += 1

            if conf_best.decoded.valid_structure:
                conf_valids += 1
            if rm_best.decoded.valid_structure:
                rm_valids += 1

            # SPICE simulation for ground truth
            if simulate:
                for best, rewards_list in [
                    (conf_best, conf_rewards),
                    (rm_best, rm_rewards),
                ]:
                    if best.decoded.valid_structure:
                        try:
                            outcome = simulate_decoded_circuit(best.decoded)
                            r = compute_reward(best.decoded, outcome, specs)
                            rewards_list.append(r)
                        except Exception:
                            rewards_list.append(1.0)
                    else:
                        rewards_list.append(0.0)

        row = {
            "n": n,
            "conf_validity": conf_valids / max(total, 1),
            "rm_validity": rm_valids / max(total, 1),
        }

        if simulate and conf_rewards:
            conf_avg = sum(conf_rewards) / len(conf_rewards)
            rm_avg = sum(rm_rewards) / len(rm_rewards)
            improvement = ((rm_avg - conf_avg) / max(conf_avg, 0.01)) * 100

            row["conf_reward"] = conf_avg
            row["rm_reward"] = rm_avg
            row["improvement_pct"] = improvement

            print(f"  Confidence ranking: reward={conf_avg:.3f}, "
                  f"validity={row['conf_validity']:.1%}")
            print(f"  Reward model rank:  reward={rm_avg:.3f}, "
                  f"validity={row['rm_validity']:.1%}")
            print(f"  Improvement: {improvement:+.1f}%")
        else:
            print(f"  Confidence validity: {row['conf_validity']:.1%}")
            print(f"  Reward model validity: {row['rm_validity']:.1%}")

        results[n] = row

    # Summary table
    print(f"\n{'=' * 60}")
    print("  Summary: Confidence vs Reward Model Ranking")
    print(f"{'=' * 60}")
    if simulate:
        print(f"  {'N':>4} | {'Conf Reward':>12} | {'RM Reward':>10} | {'Δ%':>6}")
        print(f"  {'─' * 4}-+-{'─' * 12}-+-{'─' * 10}-+-{'─' * 6}")
        for n, row in results.items():
            print(
                f"  {n:4d} | {row.get('conf_reward', 0):12.3f} | "
                f"{row.get('rm_reward', 0):10.3f} | "
                f"{row.get('improvement_pct', 0):+5.1f}%"
            )
    else:
        print(f"  {'N':>4} | {'Conf Valid':>11} | {'RM Valid':>9}")
        print(f"  {'─' * 4}-+-{'─' * 11}-+-{'─' * 9}")
        for n, row in results.items():
            print(
                f"  {n:4d} | {row['conf_validity']:11.1%} | "
                f"{row['rm_validity']:9.1%}"
            )

    # Save results
    out_path = Path("results/ranking_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ARCS Learned Reward Model")
    parser.add_argument("--train", action="store_true", help="Train reward model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--compare", action="store_true", help="Compare ranking methods")
    parser.add_argument("--simulate", action="store_true", help="Run SPICE simulation")

    parser.add_argument("--data-dir", default="data/combined")
    parser.add_argument("--config-size", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--output-dir", default="checkpoints/reward_model")
    parser.add_argument("--reward-checkpoint", default="checkpoints/reward_model/best_reward_model.pt")
    parser.add_argument(
        "--generator-checkpoint",
        default="checkpoints/arcs_rl_v2/best_rl_model.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-specs", type=int, default=16)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
    )

    args = parser.parse_args()

    if args.train:
        model, metrics = train_reward_model(
            data_dir=args.data_dir,
            generator_checkpoint=args.generator_checkpoint,
            config_size=args.config_size,
            output_dir=args.output_dir,
            device=args.device,
        )
        # Update reward-checkpoint path for subsequent steps
        args.reward_checkpoint = str(Path(args.output_dir) / "best_reward_model.pt")

    if args.evaluate:
        evaluate_model(
            reward_checkpoint=args.reward_checkpoint,
            data_dir=args.data_dir,
            device=args.device,
        )

    if args.compare:
        compare_ranking_methods(
            generator_checkpoint=args.generator_checkpoint,
            reward_checkpoint=args.reward_checkpoint,
            simulate=args.simulate,
            n_values=args.n_values,
            n_specs=args.n_specs,
            device=args.device,
        )


if __name__ == "__main__":
    main()
