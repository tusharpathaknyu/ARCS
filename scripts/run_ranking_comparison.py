#!/usr/bin/env python3
"""Generate ranking comparison data for both SL and RL generators.

Produces results/ranking_comparison.json with both SL and RL data,
as needed for the paper's Table 9.
"""

import json
import time
from pathlib import Path

import torch

from arcs.bestofn import BestOfNGenerator, rank_candidates
from arcs.constrained import ConstraintLevel
from arcs.model import ARCSConfig, ARCSModel
from arcs.reward_model import RewardModelRanker, RewardModelTrainer
from arcs.simulate import ALL_TEST_SPECS, compute_reward, simulate_decoded_circuit
from arcs.tokenizer import CircuitTokenizer


def run_ranking_for_generator(
    gen_model,
    reward_model,
    tokenizer,
    device,
    n_values=(1, 3, 5, 10, 20),
    n_specs=16,
    n_seeds=5,
    label="SL",
):
    """Run ranking comparison for a single generator model, averaged over seeds."""
    ranker = RewardModelRanker(reward_model, tokenizer, device=device)
    results = {}

    for n in n_values:
        print(f"\n  [{label}] N = {n}")
        all_conf_rewards = []
        all_rm_rewards = []

        for seed in range(n_seeds):
            torch.manual_seed(42 + seed * 1000 + n)
            test_specs = ALL_TEST_SPECS[:n_specs]
            conf_rewards = []
            rm_rewards = []

            for topo, specs in test_specs:
                gen = BestOfNGenerator(
                    gen_model, tokenizer,
                    constraint_level=ConstraintLevel.TOPOLOGY,
                    ranking_method="confidence",
                )
                result = gen.generate_n(
                    topo, specs, n=n, temperature=0.8, top_k=50,
                    device=device,
                )
                candidates = result.candidates

                # Rank by confidence
                conf_ranked = rank_candidates(candidates, method="confidence")
                conf_best = conf_ranked[0]

                # Rank by reward model
                rm_ranked = rank_candidates(
                    candidates, method="reward_model", reward_ranker=ranker
                )
                rm_best = rm_ranked[0]

                # SPICE simulation for ground truth
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

            all_conf_rewards.extend(conf_rewards)
            all_rm_rewards.extend(rm_rewards)

        conf_avg = sum(all_conf_rewards) / len(all_conf_rewards) if all_conf_rewards else 0
        rm_avg = sum(all_rm_rewards) / len(all_rm_rewards) if all_rm_rewards else 0
        improvement = ((rm_avg - conf_avg) / max(conf_avg, 0.01)) * 100

        results[str(n)] = {
            "n": n,
            "conf_reward": conf_avg,
            "rm_reward": rm_avg,
            "improvement_pct": improvement,
            "n_samples": len(all_conf_rewards),
        }
        print(f"    Conf: {conf_avg:.3f}, RM: {rm_avg:.3f}, Δ: {improvement:+.1f}% "
              f"(over {len(all_conf_rewards)} trials)")

    return results


def main():
    device = torch.device("cpu")
    tokenizer = CircuitTokenizer()

    # Average over 5 random seeds × 16 test specs = 80 trials per N value
    N_SEEDS = 5

    # Load reward model
    print("Loading reward model...")
    _, reward_model = RewardModelTrainer.load_checkpoint(
        "checkpoints/reward_model/best_reward_model.pt", device
    )
    print(f"  Reward model: {reward_model.count_parameters():,} params")

    # === SL Generator ===
    print("\n" + "=" * 60)
    print("  SL Generator — Ranking Comparison")
    print("=" * 60)
    sl_ckpt = torch.load(
        "checkpoints/arcs_combined/best_model.pt",
        map_location=device, weights_only=False,
    )
    sl_config = ARCSConfig.from_dict(sl_ckpt["config"])
    sl_model = ARCSModel(sl_config)
    sl_model.load_state_dict(sl_ckpt["model_state_dict"])
    sl_model.to(device).eval()
    print(f"  SL model: {sl_model.count_parameters():,} params")

    sl_results = run_ranking_for_generator(
        sl_model, reward_model, tokenizer, device, label="SL", n_seeds=N_SEEDS,
    )

    # === RL Generator ===
    print("\n" + "=" * 60)
    print("  RL Generator — Ranking Comparison")
    print("=" * 60)
    rl_ckpt = torch.load(
        "checkpoints/arcs_rl_v2/best_rl_model.pt",
        map_location=device, weights_only=False,
    )
    rl_config = ARCSConfig.from_dict(rl_ckpt["config"])
    rl_model = ARCSModel(rl_config)
    rl_model.load_state_dict(rl_ckpt["model_state_dict"])
    rl_model.to(device).eval()
    print(f"  RL model: {rl_model.count_parameters():,} params")

    rl_results = run_ranking_for_generator(
        rl_model, reward_model, tokenizer, device, label="RL", n_seeds=N_SEEDS,
    )

    # === Combined output ===
    combined = {
        "sl": sl_results,
        "rl": rl_results,
    }

    out_path = Path("results/ranking_comparison.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Confidence vs Reward Model Ranking")
    print(f"{'=' * 70}")
    print(f"{'N':>4} | {'SL Conf':>8} {'SL RM':>8} {'Δ%':>7} | "
          f"{'RL Conf':>8} {'RL RM':>8} {'Δ%':>7}")
    print("-" * 70)
    for n_str in ["1", "3", "5", "10", "20"]:
        sl = sl_results[n_str]
        rl = rl_results[n_str]
        print(
            f"{n_str:>4} | "
            f"{sl['conf_reward']:8.3f} {sl['rm_reward']:8.3f} {sl['improvement_pct']:+6.1f}% | "
            f"{rl['conf_reward']:8.3f} {rl['rm_reward']:8.3f} {rl['improvement_pct']:+6.1f}%"
        )


if __name__ == "__main__":
    main()
