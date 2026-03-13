#!/usr/bin/env python3
"""Generate publication-quality Phase 14 figures (GRPO/CCFM/hybrid)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 250,
    "savefig.dpi": 300,
})


def parse_grpo_log(path: Path):
    steps, rewards, sim_valid, struct_valid = [], [], [], []
    eval_steps, eval_rewards = [], []

    if not path.exists():
        return {}, {}

    for line in path.read_text().splitlines():
        m = re.search(r"Step\s+(\d+)/\d+\s+\|\s+reward=([\d.]+)±[\d.]+\s+\|.*?valid=(\d+)%.*?struct=(\d+)%", line)
        if m:
            steps.append(int(m.group(1)))
            rewards.append(float(m.group(2)))
            sim_valid.append(int(m.group(3)))
            struct_valid.append(int(m.group(4)))
            continue

        m2 = re.search(r"Eval step\s+(\d+).*?eval_reward_mean.*?([\d.]+)", line)
        if m2:
            eval_steps.append(int(m2.group(1)))
            eval_rewards.append(float(m2.group(2)))

    train = {
        "step": np.array(steps),
        "reward": np.array(rewards),
        "sim_valid": np.array(sim_valid),
        "struct_valid": np.array(struct_valid),
    }
    evals = {
        "step": np.array(eval_steps),
        "reward": np.array(eval_rewards),
    }
    return train, evals


def smooth(y: np.ndarray, w: int = 9) -> np.ndarray:
    if y.size < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def plot_grpo_progress():
    base_train, base_eval = parse_grpo_log(ROOT / "logs" / "train_grpo.log")
    ext_train, ext_eval = parse_grpo_log(ROOT / "logs" / "train_grpo_extended.log")

    if not base_train:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    if base_train["step"].size:
        sm = smooth(base_train["reward"])
        ax.plot(base_train["step"], base_train["reward"], alpha=0.2, color="tab:blue")
        ax.plot(base_train["step"][len(base_train["step"]) - len(sm):], sm, color="tab:blue", label="GRPO 0-500")
    if ext_train and ext_train["step"].size:
        x = ext_train["step"] + 500
        sm = smooth(ext_train["reward"])
        ax.plot(x, ext_train["reward"], alpha=0.2, color="tab:orange")
        ax.plot(x[len(x) - len(sm):], sm, color="tab:orange", label="GRPO 500-3500")
    ax.set_title("(a) Training Reward")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()

    ax = axes[1]
    if base_train["step"].size:
        ax.plot(base_train["step"], base_train["sim_valid"], color="tab:green", label="Sim valid % (0-500)")
    if ext_train and ext_train["step"].size:
        ax.plot(ext_train["step"] + 500, ext_train["sim_valid"], color="tab:red", label="Sim valid % (500-3500)")
    ax.set_title("(b) Sim Validity During RL")
    ax.set_xlabel("Step")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    ax = axes[2]
    if base_eval.get("step", np.array([])).size:
        ax.plot(base_eval["step"], base_eval["reward"], marker="o", color="tab:blue", label="Eval reward (0-500)")
    if ext_eval.get("step", np.array([])).size:
        ax.plot(ext_eval["step"] + 500, ext_eval["reward"], marker="o", color="tab:orange", label="Eval reward (500-3500)")
    ax.axvline(500, linestyle="--", color="gray", alpha=0.6)
    ax.set_title("(c) Eval Reward Trajectory")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)

    fig.suptitle("GRPO Training Dynamics: Early-Stopping vs Extended Fine-Tuning", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase14_grpo_progress.png")
    fig.savefig(FIG_DIR / "phase14_grpo_progress.pdf")
    plt.close(fig)


def plot_model_comparison():
    phase14 = ROOT / "results" / "phase14_comparison.json"
    if not phase14.exists():
        return

    data = json.loads(phase14.read_text())
    models = data if isinstance(data, list) else data.get("models", [])

    labels, sim_valid, reward = [], [], []
    for m in models:
        if m.get("type") != "autoregressive":
            continue
        labels.append(m["name"].replace("ARCS-", ""))
        sim_valid.append(100.0 * m.get("sim_valid_rate", 0.0))
        reward.append(m.get("mean_reward", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(labels))
    axes[0].bar(x, sim_valid, color=["#607D8B", "#9E9E9E", "#2196F3", "#FF9800"][:len(labels)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_ylabel("Sim Valid (%)")
    axes[0].set_title("(a) Autoregressive Sim Valid")

    axes[1].bar(x, reward, color=["#607D8B", "#9E9E9E", "#2196F3", "#FF9800"][:len(labels)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("(b) Autoregressive Reward")

    fig.suptitle("Phase 14: Extended GRPO Comparison", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase14_model_comparison.png")
    fig.savefig(FIG_DIR / "phase14_model_comparison.pdf")
    plt.close(fig)


def plot_hybrid_results():
    hybrid_path = ROOT / "results" / "hybrid_phase14.json"
    if not hybrid_path.exists():
        return

    data = json.loads(hybrid_path.read_text())
    summary = data.get("summary", {})
    methods = ["vcg", "ccfm", "hybrid"]

    sim_valid = [100.0 * summary.get(m, {}).get("sim_valid_rate", 0.0) for m in methods]
    struct_valid = [100.0 * summary.get(m, {}).get("struct_valid_rate", 0.0) for m in methods]
    reward = [summary.get(m, {}).get("mean_reward", 0.0) for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, struct_valid, width, label="Struct valid (%)", color="#4CAF50")
    ax.bar(x, sim_valid, width, label="Sim valid (%)", color="#FF9800")
    ax.bar(x + width, reward, width, label="Mean reward", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_title("Hybrid Ranking vs Single-Source Graph Generators")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase14_hybrid_comparison.png")
    fig.savefig(FIG_DIR / "phase14_hybrid_comparison.pdf")
    plt.close(fig)


def main():
    plot_grpo_progress()
    plot_model_comparison()
    plot_hybrid_results()
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
