#!/usr/bin/env python3
"""Generate publication-quality training curve figures for the ARCS paper."""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Parse supervised training log ──────────────────────────────────
def parse_training_log(path):
    epochs, train_loss, val_loss = [], [], []
    v_acc_list, acc_list = [], []
    with open(path) as f:
        for line in f:
            m = re.match(
                r'Epoch\s+(\d+)/\d+\s+\|\s+train=([\d.]+)\s+ppl=[\d.]+\s+\|\s+'
                r'val=([\d.]+)\s+ppl=[\d.]+\s+\|\s+acc=([\d.]+)\s+v_acc=([\d.]+)',
                line
            )
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                acc_list.append(float(m.group(4)))
                v_acc_list.append(float(m.group(5)))
    return {
        'epoch': np.array(epochs),
        'train_loss': np.array(train_loss),
        'val_loss': np.array(val_loss),
        'acc': np.array(acc_list),
        'v_acc': np.array(v_acc_list),
    }


# ─── Parse RL training log ─────────────────────────────────────────
def parse_rl_log(path):
    steps, rewards, kl_vals = [], [], []
    struct_vals, valid_vals = [], []
    eval_steps, eval_rewards = [], []

    with open(path) as f:
        for line in f:
            # Training steps
            m = re.search(
                r'Step\s+(\d+)/\d+\s+\|\s+reward=([\d.]+)±[\d.]+\s+\|.*?'
                r'kl=([\d.]+).*?valid=(\d+)%.*?struct=(\d+)%',
                line
            )
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
                kl_vals.append(float(m.group(3)))
                valid_vals.append(int(m.group(4)))
                struct_vals.append(int(m.group(5)))
                continue

            # Eval steps
            m2 = re.search(
                r'Eval step\s+(\d+).*?eval_reward_mean.*?(\d+\.\d+)',
                line
            )
            if m2:
                eval_steps.append(int(m2.group(1)))
                eval_rewards.append(float(m2.group(2)))

    return {
        'step': np.array(steps),
        'reward': np.array(rewards),
        'kl': np.array(kl_vals),
        'valid_pct': np.array(valid_vals),
        'struct_pct': np.array(struct_vals),
        'eval_step': np.array(eval_steps),
        'eval_reward': np.array(eval_rewards),
    }


def smooth(y, window=5):
    """Simple moving average smoother."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


# ─── Figure 1: Supervised Training Curves ───────────────────────────
def plot_supervised(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Loss curves
    ax1.plot(data['epoch'], data['train_loss'], 'b-', linewidth=1.5, label='Train loss')
    ax1.plot(data['epoch'], data['val_loss'], 'r-', linewidth=1.5, label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('(a) Training & Validation Loss')
    ax1.legend()
    ax1.set_xlim(0, 100)

    # Mark best val loss
    best_idx = np.argmin(data['val_loss'])
    ax1.axvline(data['epoch'][best_idx], color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(
        f"Best: {data['val_loss'][best_idx]:.3f}\n(epoch {data['epoch'][best_idx]})",
        xy=(data['epoch'][best_idx], data['val_loss'][best_idx]),
        xytext=(data['epoch'][best_idx] + 5, data['val_loss'][best_idx] + 0.15),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='gray'
    )

    # Accuracy curves
    ax2.plot(data['epoch'], data['acc'], 'b-', linewidth=1.5, label='Overall accuracy')
    ax2.plot(data['epoch'], data['v_acc'], 'g-', linewidth=1.5, label='Value accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Token Accuracy')
    ax2.set_title('(b) Token Prediction Accuracy')
    ax2.legend()
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0.35, 0.85)

    fig.suptitle('Supervised Pre-training on Combined Dataset (32K circuits, 16 topologies)', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'supervised_training.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'supervised_training.png'))
    print(f'Saved supervised_training.pdf/png')
    plt.close(fig)


# ─── Figure 2: RL Training Curves ──────────────────────────────────
def plot_rl(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Smoothing window
    w = 7

    # (a) Reward
    ax = axes[0, 0]
    ax.plot(data['step'], data['reward'], alpha=0.3, color='blue', linewidth=0.8)
    sm_steps = data['step'][w-1:][:len(smooth(data['reward'], w))]
    ax.plot(sm_steps, smooth(data['reward'], w), 'b-', linewidth=1.5, label=f'Reward (MA-{w})')
    ax.axhline(4.28, color='gray', linestyle='--', alpha=0.5, label='Initial eval=4.28')
    ax.set_xlabel('Step')
    ax.set_ylabel('Batch Reward')
    ax.set_title('(a) Training Reward')
    ax.legend(fontsize=9)

    # (b) KL divergence
    ax = axes[0, 1]
    ax.plot(data['step'], data['kl'], alpha=0.3, color='orange', linewidth=0.8)
    ax.plot(sm_steps, smooth(data['kl'], w), color='darkorange', linewidth=1.5, label=f'KL (MA-{w})')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Target=0.5')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('(b) Policy KL from Reference')
    ax.legend(fontsize=9)

    # (c) Structural validity
    ax = axes[1, 0]
    ax.plot(data['step'], data['struct_pct'], alpha=0.3, color='green', linewidth=0.8)
    ax.plot(sm_steps, smooth(data['struct_pct'], w), 'g-', linewidth=1.5, label=f'Struct valid (MA-{w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Validity (%)')
    ax.set_title('(c) Structural Validity')
    ax.set_ylim(50, 105)
    ax.legend(fontsize=9)

    # (d) Simulation validity
    ax = axes[1, 1]
    ax.plot(data['step'], data['valid_pct'], alpha=0.3, color='red', linewidth=0.8)
    ax.plot(sm_steps, smooth(data['valid_pct'], w), 'r-', linewidth=1.5, label=f'Sim valid (MA-{w})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Validity (%)')
    ax.set_title('(d) Simulation Validity')
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9)

    fig.suptitle('REINFORCE Fine-tuning with SPICE-in-the-Loop Reward', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'rl_training.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'rl_training.png'))
    print(f'Saved rl_training.pdf/png')
    plt.close(fig)


# ─── Figure 3: Results comparison bar chart ─────────────────────────
def plot_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Reward comparison
    methods = ['RS\n(200 sims)', 'GA\n(630 sims)', 'ARCS\n(sup.)', 'ARCS\n(+RL)']
    rewards = [7.28, 7.48, 3.42, 3.64]
    colors = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#C44569']
    bars = ax1.bar(methods, rewards, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Average Reward (/8.0)')
    ax1.set_title('(a) Design Quality')
    ax1.set_ylim(0, 8.5)
    for bar, val in zip(bars, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # Speed comparison (log scale)
    times = [58.8, 271.2, 0.02, 0.02]
    bars = ax2.bar(methods, times, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Time per Design (seconds)')
    ax2.set_title('(b) Inference Speed')
    ax2.set_yscale('log')
    ax2.set_ylim(0.005, 500)
    for bar, val in zip(bars, times):
        label = f'{val:.0f}s' if val >= 1 else f'{val*1000:.0f}ms'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                label, ha='center', va='bottom', fontsize=10)

    fig.suptitle('ARCS vs. Search Baselines', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'comparison.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'comparison.png'))
    print(f'Saved comparison.pdf/png')
    plt.close(fig)


# ─── Figure 4: Per-topology heatmap ────────────────────────────────
def plot_topology_heatmap():
    topos = [
        'Buck', 'Boost', 'Buck-Boost', 'Cuk', 'SEPIC', 'Flyback', 'Forward',
        'Inv. Amp', 'Non-inv.', 'Inst. Amp', 'Diff. Amp',
        'SK LP', 'SK HP', 'SK BP', 'Wien Br.', 'Colpitts'
    ]
    sim_valid = [80, 70, 80, 30, 20, 70, 60,
                 100, 100, 100, 80, 10, 0, 20, 10, 10]

    fig, ax = plt.subplots(figsize=(12, 3))

    colors_map = plt.cm.RdYlGn
    norm = plt.Normalize(0, 100)

    bars = ax.barh(range(len(topos)), sim_valid, color=[colors_map(norm(v)) for v in sim_valid],
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(topos)))
    ax.set_yticklabels(topos)
    ax.set_xlabel('Simulation Validity (%)')
    ax.set_title('Per-Topology Simulation Validity (ARCS + RL, n=10 each)')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    # Add labels
    for i, v in enumerate(sim_valid):
        ax.text(v + 2, i, f'{v}%', va='center', fontsize=9)

    # Add tier separators
    ax.axhline(6.5, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(10.5, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.text(105, 3, 'Power', ha='right', va='center', fontsize=9, fontstyle='italic', color='gray')
    ax.text(105, 8.5, 'Amps', ha='right', va='center', fontsize=9, fontstyle='italic', color='gray')
    ax.text(105, 13, 'Filters/Osc', ha='right', va='center', fontsize=9, fontstyle='italic', color='gray')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'topology_validity.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'topology_validity.png'))
    print(f'Saved topology_validity.pdf/png')
    plt.close(fig)


# ─── Figure 5: Ablation bar chart ──────────────────────────────────
def plot_ablation():
    variants = ['Full\n(RL+Spec)', 'No RL\n(Sup. only)', 'No Spec\nCond.', 'Tier 1\nOnly']
    struct = [98.1, 90.6, 58.8, 100.0]
    sim_valid = [52.5, 46.9, 38.8, 20.6]
    reward = [3.49, 3.24, 2.60, 3.77]

    x = np.arange(len(variants))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width, struct, width, label='Struct Valid (%)', color='#2196F3', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x, sim_valid, width, label='Sim Valid (%)', color='#FF9800', edgecolor='black', linewidth=0.5)
    b3 = ax.bar(x + width, [r/8.0*100 for r in reward], width, label='Reward (% of max)',
                color='#4CAF50', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Percentage')
    ax.set_title('Ablation Study: Effect of RL, Spec Conditioning, and Topology Expansion')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 115)

    # Add reward values on top of green bars
    for i, r in enumerate(reward):
        ax.text(x[i] + width, r/8.0*100 + 2, f'{r:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ablation.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'ablation.png'))
    print(f'Saved ablation.pdf/png')
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    base = os.path.join(os.path.dirname(__file__), '..')

    # 1. Supervised training
    sup_log = os.path.join(base, 'logs', 'train_combined.log')
    if os.path.exists(sup_log):
        sup_data = parse_training_log(sup_log)
        print(f'Parsed {len(sup_data["epoch"])} epochs from supervised log')
        plot_supervised(sup_data)
    else:
        print(f'WARNING: {sup_log} not found')

    # 2. RL training
    rl_log = os.path.join(base, 'logs', 'rl_v2.log')
    if os.path.exists(rl_log):
        rl_data = parse_rl_log(rl_log)
        print(f'Parsed {len(rl_data["step"])} steps from RL log')
        plot_rl(rl_data)
    else:
        print(f'WARNING: {rl_log} not found')

    # 3. Comparison chart
    plot_comparison()

    # 4. Per-topology heatmap
    plot_topology_heatmap()

    # 5. Ablation
    plot_ablation()

    print(f'\nAll figures saved to {os.path.abspath(OUT_DIR)}/')
