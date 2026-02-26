#!/usr/bin/env python3
"""Generate ARCS architecture / pipeline diagram for the paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    # Colors
    C_SPEC = '#E3F2FD'
    C_MODEL = '#FFF3E0'
    C_SPICE = '#E8F5E9'
    C_OUT = '#FCE4EC'
    C_DATA = '#F3E5F5'
    C_RL = '#FFFDE7'

    def box(x, y, w, h, text, color, fontsize=10, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                           facecolor=color, edgecolor='black', linewidth=1.2)
        ax.add_patch(b)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2, text='', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, text, ha='center', va='bottom',
                    fontsize=8, color=color, fontstyle='italic')

    # ─── Top row: Inference pipeline ─────────────────────────
    ax.text(6, 5.2, 'ARCS Pipeline', ha='center', fontsize=14, weight='bold')

    # Input specs
    box(0, 3.3, 2.2, 1.2, 'Target Specs\n─────────\nVout=5V\nIout=1A\nfsw=100kHz', C_SPEC, 9, True)

    # Tokenizer
    box(2.8, 3.5, 1.5, 0.8, 'Tokenizer\n(686 tokens)', C_MODEL, 9)

    # Spec prefix
    box(4.8, 3.5, 2.2, 0.8, 'Spec Prefix\nSTART TOPO_BUCK SEP\nSPEC_VIN 12.2 ...', C_SPEC, 8)

    # Model
    box(7.6, 3.2, 2.0, 1.3, 'GPT Decoder\n────────\n6.5M params\n6 layers\nSwiGLU\nRMSNorm', C_MODEL, 9, True)

    # Component output
    box(10.2, 3.5, 2.0, 0.8, 'Components\nCAP 38µF\nRES 130mΩ\nIND 300µH', C_OUT, 8)

    # Arrows (top row)
    arrow(2.2, 3.9, 2.8, 3.9)
    arrow(4.3, 3.9, 4.8, 3.9)
    arrow(7.0, 3.9, 7.6, 3.9)
    arrow(9.6, 3.9, 10.2, 3.9)

    # ─── Bottom row: Training pipeline ───────────────────────

    # Data generation
    box(0, 0.5, 2.0, 1.5, 'SPICE Templates\n──────────\n16 Topologies\nRandom Sweep\n→ ngspice\n→ Metrics', C_DATA, 8, True)

    # Dataset
    box(2.6, 0.7, 1.8, 1.0, 'Training Data\n──────────\n32K circuits\n161K augmented', C_DATA, 8)

    # Stage 1
    box(5.0, 0.7, 2.0, 1.0, 'Stage 1:\nSupervised\n──────────\nCE loss\n5× value weight', C_MODEL, 8, True)

    # Stage 2
    box(7.6, 0.7, 2.2, 1.0, 'Stage 2:\nRL (REINFORCE)\n──────────\nSPICE reward\nKL penalty', C_RL, 8, True)

    # SPICE reward loop
    box(10.4, 0.7, 1.8, 1.0, 'SPICE-in-\nthe-Loop\n──────────\nSimulate\n→ Reward', C_SPICE, 8)

    # Arrows (bottom row)
    arrow(2.0, 1.2, 2.6, 1.2)
    arrow(4.4, 1.2, 5.0, 1.2)
    arrow(7.0, 1.2, 7.6, 1.2)
    arrow(9.8, 1.2, 10.4, 1.2)

    # RL feedback loop arrow
    ax.annotate('', xy=(8.7, 0.7), xytext=(10.8, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5,
                               connectionstyle='arc3,rad=0.5'))
    ax.text(9.7, 0.1, 'reward', ha='center', fontsize=8, color='red', fontstyle='italic')

    # Stage 2 → Model connection (vertical)
    ax.annotate('', xy=(8.7, 3.2), xytext=(8.7, 1.7),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5,
                               linestyle='dashed'))
    ax.text(9.0, 2.5, 'fine-tune', ha='left', fontsize=8, color='green', fontstyle='italic')

    # Stage 1 → Model connection (vertical)
    ax.annotate('', xy=(7.8, 3.2), xytext=(6.0, 1.7),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5,
                               linestyle='dashed'))
    ax.text(6.5, 2.5, 'pre-train', ha='left', fontsize=8, color='blue', fontstyle='italic')

    # Section labels
    ax.text(6, 2.7, '─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─',
            ha='center', fontsize=8, color='gray', alpha=0.5)
    ax.text(-0.3, 4.6, 'Inference', fontsize=10, color='gray', fontstyle='italic')
    ax.text(-0.3, 1.8, 'Training', fontsize=10, color='gray', fontstyle='italic')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'architecture.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'architecture.png'), dpi=300, bbox_inches='tight')
    print('Saved architecture.pdf/png')
    plt.close(fig)


if __name__ == '__main__':
    draw_pipeline()
