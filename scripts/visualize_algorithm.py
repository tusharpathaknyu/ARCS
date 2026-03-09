#!/usr/bin/env python3
"""
ARCS Algorithm Architecture Visualization
==========================================
Generates publication-quality diagrams of the ARCS pipeline:
  1. End-to-end pipeline overview
  2. Token sequence structure
  3. GraphTransformer block internals
  4. RWPE computation
  5. Constrained generation FSM
  6. Two-head output routing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ── Color palette ──────────────────────────────────────────────────────
C = {
    "bg":       "#FAFAFA",
    "spec":     "#E3F2FD",   # blue-50
    "spec_d":   "#1565C0",   # blue-800
    "topo":     "#FFF3E0",   # orange-50
    "topo_d":   "#E65100",   # orange-900
    "comp":     "#E8F5E9",   # green-50
    "comp_d":   "#2E7D32",   # green-800
    "val":      "#FCE4EC",   # pink-50
    "val_d":    "#AD1457",   # pink-800
    "special":  "#F3E5F5",   # purple-50
    "special_d":"#6A1B9A",   # purple-900
    "attn":     "#E0F7FA",   # cyan-50
    "attn_d":   "#00695C",   # teal-800
    "ffn":      "#FFF8E1",   # amber-50
    "ffn_d":    "#FF8F00",   # amber-800
    "rwpe":     "#EDE7F6",   # deep-purple-50
    "rwpe_d":   "#4527A0",   # deep-purple-800
    "gray":     "#ECEFF1",
    "gray_d":   "#455A64",
    "arrow":    "#37474F",
    "white":    "#FFFFFF",
    "black":    "#212121",
    "accent":   "#D32F2F",
}


def rounded_box(ax, xy, w, h, color, border, text, fontsize=9,
                fontweight="normal", text_color=None, alpha=1.0, radius=0.02):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=color, edgecolor=border, linewidth=1.5, alpha=alpha,
    )
    ax.add_patch(box)
    tc = text_color or border
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight=fontweight, color=tc,
        path_effects=[pe.withStroke(linewidth=2, foreground=color)] if alpha < 1 else [],
    )
    return box


def arrow(ax, start, end, color=None, style="-|>", lw=1.5, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    c = color or C["arrow"]
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                        connectionstyle=connectionstyle),
    )


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    ax.text(8, 5.2, "ARCS: End-to-End Pipeline",
            ha="center", fontsize=16, fontweight="bold", color=C["black"])

    # Stage boxes
    stages = [
        (0.0, 2.0, 2.2, 1.8, C["spec"],    C["spec_d"],  "Design\nSpecification\n\n$V_{in}$, $V_{out}$, $I_{out}$\n$f_{sw}$, topology"),
        (3.0, 2.0, 2.2, 1.8, C["topo"],    C["topo_d"],  "Tokenizer\n(686 vocab)\n\nLog-discretize\nvalues → 500 bins"),
        (6.0, 2.0, 2.6, 1.8, C["rwpe"],    C["rwpe_d"],  "GraphTransformer\n(6.8M params)\n\n6 layers, 4 heads\n+ RWPE + Graph bias"),
        (9.5, 2.0, 2.2, 1.8, C["comp"],    C["comp_d"],  "Constrained\nDecoder\n\nGrammar FSM\n+ Value bounds"),
        (12.5, 2.0, 2.2, 1.8, C["val"],    C["val_d"],   "SPICE\nNetlist\n\n16 topologies\n.measure stmts"),
    ]
    for x, y, w, h, fc, ec, txt in stages:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=8.5, fontweight="bold")

    # Arrows between stages
    arrow(ax, (2.2, 2.9), (3.0, 2.9))
    arrow(ax, (5.2, 2.9), (6.0, 2.9))
    arrow(ax, (8.6, 2.9), (9.5, 2.9))
    arrow(ax, (11.7, 2.9), (12.5, 2.9))

    # Token sequence below
    ax.text(8, 1.2, "Token Sequence Flow", ha="center", fontsize=11,
            fontweight="bold", color=C["gray_d"])

    tokens = [
        (0.3,  0.2, 0.9,  0.6, C["special"],  C["special_d"], "START"),
        (1.4,  0.2, 1.4,  0.6, C["topo"],     C["topo_d"],    "TOPO_BUCK"),
        (3.0,  0.2, 0.7,  0.6, C["special"],  C["special_d"], "SEP"),
        (3.9,  0.2, 1.2,  0.6, C["spec"],     C["spec_d"],    "SPEC_VIN"),
        (5.3,  0.2, 0.7,  0.6, C["val"],      C["val_d"],     "12.0"),
        (6.2,  0.2, 1.3,  0.6, C["spec"],     C["spec_d"],    "SPEC_VOUT"),
        (7.7,  0.2, 0.7,  0.6, C["val"],      C["val_d"],     "5.0"),
        (8.6,  0.2, 0.5,  0.6, C["gray"],     C["gray_d"],    "..."),
        (9.3,  0.2, 0.7,  0.6, C["special"],  C["special_d"], "SEP"),
        (10.2, 0.2, 0.7,  0.6, C["comp"],     C["comp_d"],    "R"),
        (11.1, 0.2, 0.7,  0.6, C["val"],      C["val_d"],     "4.7k"),
        (12.0, 0.2, 0.7,  0.6, C["comp"],     C["comp_d"],    "L"),
        (12.9, 0.2, 0.8,  0.6, C["val"],      C["val_d"],     "22µH"),
        (13.9, 0.2, 0.9,  0.6, C["special"],  C["special_d"], "END"),
    ]
    for x, y, w, h, fc, ec, txt in tokens:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=7, radius=0.01)

    # Brace labels
    ax.annotate("prefix (given)", xy=(5, 0.05), fontsize=8, color=C["spec_d"],
                ha="center", style="italic")
    ax.annotate("generated →", xy=(12, 0.05), fontsize=8, color=C["comp_d"],
                ha="center", style="italic")

    # Training loop annotation
    rounded_box(ax, (3.0, 4.2), 6.0, 0.7, C["gray"], C["gray_d"],
                "Training: CE loss (value_weight=5×) + AdamW (lr=3e-4) + Cosine LR + 5× Augmentation",
                fontsize=8)
    arrow(ax, (6.0, 4.2), (6.0, 3.8), color=C["gray_d"], style="-|>")
    arrow(ax, (9.0, 4.2), (9.0, 3.8), color=C["gray_d"], style="-|>")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: GraphTransformer Block Detail
# ═══════════════════════════════════════════════════════════════════════
def fig_transformer_block():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-0.5, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    ax.text(6, 9.6, "GraphTransformer Decoder Architecture",
            ha="center", fontsize=15, fontweight="bold", color=C["black"])

    # ── Embeddings (bottom) ──
    ax.text(6, 0.1, "Input Embeddings (summed)", ha="center", fontsize=10,
            fontweight="bold", color=C["gray_d"])

    embs = [
        (0.2, 0.5, 2.0, 0.7, C["special"], C["special_d"], "Token Emb\n(686 × 256)"),
        (2.5, 0.5, 2.0, 0.7, C["spec"],    C["spec_d"],    "Position Emb\n(128 × 256)"),
        (4.8, 0.5, 2.0, 0.7, C["topo"],    C["topo_d"],    "Type Emb\n(7 × 256)"),
        (7.1, 0.5, 2.6, 0.7, C["rwpe"],    C["rwpe_d"],    "RWPE Proj\n8→64→256 (GELU)"),
    ]
    for x, y, w, h, fc, ec, txt in embs:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=8)

    # Sum symbol
    ax.text(6, 1.5, "⊕", ha="center", va="center", fontsize=22, color=C["black"])
    for ex in [1.2, 3.5, 5.8, 8.4]:
        arrow(ax, (ex, 1.2), (6, 1.35), color=C["gray_d"], lw=1)

    # Dropout
    rounded_box(ax, (5, 1.8), 2, 0.4, C["gray"], C["gray_d"], "Dropout(0.1)", fontsize=8)
    arrow(ax, (6, 1.5), (6, 1.8), color=C["arrow"])

    # ── Transformer block ──
    bx, by, bw, bh = 1.5, 2.8, 9.0, 4.8
    block_rect = FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=C["white"], edgecolor=C["attn_d"],
        linewidth=2, linestyle="--", alpha=0.5,
    )
    ax.add_patch(block_rect)
    ax.text(bx + 0.2, by + bh - 0.3, "× 6  GraphTransformerBlock",
            fontsize=11, fontweight="bold", color=C["attn_d"])

    # RMSNorm 1
    rounded_box(ax, (4.5, 3.0), 3.0, 0.5, C["gray"], C["gray_d"],
                "RMSNorm(256)", fontsize=9)
    arrow(ax, (6, 2.2), (6, 3.0), color=C["arrow"])

    # Graph-Aware Attention
    rounded_box(ax, (2.5, 3.8), 7.0, 1.5, C["attn"], C["attn_d"], "", fontsize=9)
    ax.text(6, 5.0, "Graph-Aware Causal Self-Attention", ha="center",
            fontsize=10, fontweight="bold", color=C["attn_d"])

    # Inside attention
    attn_items = [
        (2.8, 4.0, 1.8, 0.6, C["white"], C["attn_d"], "QKV\n(256→768)"),
        (5.0, 4.0, 2.2, 0.6, C["topo"],  C["topo_d"], "adj_bias\n(4 heads)"),
        (7.6, 4.0, 1.6, 0.6, C["comp"],  C["comp_d"], "edge_type\n(17×4)"),
    ]
    for x, y, w, h, fc, ec, txt in attn_items:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=7)

    # Attention formula
    ax.text(6, 3.7, r"$\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{64}} + b_{\mathrm{adj}} \cdot A + b_{\mathrm{edge}}(E) + M_{\mathrm{causal}}\right)V$",
            ha="center", fontsize=9, color=C["attn_d"], style="italic")

    arrow(ax, (6, 3.5), (6, 3.8), color=C["attn_d"])

    # Residual arrow (left side)
    ax.annotate("", xy=(2.0, 5.6), xytext=(2.0, 3.0),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(1.3, 4.3, "+", fontsize=14, color=C["accent"], fontweight="bold")

    # RMSNorm 2
    rounded_box(ax, (4.5, 5.6), 3.0, 0.5, C["gray"], C["gray_d"],
                "RMSNorm(256)", fontsize=9)

    # SwiGLU FFN
    rounded_box(ax, (2.5, 6.4), 7.0, 0.8, C["ffn"], C["ffn_d"], "", fontsize=9)
    ax.text(6, 6.8, "SwiGLU FFN:  SiLU(W₁x) ⊙ W₃x → W₂ → Dropout",
            ha="center", fontsize=9, fontweight="bold", color=C["ffn_d"])
    ax.text(6, 6.5, "256 → 1024 → 256", ha="center", fontsize=8, color=C["ffn_d"])

    arrow(ax, (6, 6.1), (6, 6.4), color=C["arrow"])

    # Residual arrow (right side)
    ax.annotate("", xy=(10.0, 7.4), xytext=(10.0, 5.8),
                arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.5,
                                connectionstyle="arc3,rad=0.3"))
    ax.text(10.3, 6.6, "+", fontsize=14, color=C["accent"], fontweight="bold")

    # ── Output heads ──
    rounded_box(ax, (4.5, 7.6), 3.0, 0.5, C["gray"], C["gray_d"],
                "RMSNorm(256)", fontsize=9)
    arrow(ax, (6, 7.2), (6, 7.6), color=C["arrow"])

    # Two heads
    rounded_box(ax, (1.5, 8.4), 3.5, 0.7, C["comp"], C["comp_d"],
                "Structure Head\nLinear(256→686)\n[weight-tied w/ tok_emb]", fontsize=7.5)
    rounded_box(ax, (7.0, 8.4), 3.8, 0.7, C["val"], C["val_d"],
                "Value Head\nMLP(256→256→256, SiLU) + residual\n→ Linear(256→686)", fontsize=7.5)

    arrow(ax, (4.5, 8.1), (3.2, 8.4), color=C["comp_d"])
    arrow(ax, (7.5, 8.1), (8.9, 8.4), color=C["val_d"])

    # Routing label
    ax.text(6, 8.2, "route by\ntoken type", ha="center", fontsize=7,
            color=C["gray_d"], style="italic")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: RWPE Computation
# ═══════════════════════════════════════════════════════════════════════
def fig_rwpe():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Random-Walk Positional Encoding (RWPE)", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.patch.set_facecolor(C["white"])

    # Panel 1: Graph example (Buck converter)
    ax = axes[0]
    ax.set_xlim(-0.3, 3.3)
    ax.set_ylim(-0.3, 3.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Buck Converter Topology Graph", fontsize=11, fontweight="bold",
                 color=C["rwpe_d"])

    nodes = {
        "L":  (1.5, 2.5),
        "C":  (0.5, 0.5),
        "ESR": (2.5, 0.5),
        "M":  (1.5, 1.5),
    }
    edges = [("L", "M"), ("L", "C"), ("C", "ESR")]

    for n1, n2 in edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], color=C["gray_d"], linewidth=2, zorder=1)

    colors = {"L": C["comp"], "C": C["spec"], "ESR": C["val"], "M": C["topo"]}
    borders = {"L": C["comp_d"], "C": C["spec_d"], "ESR": C["val_d"], "M": C["topo_d"]}
    degrees = {"L": 2, "C": 2, "ESR": 1, "M": 1}

    for name, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.35, facecolor=colors[name],
                            edgecolor=borders[name], linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y + 0.05, name, ha="center", va="center", fontsize=10,
                fontweight="bold", color=borders[name])
        ax.text(x, y - 0.18, f"deg={degrees[name]}", ha="center", fontsize=7,
                color=C["gray_d"])

    # Panel 2: Matrix computation
    ax = axes[1]
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-3, 5.5)
    ax.axis("off")
    ax.set_title("Transition Matrix & Walk Probabilities", fontsize=11,
                 fontweight="bold", color=C["rwpe_d"])

    # Adjacency matrix
    A = np.array([
        [0, 1, 0, 1],  # L
        [1, 0, 1, 0],  # C
        [0, 1, 0, 0],  # ESR
        [1, 0, 0, 0],  # M
    ])
    labels_short = ["L", "C", "R", "M"]

    ax.text(1.5, 5.0, r"$A$ (adjacency)", ha="center", fontsize=10, color=C["black"])
    for i in range(4):
        ax.text(-0.3, 3.8 - i * 0.6, labels_short[i], fontsize=8, ha="center",
                va="center", color=C["gray_d"], fontweight="bold")
        for j in range(4):
            color = C["attn"] if A[i, j] else C["white"]
            rect = FancyBboxPatch((j * 0.6 + 0.1, 3.5 - i * 0.6), 0.5, 0.5,
                                  boxstyle="round,pad=0.02", facecolor=color,
                                  edgecolor=C["gray_d"], linewidth=0.5)
            ax.add_patch(rect)
            ax.text(j * 0.6 + 0.35, 3.75 - i * 0.6, str(A[i, j]),
                    ha="center", va="center", fontsize=8)

    # T = D^{-1}A
    D_inv = np.array([
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    T = D_inv @ A

    ax.text(4.5, 5.0, r"$T = D^{-1}A$", ha="center", fontsize=10, color=C["black"])
    for i in range(4):
        ax.text(3.2, 3.8 - i * 0.6, labels_short[i], fontsize=8, ha="center",
                va="center", color=C["gray_d"], fontweight="bold")
        for j in range(4):
            val = T[i, j]
            color = C["rwpe"] if val > 0 else C["white"]
            rect = FancyBboxPatch((j * 0.7 + 3.6, 3.5 - i * 0.6), 0.6, 0.5,
                                  boxstyle="round,pad=0.02", facecolor=color,
                                  edgecolor=C["gray_d"], linewidth=0.5)
            ax.add_patch(rect)
            txt = f"{val:.1f}" if val > 0 else "0"
            ax.text(j * 0.7 + 3.9, 3.75 - i * 0.6, txt,
                    ha="center", va="center", fontsize=7.5)

    # RWPE formula
    ax.text(3.0, 1.2, r"$\mathrm{RWPE}_i = \left[T^1_{ii},\; T^2_{ii},\; \ldots,\; T^8_{ii}\right]$",
            ha="center", fontsize=11, color=C["rwpe_d"], fontweight="bold")

    # Example values
    Tk = T.copy()
    rwpe_L = []
    for k in range(1, 9):
        Tk = Tk @ T if k > 1 else T
        rwpe_L.append(Tk[0, 0])

    ax.text(3.0, 0.5, f"RWPE(L) = [{', '.join(f'{v:.2f}' for v in rwpe_L)}]",
            ha="center", fontsize=8, color=C["rwpe_d"], family="monospace")

    rwpe_ESR = []
    Tk = T.copy()
    for k in range(1, 9):
        Tk = Tk @ T if k > 1 else T
        rwpe_ESR.append(Tk[2, 2])

    ax.text(3.0, 0.0, f"RWPE(ESR) = [{', '.join(f'{v:.2f}' for v in rwpe_ESR)}]",
            ha="center", fontsize=8, color=C["val_d"], family="monospace")

    ax.text(3.0, -0.6, "degree-2 nodes: higher return prob\ndegree-1 nodes: lower return prob",
            ha="center", fontsize=8, color=C["gray_d"], style="italic")

    # Panel 3: Projection pipeline
    ax = axes[2]
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")
    ax.set_title("RWPE → Embedding", fontsize=11, fontweight="bold",
                 color=C["rwpe_d"])

    steps = [
        (0.8, 5.0, 3.0, 0.6, C["rwpe"],   C["rwpe_d"], "RWPE features\n(B, T, 8)"),
        (0.8, 3.8, 3.0, 0.6, C["gray"],   C["gray_d"], "Linear(8 → 64)"),
        (0.8, 2.8, 3.0, 0.5, C["ffn"],    C["ffn_d"],  "GELU"),
        (0.8, 1.8, 3.0, 0.6, C["gray"],   C["gray_d"], "Linear(64 → 256)"),
        (0.8, 0.5, 3.0, 0.8, C["attn"],   C["attn_d"], "⊕ tok_emb + pos_emb\n+ type_emb + rwpe_proj"),
    ]
    for x, y, w, h, fc, ec, txt in steps:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=8.5)

    for y_top, y_bot in [(5.0, 4.4), (3.8, 3.3), (2.8, 2.4), (1.8, 1.3)]:
        arrow(ax, (2.3, y_bot), (2.3, y_top - 0.6), color=C["rwpe_d"])

    ax.text(2.3, -0.2, "17,216 params (0.25% of model)", ha="center",
            fontsize=8, color=C["gray_d"], style="italic")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Constrained Generation FSM
# ═══════════════════════════════════════════════════════════════════════
def fig_constrained():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    ax.text(7, 5.2, "Constrained Generation: Grammar State Machine",
            ha="center", fontsize=14, fontweight="bold", color=C["black"])

    # States
    states = [
        (0.5, 2.5, 2.2, 1.2, C["special"], C["special_d"],
         "PREFIX\n\nSTART→TOPO→SEP\n→specs→SEP"),
        (4.0, 2.5, 2.2, 1.2, C["comp"],    C["comp_d"],
         "EXPECT\nCOMP\n\nMask: only valid\nCOMP_* tokens"),
        (7.5, 2.5, 2.2, 1.2, C["val"],     C["val_d"],
         "EXPECT\nVALUE\n\nMask: VAL_*\nin phys. range"),
        (11.5, 2.5, 2.0, 1.2, C["special"], C["special_d"],
         "DONE\n\nEND token\nemitted"),
    ]
    for x, y, w, h, fc, ec, txt in states:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=7.5, fontweight="bold")

    # Transitions
    arrow(ax, (2.7, 3.1), (4.0, 3.1), color=C["arrow"], lw=2)
    ax.text(3.35, 3.35, "auto", fontsize=7, color=C["gray_d"], ha="center")

    arrow(ax, (6.2, 3.4), (7.5, 3.4), color=C["comp_d"], lw=2)
    ax.text(6.85, 3.65, "COMP_*", fontsize=7, color=C["comp_d"],
            ha="center", fontweight="bold")

    arrow(ax, (7.5, 2.8), (6.2, 2.8), color=C["val_d"], lw=2)
    ax.text(6.85, 2.55, "VAL_*", fontsize=7, color=C["val_d"],
            ha="center", fontweight="bold")

    arrow(ax, (9.7, 3.1), (11.5, 3.1), color=C["special_d"], lw=2)
    ax.text(10.6, 3.35, "END\n(≥2 comps)", fontsize=7, color=C["special_d"],
            ha="center")

    # Constraint levels
    ax.text(7, 1.5, "Three Constraint Levels", ha="center", fontsize=11,
            fontweight="bold", color=C["gray_d"])

    levels = [
        (1.0, 0.2, 3.0, 0.9, C["gray"],  C["gray_d"],
         "Level 1: GRAMMAR\nAlternating COMP→VAL\nEND after ≥2 components"),
        (5.0, 0.2, 3.5, 0.9, C["topo"],   C["topo_d"],
         "Level 2: TOPOLOGY\nOnly expected component types\n(e.g., Buck: L, C, ESR, MOSFET)"),
        (9.5, 0.2, 3.5, 0.9, C["val"],    C["val_d"],
         "Level 3: FULL\n+ Physical value bounds per\ncomponent (from ComponentBounds)"),
    ]
    for x, y, w, h, fc, ec, txt in levels:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=7.5)

    # Arrows showing nesting
    arrow(ax, (4.0, 0.65), (5.0, 0.65), color=C["arrow"], style="-|>")
    arrow(ax, (8.5, 0.65), (9.5, 0.65), color=C["arrow"], style="-|>")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Figure 5: Value Tokenization
# ═══════════════════════════════════════════════════════════════════════
def fig_tokenization():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("Value Tokenization: Log-Scale Discretization", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.patch.set_facecolor(C["white"])

    # Panel 1: Log scale bins
    ax = axes[0]
    vmin, vmax, nbins = 1e-12, 1e6, 500
    edges = np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # Show distribution of bin widths
    ax.set_xscale("log")
    widths = np.diff(edges)
    ax.bar(centers[:100], widths[:100], width=widths[:100] * 0.8,
           color=C["rwpe"], edgecolor=C["rwpe_d"], alpha=0.7, linewidth=0.3)
    ax.bar(centers[200:300], widths[200:300],
           width=widths[200:300] * 0.8,
           color=C["spec"], edgecolor=C["spec_d"], alpha=0.7, linewidth=0.3)
    ax.bar(centers[400:], widths[400:],
           width=widths[400:] * 0.8,
           color=C["val"], edgecolor=C["val_d"], alpha=0.7, linewidth=0.3)

    ax.set_xlabel("Physical Value", fontsize=10)
    ax.set_ylabel("Bin Width", fontsize=10)
    ax.set_title("500 Log-Spaced Bins: [1pF → 1MΩ]", fontsize=10)
    ax.set_yscale("log")

    # Annotate example values
    examples = [
        (1e-12, "1pF"),
        (1e-6, "1µH"),
        (1e3, "1kΩ"),
        (1e6, "1MΩ"),
    ]
    for val, label in examples:
        ax.axvline(val, color=C["accent"], linestyle="--", alpha=0.5, linewidth=1)
        ax.text(val * 1.5, ax.get_ylim()[1] * 0.3, label, fontsize=8,
                color=C["accent"], rotation=45)

    # Panel 2: Token vocabulary layout
    ax = axes[1]
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)
    ax.axis("off")
    ax.set_title("686-Token Vocabulary Structure", fontsize=10, fontweight="bold")

    blocks = [
        (0, 4.0, 1.5, 0.7, C["special"], C["special_d"], "Special\n[0–4] (5)"),
        (2, 4.0, 2.0, 0.7, C["comp"],    C["comp_d"],    "Component\n[5–24] (20)"),
        (4.5, 4.0, 2.0, 0.7, C["topo"],  C["topo_d"],    "Topology\n[25–44] (20)"),
        (7, 4.0, 2.0, 0.7, C["spec"],    C["spec_d"],    "Spec\n[45–64] (20)"),
        (0, 2.8, 2.5, 0.7, C["attn"],    C["attn_d"],    "Pin/Net\n[65–185] (121)"),
        (3, 2.8, 6.0, 0.7, C["val"],     C["val_d"],     "Value Bins [186–685] (500)\nlog₁₀ range: −12 → +6"),
    ]
    for x, y, w, h, fc, ec, txt in blocks:
        rounded_box(ax, (x, y), w, h, fc, ec, txt, fontsize=7.5)

    # Example encoding
    ax.text(4.5, 2.0, "Example:  22µH → bin 222 → token VAL_222 → id 408",
            ha="center", fontsize=9, color=C["val_d"], family="monospace")
    ax.text(4.5, 1.4, "Decode:   id 408 → VAL_222 → 10^(center[222]) → 2.19×10⁻⁵ ≈ 22µH",
            ha="center", fontsize=9, color=C["comp_d"], family="monospace")

    ax.text(4.5, 0.5, r"$\mathrm{bin}(v) = \left\lfloor \frac{\log_{10}(v) + 12}{18} \times 500 \right\rfloor$",
            ha="center", fontsize=12, color=C["black"])

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    out = Path("results/algorithm_figures")
    out.mkdir(parents=True, exist_ok=True)

    figures = {
        "1_pipeline":          fig_pipeline,
        "2_transformer_block": fig_transformer_block,
        "3_rwpe":              fig_rwpe,
        "4_constrained_gen":   fig_constrained,
        "5_tokenization":      fig_tokenization,
    }

    for name, func in figures.items():
        print(f"Generating {name}...")
        fig = func()
        path = out / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → {path}")

    print(f"\nAll figures saved to {out}/")
