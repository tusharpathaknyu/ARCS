"""
Visualize how AnalogGenie works — step by step.
Creates an animated GIF showing:
1. A simple analog circuit (NMOS + Resistor + Capacitor)
2. The pin-level graph representation 
3. The Eulerian circuit traversal
4. Token-by-token GPT prediction
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
import io
import os

# ============================================================
# CONFIG
# ============================================================
FIG_W, FIG_H = 16, 10
DPI = 120
BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
ACCENT1 = '#58a6ff'   # blue
ACCENT2 = '#3fb950'   # green
ACCENT3 = '#f0883e'   # orange
ACCENT4 = '#f778ba'   # pink
ACCENT5 = '#d2a8ff'   # purple
GRID_COLOR = '#21262d'
CARD_COLOR = '#161b22'
BORDER_COLOR = '#30363d'

frames = []

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=BG_COLOR, 
                bbox_inches='tight', pad_inches=0.3)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img

def add_frame(fig, duration_frames=1):
    img = fig_to_image(fig)
    for _ in range(duration_frames):
        frames.append(img)

def draw_rounded_box(ax, x, y, w, h, color=CARD_COLOR, edge_color=BORDER_COLOR, alpha=0.9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor=edge_color, linewidth=1.5, alpha=alpha)
    ax.add_patch(box)

# ============================================================
# FRAME SET 1: Title
# ============================================================
def draw_title():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    ax.text(5, 7, 'How AnalogGenie Works', fontsize=40, fontweight='bold',
            color=ACCENT1, ha='center', va='center', fontfamily='sans-serif')
    ax.text(5, 5.8, 'ICLR 2025 — Gao et al.', fontsize=20,
            color=TEXT_COLOR, ha='center', va='center', alpha=0.7)
    ax.text(5, 4.2, '"Predict the next device pin to connect"', fontsize=22,
            color=ACCENT3, ha='center', va='center', style='italic')
    ax.text(5, 2.5, 'A GPT model where tokens are circuit pins,\nnot words.',
            fontsize=18, color=TEXT_COLOR, ha='center', va='center', alpha=0.8)
    
    add_frame(fig, 8)
    plt.close()

# ============================================================
# FRAME SET 2: The Circuit Schematic
# ============================================================
def draw_circuit():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    # Title
    ax.text(5, 9.3, 'Step 1: Start with a Circuit', fontsize=28, fontweight='bold',
            color=ACCENT1, ha='center', va='center')
    ax.text(5, 8.6, 'A simple NMOS amplifier with resistor load and capacitor',
            fontsize=16, color=TEXT_COLOR, ha='center', va='center', alpha=0.7)
    
    # Draw VDD rail
    ax.plot([5, 5], [7.5, 7.2], color=ACCENT2, linewidth=3)
    ax.text(5, 7.7, 'VDD', fontsize=14, fontweight='bold', color=ACCENT2, ha='center')
    
    # Resistor R1 (between VDD and drain)
    rx, ry = 5, 6.1
    draw_rounded_box(ax, rx-0.5, ry-0.4, 1.0, 0.8, color='#1a2233', edge_color=ACCENT3)
    ax.text(rx, ry, 'R1', fontsize=16, fontweight='bold', color=ACCENT3, ha='center', va='center')
    ax.plot([5, 5], [7.2, ry+0.4], color=ACCENT3, linewidth=2)  # wire up
    ax.plot([5, 5], [ry-0.4, 5.0], color=ACCENT3, linewidth=2)  # wire down to drain
    
    # NMOS transistor NM1
    tx, ty = 5, 4.0
    draw_rounded_box(ax, tx-0.7, ty-0.6, 1.4, 1.2, color='#1a2233', edge_color=ACCENT1)
    ax.text(tx, ty, 'NM1', fontsize=16, fontweight='bold', color=ACCENT1, ha='center', va='center')
    # Drain (top)
    ax.plot([5, 5], [5.0, ty+0.6], color=ACCENT1, linewidth=2)
    ax.text(5.3, 4.8, 'D', fontsize=11, color=ACCENT1, alpha=0.8)
    # Source (bottom)
    ax.plot([5, 5], [ty-0.6, 2.5], color=ACCENT1, linewidth=2)
    ax.text(5.3, 3.2, 'S', fontsize=11, color=ACCENT1, alpha=0.8)
    # Gate (left)
    ax.plot([tx-0.7, tx-1.5], [ty, ty], color=ACCENT1, linewidth=2)
    ax.text(3.2, ty, 'G', fontsize=11, color=ACCENT1, alpha=0.8)
    ax.text(2.5, ty, 'VIN', fontsize=14, fontweight='bold', color=ACCENT5, ha='center')
    
    # Capacitor C1 (from drain to ground)
    cx, cy = 7, 4.0
    draw_rounded_box(ax, cx-0.5, cy-0.4, 1.0, 0.8, color='#1a2233', edge_color=ACCENT4)
    ax.text(cx, cy, 'C1', fontsize=16, fontweight='bold', color=ACCENT4, ha='center', va='center')
    # Connect C1 to drain node
    ax.plot([5, 7], [5.0, 5.0], color='#8b949e', linewidth=2)
    ax.plot([7, 7], [5.0, cy+0.4], color=ACCENT4, linewidth=2)
    # C1 to ground
    ax.plot([7, 7], [cy-0.4, 2.5], color=ACCENT4, linewidth=2)
    
    # VSS rail
    ax.plot([3.5, 8], [2.5, 2.5], color='#8b949e', linewidth=3)
    ax.text(5, 2.1, 'VSS (Ground)', fontsize=14, fontweight='bold', color='#8b949e', ha='center')
    
    # VOUT label
    ax.annotate('VOUT', xy=(6.0, 5.0), fontsize=14, fontweight='bold', color=ACCENT2,
                ha='center', va='bottom')
    ax.plot(6.0, 5.0, 'o', color=ACCENT2, markersize=8)
    
    # Component labels on the side
    info_x = 1.0
    ax.text(info_x, 1.2, 'Components:', fontsize=14, fontweight='bold', color=TEXT_COLOR)
    ax.text(info_x, 0.7, '• NM1: NMOS transistor (D, G, S, B pins)', fontsize=12, color=ACCENT1)  
    ax.text(info_x, 0.3, '• R1: Resistor (P, N pins)    • C1: Capacitor (P, N pins)', fontsize=12, color=TEXT_COLOR, alpha=0.7)
    
    add_frame(fig, 8)
    plt.close()

# ============================================================
# FRAME SET 3: Pin-Level Graph
# ============================================================
def draw_pin_graph(highlight_step=-1):
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    ax.text(5, 9.3, 'Step 2: Convert to Pin-Level Graph', fontsize=28, fontweight='bold',
            color=ACCENT1, ha='center', va='center')
    ax.text(5, 8.6, 'Each device PIN becomes a node (not each device)',
            fontsize=16, color=TEXT_COLOR, ha='center', va='center', alpha=0.7)
    
    # Node positions
    nodes = {
        'VDD':    (5.0, 7.5),
        'VSS':    (5.0, 1.5),
        'VIN':    (1.5, 4.5),
        'VOUT':   (8.0, 5.5),
        'NM1_D':  (4.0, 5.5),
        'NM1_G':  (2.5, 4.5),
        'NM1_S':  (4.0, 3.0),
        'NM1_B':  (2.5, 3.0),
        'R1_P':   (5.0, 6.5),
        'R1_N':   (4.0, 6.5),
        'C1_P':   (6.5, 5.5),
        'C1_N':   (6.5, 3.0),
    }
    
    # Edges (connections between pins)
    edges = [
        ('VDD', 'R1_P'),      # VDD connects to R1 positive
        ('R1_N', 'NM1_D'),    # R1 negative to NMOS drain
        ('NM1_D', 'C1_P'),    # Drain node to C1 positive  
        ('NM1_D', 'VOUT'),    # Drain node to output
        ('VIN', 'NM1_G'),     # Input to gate
        ('NM1_S', 'VSS'),     # Source to ground
        ('NM1_B', 'VSS'),     # Body to ground
        ('C1_N', 'VSS'),      # C1 negative to ground
    ]
    
    # Color map for nodes
    node_colors = {
        'VDD': ACCENT2, 'VSS': '#8b949e', 'VIN': ACCENT5, 'VOUT': ACCENT2,
        'NM1_D': ACCENT1, 'NM1_G': ACCENT1, 'NM1_S': ACCENT1, 'NM1_B': ACCENT1,
        'R1_P': ACCENT3, 'R1_N': ACCENT3,
        'C1_P': ACCENT4, 'C1_N': ACCENT4,
    }
    
    # Draw edges
    for n1, n2 in edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax.plot([x1, x2], [y1, y2], color='#484f58', linewidth=2, zorder=1)
    
    # Draw nodes
    for name, (x, y) in nodes.items():
        color = node_colors[name]
        circle = plt.Circle((x, y), 0.35, facecolor=color, edgecolor='white',
                           linewidth=1.5, zorder=2, alpha=0.9)
        ax.add_patch(circle)
        fontsize = 9 if len(name) > 4 else 11
        ax.text(x, y, name, fontsize=fontsize, fontweight='bold',
                color='white', ha='center', va='center', zorder=3)
    
    # Legend
    ax.text(0.3, 1.0, 'Each circle = one device PIN', fontsize=13, color=TEXT_COLOR, fontweight='bold')
    ax.text(0.3, 0.5, 'AnalogGenie: NM1 → 4 nodes (D,G,S,B)  |  R1 → 2 nodes (P,N)  |  C1 → 2 nodes (P,N)',
            fontsize=11, color=TEXT_COLOR, alpha=0.7)
    ax.text(0.3, 0.1, 'Previous work: NM1 → 1 node (ambiguous which pin connects where!)',
            fontsize=11, color='#f85149', alpha=0.8)
    
    # Device grouping boxes
    # NM1 box
    draw_rounded_box(ax, 1.8, 2.5, 2.8, 3.5, color='none', edge_color=ACCENT1)
    ax.text(3.2, 2.7, 'NM1', fontsize=11, color=ACCENT1, ha='center', alpha=0.6)
    # R1 box
    draw_rounded_box(ax, 3.5, 6.0, 2.0, 1.2, color='none', edge_color=ACCENT3)
    ax.text(4.5, 6.1, 'R1', fontsize=11, color=ACCENT3, ha='center', alpha=0.6)
    # C1 box
    draw_rounded_box(ax, 5.9, 2.5, 1.2, 3.5, color='none', edge_color=ACCENT4)
    ax.text(6.5, 2.7, 'C1', fontsize=11, color=ACCENT4, ha='center', alpha=0.6)
    
    add_frame(fig, 8)
    plt.close()

# ============================================================
# FRAME SET 4: Eulerian Circuit concept
# ============================================================
def draw_eulerian_concept():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    ax.text(5, 9.3, 'Step 3: Flatten Graph → Sequence', fontsize=28, fontweight='bold',
            color=ACCENT1, ha='center', va='center')
    ax.text(5, 8.5, 'Find an Eulerian Circuit: a path that visits every edge exactly once', fontsize=16,
            color=TEXT_COLOR, ha='center', va='center', alpha=0.7)
    
    # Show a simple example with 4 nodes
    # Triangle + extra node
    example_nodes = {
        'A': (2.5, 5.5),
        'B': (4.5, 5.5),
        'C': (3.5, 3.5),
    }
    example_edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    
    # Draw graph on left
    ax.text(3.5, 7.2, 'Graph (2D)', fontsize=18, fontweight='bold', color=ACCENT3, ha='center')
    draw_rounded_box(ax, 1.3, 2.5, 4.5, 5.0, color=CARD_COLOR, edge_color=BORDER_COLOR)
    
    for n1, n2 in example_edges:
        x1, y1 = example_nodes[n1]
        x2, y2 = example_nodes[n2]
        ax.plot([x1, x2], [y1, y2], color='#484f58', linewidth=3, zorder=1)
    
    for name, (x, y) in example_nodes.items():
        circle = plt.Circle((x, y), 0.35, facecolor=ACCENT1, edgecolor='white',
                           linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, name, fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=3)
    
    # Arrow
    ax.annotate('', xy=(6.2, 4.5), xytext=(5.5, 4.5),
                arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=3))
    ax.text(5.85, 5.0, 'Eulerian\nCircuit', fontsize=12, color=ACCENT2, ha='center', fontweight='bold')
    
    # Sequence on right
    ax.text(8.0, 7.2, 'Sequence (1D)', fontsize=18, fontweight='bold', color=ACCENT2, ha='center')
    draw_rounded_box(ax, 6.3, 2.5, 3.5, 5.0, color=CARD_COLOR, edge_color=BORDER_COLOR)
    
    seq = ['A', '→', 'B', '→', 'C', '→', 'A']
    for i, s in enumerate(seq):
        y = 6.0 - i * 0.5
        if s == '→':
            ax.text(8.0, y, '↓', fontsize=16, color='#484f58', ha='center', va='center')
        else:
            draw_rounded_box(ax, 7.3, y-0.2, 1.4, 0.4, color='#1a2233', edge_color=ACCENT1)
            ax.text(8.0, y, s, fontsize=14, fontweight='bold', color=ACCENT1, ha='center', va='center')
    
    # Bottom explanation
    ax.text(5, 1.8, 'Why? Transformers need 1D sequences, not 2D graphs.', fontsize=15,
            color=ACCENT3, ha='center', fontweight='bold')
    ax.text(5, 1.2, 'Eulerian circuits are more compact than adjacency matrices', fontsize=13,
            color=TEXT_COLOR, ha='center', alpha=0.7)
    ax.text(5, 0.6, '(6 nodes, 6 edges: adjacency = 36 elements  vs  Eulerian = 13 elements)',
            fontsize=12, color=TEXT_COLOR, ha='center', alpha=0.5)
    
    add_frame(fig, 8)
    plt.close()

# ============================================================
# FRAME SET 5: Real Circuit Eulerian Walk (animated)
# ============================================================
def draw_eulerian_walk():
    # The Eulerian walk through our circuit
    walk = ['VSS', 'NM1_S', 'NM1_D', 'R1_N', 'R1_P', 'VDD', 'R1_P', 'R1_N',
            'NM1_D', 'C1_P', 'C1_N', 'VSS', 'NM1_B', 'VSS', 'C1_N', 'C1_P',
            'NM1_D', 'VOUT', 'NM1_D', 'NM1_S', 'VSS']  # simplified
    
    # Actually let's use a cleaner simplified walk for visualization
    walk = ['VSS', 'NM1_S', 'NM1_D', 'R1_N', 'R1_P', 'VDD',
            'R1_P', 'R1_N', 'NM1_D', 'C1_P', 'C1_N', 'VSS',
            'NM1_B', 'VSS']
    
    node_positions = {
        'VDD':    (5.5, 7.0),
        'VSS':    (5.5, 2.0),
        'VIN':    (1.5, 4.5),
        'VOUT':   (9.0, 5.0),
        'NM1_D':  (4.0, 5.0),
        'NM1_G':  (2.5, 4.5),
        'NM1_S':  (4.0, 3.0),
        'NM1_B':  (2.5, 3.0),
        'R1_P':   (5.5, 6.0),
        'R1_N':   (4.0, 6.0),
        'C1_P':   (7.0, 5.0),
        'C1_N':   (7.0, 3.0),
    }
    
    node_colors = {
        'VDD': ACCENT2, 'VSS': '#8b949e', 'VIN': ACCENT5, 'VOUT': ACCENT2,
        'NM1_D': ACCENT1, 'NM1_G': ACCENT1, 'NM1_S': ACCENT1, 'NM1_B': ACCENT1,
        'R1_P': ACCENT3, 'R1_N': ACCENT3,
        'C1_P': ACCENT4, 'C1_N': ACCENT4,
    }
    
    edges = [
        ('VDD', 'R1_P'), ('R1_N', 'NM1_D'), ('NM1_D', 'C1_P'),
        ('NM1_D', 'VOUT'), ('VIN', 'NM1_G'),
        ('NM1_S', 'VSS'), ('NM1_B', 'VSS'), ('C1_N', 'VSS'),
    ]
    
    for step in range(len(walk)):
        fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor(BG_COLOR)
        
        ax.text(5, 9.3, 'Step 4: Walk the Eulerian Circuit', fontsize=28, fontweight='bold',
                color=ACCENT1, ha='center', va='center')
        ax.text(5, 8.5, f'Traversal step {step+1}/{len(walk)}  —  Current: {walk[step]}',
                fontsize=16, color=ACCENT3, ha='center', va='center')
        
        # Draw all edges (dim)
        for n1, n2 in edges:
            if n1 in node_positions and n2 in node_positions:
                x1, y1 = node_positions[n1]
                x2, y2 = node_positions[n2]
                ax.plot([x1, x2], [y1, y2], color='#21262d', linewidth=2, zorder=1)
        
        # Draw traversed edges (bright)
        for i in range(step):
            if walk[i] in node_positions and walk[i+1] in node_positions:
                x1, y1 = node_positions[walk[i]]
                x2, y2 = node_positions[walk[i+1]]
                ax.plot([x1, x2], [y1, y2], color=ACCENT2, linewidth=3, zorder=2, alpha=0.8)
        
        # Draw all nodes
        visited = set(walk[:step+1])
        for name, (x, y) in node_positions.items():
            color = node_colors[name]
            alpha = 1.0 if name in visited else 0.3
            circle = plt.Circle((x, y), 0.3, facecolor=color, edgecolor='white',
                               linewidth=1.5, zorder=3, alpha=alpha)
            ax.add_patch(circle)
            fontsize = 8 if len(name) > 4 else 10
            ax.text(x, y, name, fontsize=fontsize, fontweight='bold',
                    color='white', ha='center', va='center', zorder=4, alpha=alpha)
        
        # Highlight current node
        cx, cy = node_positions[walk[step]]
        highlight = plt.Circle((cx, cy), 0.4, facecolor='none', edgecolor=ACCENT3,
                              linewidth=3, zorder=5, linestyle='--')
        ax.add_patch(highlight)
        
        # Show sequence so far at bottom
        seq_str = ' → '.join(walk[:step+1])
        if len(seq_str) > 80:
            seq_str = '... → ' + ' → '.join(walk[max(0,step-5):step+1])
        
        draw_rounded_box(ax, 0.3, 0.3, 9.4, 1.2, color=CARD_COLOR, edge_color=BORDER_COLOR)
        ax.text(0.5, 1.2, 'Sequence so far:', fontsize=11, color=TEXT_COLOR, fontweight='bold')
        ax.text(0.5, 0.7, seq_str, fontsize=10, color=ACCENT2, fontfamily='monospace')
        
        add_frame(fig, 3)
        plt.close()

# ============================================================
# FRAME SET 6: Tokenization
# ============================================================
def draw_tokenization():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    ax.text(5, 9.3, 'Step 5: Tokenize (Components = Words)', fontsize=28, fontweight='bold',
            color=ACCENT1, ha='center', va='center')
    ax.text(5, 8.5, 'Each device pin maps to a unique integer token (vocab = 1029)',
            fontsize=16, color=TEXT_COLOR, ha='center', va='center', alpha=0.7)
    
    # Token table
    tokens = [
        ('NM1_D', '1', ACCENT1, 'NMOS #1, Drain'),
        ('NM1_G', '2', ACCENT1, 'NMOS #1, Gate'),
        ('NM1_S', '3', ACCENT1, 'NMOS #1, Source'),
        ('NM1_B', '4', ACCENT1, 'NMOS #1, Body'),
        ('R1_P', '451', ACCENT3, 'Resistor #1, Positive'),
        ('R1_N', '452', ACCENT3, 'Resistor #1, Negative'),
        ('C1_P', '526', ACCENT4, 'Capacitor #1, Positive'),
        ('C1_N', '527', ACCENT4, 'Capacitor #1, Negative'),
        ('VDD', '1026', ACCENT2, 'Power supply'),
        ('VSS', '1027', '#8b949e', 'Ground (always START token)'),
        ('TRUNCATE', '1028', '#f85149', 'End of circuit'),
    ]
    
    y_start = 7.5
    for i, (name, idx, color, desc) in enumerate(tokens):
        y = y_start - i * 0.55
        
        # Token name box
        draw_rounded_box(ax, 1.0, y-0.2, 1.8, 0.4, color='#1a2233', edge_color=color)
        ax.text(1.9, y, name, fontsize=12, fontweight='bold', color=color,
                ha='center', va='center', fontfamily='monospace')
        
        # Arrow
        ax.text(3.2, y, '→', fontsize=14, color='#484f58', ha='center', va='center')
        
        # Token index
        draw_rounded_box(ax, 3.6, y-0.2, 1.0, 0.4, color='#1a2233', edge_color=BORDER_COLOR)
        ax.text(4.1, y, idx, fontsize=12, fontweight='bold', color=TEXT_COLOR,
                ha='center', va='center', fontfamily='monospace')
        
        # Description
        ax.text(5.0, y, desc, fontsize=11, color=TEXT_COLOR, alpha=0.6, va='center')
    
    # Bottom: example sequence
    draw_rounded_box(ax, 0.5, 0.3, 9.0, 1.2, color=CARD_COLOR, edge_color=ACCENT2)
    ax.text(0.7, 1.2, 'Circuit as token sequence:', fontsize=12, color=ACCENT2, fontweight='bold')
    ax.text(0.7, 0.7, '[1027, 3, 1, 452, 451, 1026, 451, 452, 1, 526, 527, 1027, 4, 1027]',
            fontsize=11, color=TEXT_COLOR, fontfamily='monospace')
    
    add_frame(fig, 8)
    plt.close()

# ============================================================
# FRAME SET 7: GPT Prediction (animated)
# ============================================================
def draw_gpt_generation():
    generated = ['VSS']
    full_seq = ['VSS', 'NM1_S', 'NM1_D', 'R1_N', 'R1_P', 'VDD', '...', 'TRUNCATE']
    
    token_colors = {
        'VSS': '#8b949e', 'NM1_S': ACCENT1, 'NM1_D': ACCENT1,
        'R1_N': ACCENT3, 'R1_P': ACCENT3, 'VDD': ACCENT2,
        '...': '#484f58', 'TRUNCATE': '#f85149',
    }
    
    for step in range(len(full_seq)):
        fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor(BG_COLOR)
        
        ax.text(5, 9.3, 'Step 6: GPT Predicts Next Pin', fontsize=28, fontweight='bold',
                color=ACCENT1, ha='center', va='center')
        ax.text(5, 8.5, 'Autoregressive generation — one token at a time',
                fontsize=16, color=TEXT_COLOR, ha='center', va='center', alpha=0.7)
        
        # Draw GPT model box
        draw_rounded_box(ax, 3, 4.5, 4, 2.5, color=CARD_COLOR, edge_color=ACCENT5)
        ax.text(5, 6.5, 'GPT Model', fontsize=20, fontweight='bold', color=ACCENT5, ha='center')
        ax.text(5, 5.9, '6 layers × 6 heads', fontsize=12, color=TEXT_COLOR, ha='center', alpha=0.5)
        ax.text(5, 5.4, '11.8M parameters', fontsize=12, color=TEXT_COLOR, ha='center', alpha=0.5)
        ax.text(5, 4.9, 'Next-token prediction', fontsize=12, color=ACCENT3, ha='center')
        
        # Input tokens (below model)
        ax.text(5, 3.8, 'Input: tokens so far', fontsize=12, color=TEXT_COLOR, ha='center', alpha=0.7)
        current_seq = full_seq[:step+1]
        x_start = 5 - len(current_seq) * 0.6
        for i, tok in enumerate(current_seq):
            x = x_start + i * 1.2
            if x < 0.5 or x > 9.5: continue
            color = token_colors.get(tok, ACCENT1)
            draw_rounded_box(ax, x-0.5, 2.8, 1.0, 0.5, color='#1a2233', edge_color=color)
            ax.text(x, 3.05, tok, fontsize=8, fontweight='bold', color=color,
                    ha='center', va='center', fontfamily='monospace')
        
        # Arrow up to model
        ax.annotate('', xy=(5, 4.5), xytext=(5, 3.4),
                    arrowprops=dict(arrowstyle='->', color='#484f58', lw=2))
        
        # Output prediction (above model)
        if step < len(full_seq) - 1:
            next_tok = full_seq[step+1]
            ax.annotate('', xy=(5, 8.0), xytext=(5, 7.0),
                        arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=2))
            next_color = token_colors.get(next_tok, ACCENT1)
            draw_rounded_box(ax, 4.0, 7.7, 2.0, 0.6, color='#1a2233', edge_color=next_color)
            ax.text(5, 8.0, f'→ {next_tok}', fontsize=14, fontweight='bold', color=next_color,
                    ha='center', va='center')
            ax.text(5, 7.5, 'Predicts:', fontsize=11, color=ACCENT2, ha='center')
        else:
            ax.text(5, 7.8, '✓ Circuit Complete!', fontsize=20, fontweight='bold',
                    color=ACCENT2, ha='center')
        
        # Probability distribution hint
        if step < len(full_seq) - 1:
            probs = [('NM1_S', 0.42), ('R1_P', 0.18), ('C1_P', 0.12), ('NM1_G', 0.08)] if step == 0 else \
                    [('NM1_D', 0.55), ('NM1_G', 0.15), ('VDD', 0.10), ('C1_P', 0.05)] if step == 1 else \
                    [(full_seq[step+1], 0.6), ('VDD', 0.15), ('VSS', 0.1), ('C1_N', 0.05)]
            
            ax.text(8.5, 7.5, 'P(next):', fontsize=10, color=TEXT_COLOR, ha='center', fontweight='bold')
            for j, (p_name, p_val) in enumerate(probs):
                y = 7.0 - j * 0.35
                bar_w = p_val * 2.5
                ax.barh(y, bar_w, height=0.25, left=7.5, color=ACCENT1, alpha=0.4 + p_val)
                ax.text(7.4, y, p_name, fontsize=8, color=TEXT_COLOR, ha='right', va='center',
                        fontfamily='monospace')
                ax.text(7.5 + bar_w + 0.1, y, f'{p_val:.0%}', fontsize=8, color=TEXT_COLOR,
                        ha='left', va='center', alpha=0.6)
        
        # Bottom explanation
        draw_rounded_box(ax, 0.5, 0.3, 9.0, 1.8, color=CARD_COLOR, edge_color=BORDER_COLOR)
        ax.text(5, 1.7, 'This is IDENTICAL to how ChatGPT generates text:', fontsize=13,
                color=ACCENT3, ha='center', fontweight='bold')
        ax.text(5, 1.1, 'ChatGPT:     "The" → "cat" → "sat" → "on" → "the" → "mat"',
                fontsize=12, color=TEXT_COLOR, ha='center', fontfamily='monospace', alpha=0.7)
        ax.text(5, 0.6, 'AnalogGenie: "VSS" → "NM1_S" → "NM1_D" → "R1_N" → "R1_P" → "VDD"',
                fontsize=12, color=ACCENT1, ha='center', fontfamily='monospace')
        
        add_frame(fig, 5)
        plt.close()

# ============================================================
# FRAME SET 8: Summary
# ============================================================
def draw_summary():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(BG_COLOR)
    
    ax.text(5, 9.0, 'AnalogGenie: The Full Pipeline', fontsize=32, fontweight='bold',
            color=ACCENT1, ha='center', va='center')
    
    steps = [
        ('1', 'Circuit\nSchematic', ACCENT3, '3,350 from\ntextbooks'),
        ('2', 'Pin-Level\nGraph', ACCENT1, 'Each pin =\na node'),
        ('3', 'Eulerian\nCircuit', ACCENT2, 'Graph → \nSequence'),
        ('4', 'Tokenize', ACCENT5, '1,029 token\nvocabulary'),
        ('5', 'GPT\nTrain', ACCENT4, '11.8M params\n100K iters'),
        ('6', 'Generate\nCircuits', ACCENT2, '93.2% valid\n~100% novel'),
    ]
    
    for i, (num, label, color, detail) in enumerate(steps):
        x = 1.0 + i * 1.5
        y = 5.5
        
        # Step circle
        circle = plt.Circle((x, y+1), 0.4, facecolor=color, edgecolor='white', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y+1, num, fontsize=16, fontweight='bold', color='white',
                ha='center', va='center', zorder=3)
        
        # Label
        ax.text(x, y, label, fontsize=11, fontweight='bold', color=color,
                ha='center', va='center')
        
        # Detail
        ax.text(x, y-1.0, detail, fontsize=9, color=TEXT_COLOR,
                ha='center', va='center', alpha=0.6)
        
        # Arrow
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 1.0, y+1), xytext=(x + 0.5, y+1),
                        arrowprops=dict(arrowstyle='->', color='#484f58', lw=2))
    
    # What's missing
    draw_rounded_box(ax, 0.5, 0.5, 9.0, 2.5, color='#1a1a00', edge_color=ACCENT3)
    ax.text(5, 2.6, 'What AnalogGenie CANNOT do (= our opportunity):', fontsize=15,
            color=ACCENT3, ha='center', fontweight='bold')
    ax.text(5, 2.0, '❌ No component VALUES (just wiring)     ❌ No spec conditioning ("give me 5V 2A")',
            fontsize=12, color=TEXT_COLOR, ha='center')
    ax.text(5, 1.4, '❌ IC-only (no discrete/board-level)     ❌ No simulation during generation',
            fontsize=12, color=TEXT_COLOR, ha='center')
    ax.text(5, 0.8, '✅ CircuitGenie: values + specs + all analog + SPICE-in-the-loop',
            fontsize=14, color=ACCENT2, ha='center', fontweight='bold')
    
    add_frame(fig, 10)
    plt.close()

# ============================================================
# GENERATE ALL FRAMES
# ============================================================
print("Generating frames...")
print("  Title...")
draw_title()
print("  Circuit schematic...")
draw_circuit()
print("  Pin-level graph...")
draw_pin_graph()
print("  Eulerian concept...")
draw_eulerian_concept()
print("  Eulerian walk (animated)...")
draw_eulerian_walk()
print("  Tokenization...")
draw_tokenization()
print("  GPT generation (animated)...")
draw_gpt_generation()
print("  Summary...")
draw_summary()

# Save GIF
output_path = os.path.join(os.path.dirname(__file__), 'analoggenie_explained.gif')
print(f"Saving GIF with {len(frames)} frames to {output_path}...")

# Resize all frames to match
target_size = frames[0].size
frames_resized = []
for f in frames:
    if f.size != target_size:
        f = f.resize(target_size, Image.LANCZOS)
    frames_resized.append(f)

frames_resized[0].save(
    output_path,
    save_all=True,
    append_images=frames_resized[1:],
    duration=400,  # ms per frame
    loop=0
)
print(f"Done! GIF saved to {output_path}")
print(f"Total frames: {len(frames_resized)}")
