#!/usr/bin/env python3
"""Visualize ARCS GraphTransformer training curves."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# --- Load history ---
history_path = Path("checkpoints/arcs_graph_transformer/history.json")
with open(history_path) as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
train_ppl = [h["train_ppl"] for h in history]
val_ppl = [h["val_ppl"] for h in history]
val_acc = [h["val_accuracy"] for h in history]
val_value_acc = [h["val_value_acc"] for h in history]
val_struct_acc = [h["val_struct_acc"] for h in history]
lr = [h["lr"] for h in history]
time_per_epoch = [h["time"] for h in history]

best_epoch = epochs[np.argmin(val_loss)]
best_val = min(val_loss)

# --- Style ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"ARCS GraphTransformer (6.8M) — RWPE Training\n"
    f"Best val loss: {best_val:.4f} @ epoch {best_epoch}",
    fontsize=14, fontweight="bold", y=0.98,
)

# Color palette
c_train = "#2196F3"
c_val = "#F44336"
c_acc = "#4CAF50"
c_vacc = "#FF9800"
c_sacc = "#9C27B0"
c_lr = "#607D8B"

# 1) Loss curves
ax = axes[0, 0]
ax.plot(epochs, train_loss, color=c_train, linewidth=1.5, label="Train loss", alpha=0.8)
ax.plot(epochs, val_loss, color=c_val, linewidth=2, label="Val loss")
ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best @ {best_epoch}")
ax.axhline(best_val, color=c_val, linestyle=":", alpha=0.3)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Loss Curves")
ax.legend(fontsize=9)
ax.set_xlim(1, max(epochs))

# 2) Perplexity
ax = axes[0, 1]
ax.plot(epochs, train_ppl, color=c_train, linewidth=1.5, label="Train PPL", alpha=0.8)
ax.plot(epochs, val_ppl, color=c_val, linewidth=2, label="Val PPL")
ax.set_xlabel("Epoch")
ax.set_ylabel("Perplexity")
ax.set_title("Perplexity")
ax.legend(fontsize=9)
ax.set_xlim(1, max(epochs))

# 3) Accuracy breakdown
ax = axes[0, 2]
ax.plot(epochs, val_acc, color=c_acc, linewidth=2, label="Overall acc")
ax.plot(epochs, val_value_acc, color=c_vacc, linewidth=1.5, label="Value acc", alpha=0.8)
ax.plot(epochs, val_struct_acc, color=c_sacc, linewidth=1.5, label="Struct acc", alpha=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Validation Accuracy Breakdown")
ax.legend(fontsize=9)
ax.set_xlim(1, max(epochs))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

# 4) Learning rate schedule
ax = axes[1, 0]
ax.plot(epochs, lr, color=c_lr, linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("LR Schedule (Warmup + Cosine)")
ax.ticklabel_format(axis="y", style="scientific", scilimits=(-4, -4))
ax.set_xlim(1, max(epochs))

# 5) Train-val gap (overfitting indicator)
ax = axes[1, 1]
gap = [v - t for t, v in zip(train_loss, val_loss)]
ax.fill_between(epochs, 0, gap, alpha=0.3, color=c_val, label="Val - Train gap")
ax.plot(epochs, gap, color=c_val, linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss Gap")
ax.set_title("Generalization Gap (Overfitting Monitor)")
ax.legend(fontsize=9)
ax.set_xlim(1, max(epochs))

# 6) Time per epoch
ax = axes[1, 2]
ax.bar(epochs, time_per_epoch, color=c_lr, alpha=0.6, width=0.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Seconds")
ax.set_title(f"Time per Epoch (total: {sum(time_per_epoch)/3600:.1f}h)")
ax.set_xlim(0, max(epochs) + 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = Path("results/training_curves.png")
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
