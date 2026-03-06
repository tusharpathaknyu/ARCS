"""Learned Reward Model (Verifier) for ARCS.

A lightweight transformer encoder that predicts SPICE simulation reward
from circuit token sequences, enabling oracle-free Best-of-N ranking.

Key insight from Phase 9: model confidence (log-prob) scales monotonically
with N but SPICE reward peaks at N=3 then plateaus.  A learned reward model
bridges this gap by predicting simulation quality without running SPICE.

Architecture:
  - Shared vocab embedding (optionally initialized from generator weights)
  - 2-layer transformer encoder with bidirectional attention
  - Mean-pool over non-padding tokens → d_model
  - 2-layer MLP → scalar reward prediction

Training data:
  - 53K circuits in data/combined/*.jsonl with full SPICE metrics
  - Reward labels computed from metrics using domain-aware reward function

References:
  [14] Snell et al., "Scaling LLM Test-Time Compute Optimally Can
       Be More Effective than Scaling Model Parameters", 2024.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arcs.tokenizer import CircuitTokenizer
from arcs.templates import _TIER1_NAMES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RewardModelConfig:
    """Reward model hyperparameters."""

    vocab_size: int = 686           # Must match CircuitTokenizer
    max_seq_len: int = 128          # Max token sequence length
    d_model: int = 128              # Hidden dimension (smaller than generator)
    n_heads: int = 4                # Attention heads
    n_layers: int = 2               # Encoder layers (lightweight)
    d_ff: int = 512                 # FFN inner dimension
    dropout: float = 0.1
    pad_id: int = 0

    # MLP head
    mlp_hidden: int = 256           # Hidden dim of reward MLP
    reward_range: tuple[float, float] = (0.0, 8.0)  # Output clamp range

    # Training
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 30
    patience: int = 5               # Early stopping patience
    val_fraction: float = 0.1       # Validation split

    @classmethod
    def tiny(cls) -> "RewardModelConfig":
        """~200K param config for fast experiments."""
        return cls(d_model=64, n_heads=2, n_layers=1, d_ff=256, mlp_hidden=128)

    @classmethod
    def small(cls) -> "RewardModelConfig":
        """~800K param config (default)."""
        return cls()

    @classmethod
    def medium(cls) -> "RewardModelConfig":
        """~3M param config for high accuracy."""
        return cls(d_model=256, n_heads=4, n_layers=3, d_ff=1024, mlp_hidden=512)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reward_range"] = list(d["reward_range"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RewardModelConfig":
        if "reward_range" in d and isinstance(d["reward_range"], list):
            d["reward_range"] = tuple(d["reward_range"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Reward computation from JSONL data
# ---------------------------------------------------------------------------


def compute_reward_from_sample(
    topology: str,
    metrics: dict[str, float],
    valid: bool,
    struct_bonus: float = 1.0,
) -> float:
    """Compute scalar reward directly from JSONL sample fields.

    Mirrors simulate.compute_reward() but works on raw data instead of
    requiring DecodedCircuit + SimulationOutcome objects.

    Returns:
        Reward in [0, 8].
    """
    reward = struct_bonus  # All JSONL samples have valid structure

    if not valid:
        return reward

    # +1.0 for sim convergence
    reward += 1.0

    topo_lower = topology.lower().replace("-", "_")

    if topo_lower in _TIER1_NAMES:
        reward += _power_reward_from_metrics(metrics)
    else:
        reward += _signal_reward_from_metrics(metrics, topo_lower)

    return reward


def _power_reward_from_metrics(m: dict[str, float]) -> float:
    """Power converter reward from raw metrics (max 6.0)."""
    reward = 0.0

    # Vout accuracy: 3.0 × max(0, 1 - error/10)
    verr = m.get("vout_error_pct", 100)
    reward += 3.0 * max(0.0, 1.0 - verr / 10.0)

    # Efficiency: 2.0 × efficiency
    eff = m.get("efficiency", 0)
    reward += 2.0 * max(0.0, min(1.0, eff))

    # Low ripple: 1.0 × max(0, 1 - ripple×10)
    rip = m.get("ripple_ratio", 1.0)
    reward += 1.0 * max(0.0, 1.0 - rip * 10.0)

    return reward


def _signal_reward_from_metrics(m: dict[str, float], topology: str) -> float:
    """Signal circuit reward from raw metrics (max 6.0)."""
    reward = 0.0

    amp_types = {
        "inverting_amp", "noninverting_amp",
        "instrumentation_amp", "differential_amp",
    }
    filter_types = {
        "sallen_key_lowpass", "sallen_key_highpass", "sallen_key_bandpass",
    }
    osc_types = {"wien_bridge", "colpitts"}

    if topology in amp_types:
        gain_db = m.get("gain_db", m.get("gain_dc"))
        if gain_db is not None and abs(gain_db) <= 120:
            reward += 3.0
            if abs(gain_db) > 0:
                reward += min(2.0, abs(gain_db) / 30.0)
            if m.get("bw_3db") is not None and m["bw_3db"] > 0:
                reward += 1.0

    elif topology in filter_types:
        gain_dc = m.get("gain_dc")
        if gain_dc is not None:
            reward += 2.0
        bw = m.get("bw_3db")
        if bw is not None and bw > 0:
            reward += 3.0
        if gain_dc is not None and gain_dc > -6:
            reward += 1.0

    elif topology in osc_types:
        vosc = m.get("vosc_pp", 0)
        if vosc >= 0.01:
            reward += 3.0
        if 0.1 <= vosc <= 20:
            reward += 2.0
        if m.get("f_peak") is not None and m["f_peak"] > 0:
            reward += 1.0

    return reward


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CircuitRewardDataset(Dataset):
    """Dataset of (token_ids, reward) pairs from JSONL simulation data.

    Loads circuits from data/combined/*.jsonl, tokenizes each, and computes
    a reward label from the simulation metrics.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: CircuitTokenizer,
        max_seq_len: int = 128,
        topologies: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], float]] = []

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Import CircuitSample for tokenization
        from arcs.datagen import CircuitSample

        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        if topologies:
            topo_set = {t.lower() for t in topologies}
            jsonl_files = [f for f in jsonl_files if f.stem.lower() in topo_set]

        for fpath in jsonl_files:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        sample = CircuitSample.from_dict(raw)

                        # Tokenize
                        token_ids = tokenizer.encode_circuit_sample(sample)

                        # Compute reward label
                        reward = compute_reward_from_sample(
                            sample.topology,
                            sample.metrics,
                            sample.valid,
                        )

                        self.samples.append((token_ids, reward))
                    except Exception:
                        continue  # Skip malformed samples

        if not self.samples:
            raise ValueError(f"No valid samples loaded from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, reward = self.samples[idx]

        # Pad/truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
        else:
            token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
        )

    @property
    def reward_stats(self) -> dict[str, float]:
        """Basic statistics about reward distribution."""
        rewards = [r for _, r in self.samples]
        return {
            "count": len(rewards),
            "mean": sum(rewards) / len(rewards),
            "min": min(rewards),
            "max": max(rewards),
            "std": (
                sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards)
                / len(rewards)
            )
            ** 0.5,
            "frac_zero": sum(1 for r in rewards if r < 1.5) / len(rewards),
        }


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class BidirectionalAttention(nn.Module):
    """Multi-head bidirectional self-attention (encoder-style)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if pad_mask is not None:
            # pad_mask: (B, T), True = pad → mask out
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class EncoderBlock(nn.Module):
    """Transformer encoder block with pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        # SwiGLU FFN
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pad_mask)
        h = self.ln2(x)
        x = x + self.ffn_drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))
        return x


class CircuitRewardModel(nn.Module):
    """Transformer encoder that predicts SPICE reward from token sequences.

    Architecture:
        Token IDs → Embedding + PosEmb → N EncoderBlocks → MeanPool → MLP → reward

    The model uses bidirectional attention (unlike the causal generator)
    because the full circuit is available at scoring time.
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        # Reward prediction MLP
        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: token IDs → predicted reward.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) True for valid tokens, False for padding

        Returns:
            (B,) predicted rewards (unclamped during training for gradient flow)
        """
        B, T = input_ids.shape

        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(x)

        # Build padding mask for attention (True = pad)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = ~attention_mask  # Invert: True → padding positions

        for block in self.blocks:
            x = block(x, pad_mask)

        x = self.ln_f(x)

        # Mean-pool over non-padding tokens
        if attention_mask is not None:
            mask_f = attention_mask.float().unsqueeze(-1)  # (B, T, 1)
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            pooled = x.mean(dim=1)

        # Predict reward (no clamp during training — handled in predict())
        reward = self.reward_head(pooled).squeeze(-1)  # (B,)

        return reward

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convenience: predict rewards with no_grad, eval mode, and clamping."""
        self.eval()
        with torch.no_grad():
            mask = (input_ids != self.config.pad_id)
            reward = self.forward(input_ids, attention_mask=mask)
            lo, hi = self.config.reward_range
            return reward.clamp(lo, hi)

    def load_generator_embeddings(self, generator_state_dict: dict) -> int:
        """Initialize token embeddings from a trained generator model.

        Transfer learning: reuse the generator's learned token representations
        as a warm start for the reward model.

        Returns:
            Number of parameters transferred.
        """
        gen_emb = generator_state_dict.get("tok_emb.weight")
        if gen_emb is None:
            return 0

        # Only transfer if dimensions match
        if gen_emb.shape[1] != self.config.d_model:
            # Project down if generator has larger d_model
            with torch.no_grad():
                # SVD-based projection to smaller dimension
                U, S, Vh = torch.linalg.svd(gen_emb, full_matrices=False)
                projected = U[:, : self.config.d_model] * S[: self.config.d_model]
                self.tok_emb.weight.data.copy_(projected[: self.config.vocab_size])
            return self.tok_emb.weight.numel()

        with torch.no_grad():
            n = min(gen_emb.shape[0], self.config.vocab_size)
            self.tok_emb.weight.data[:n] = gen_emb[:n]
        return n * self.config.d_model

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float                # Mean absolute error
    val_correlation: float        # Pearson correlation
    val_rank_correlation: float   # Spearman rank correlation (approx)
    lr: float
    time_s: float


class RewardModelTrainer:
    """Train the reward model on SPICE simulation data.

    Features:
        - Adam with cosine annealing LR schedule
        - Huber loss for robust regression
        - Early stopping on validation MAE
        - Optional embedding transfer from generator
    """

    def __init__(
        self,
        config: RewardModelConfig,
        device: torch.device | str = "cpu",
        generator_checkpoint: str | Path | None = None,
    ):
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device

        # Build model
        self.model = CircuitRewardModel(config).to(self.device)

        # Optional embedding transfer
        if generator_checkpoint is not None:
            ckpt = torch.load(
                generator_checkpoint, map_location=self.device, weights_only=True
            )
            state = ckpt.get("model_state_dict", ckpt)
            n_transferred = self.model.load_generator_embeddings(state)
            if n_transferred > 0:
                print(f"  Transferred {n_transferred:,} embedding params from generator")

        # Optimizer + scheduler (set up in train())
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler | None = None

        self.history: list[TrainingMetrics] = []

    def train(
        self,
        dataset: CircuitRewardDataset,
        verbose: bool = True,
    ) -> CircuitRewardModel:
        """Train the reward model and return the best model.

        Splits dataset into train/val, trains with early stopping.
        """
        config = self.config

        # Train/val split
        n_val = max(1, int(len(dataset) * config.val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
        )

        if verbose:
            print(f"  Train: {n_train}, Val: {n_val}")
            print(f"  Model: {self.model.count_parameters():,} params")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6,
        )

        # Training loop
        best_val_mae = float("inf")
        best_state = None
        patience_counter = 0
        loss_fn = nn.HuberLoss(delta=1.0)

        for epoch in range(1, config.epochs + 1):
            t0 = time.perf_counter()

            # --- Train ---
            self.model.train()
            train_losses = []
            for ids, rewards in train_loader:
                ids = ids.to(self.device)
                rewards = rewards.to(self.device)
                mask = (ids != config.pad_id)

                pred = self.model(ids, attention_mask=mask)
                loss = loss_fn(pred, rewards)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            self.scheduler.step()

            # --- Validate ---
            val_metrics = self._validate(val_loader, loss_fn)

            dt = time.perf_counter() - t0
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=sum(train_losses) / max(len(train_losses), 1),
                val_loss=val_metrics["loss"],
                val_mae=val_metrics["mae"],
                val_correlation=val_metrics["correlation"],
                val_rank_correlation=val_metrics["rank_correlation"],
                lr=self.scheduler.get_last_lr()[0],
                time_s=dt,
            )
            self.history.append(metrics)

            if verbose:
                print(
                    f"  Epoch {epoch:3d} | "
                    f"train_loss={metrics.train_loss:.4f} | "
                    f"val_mae={metrics.val_mae:.4f} | "
                    f"r={metrics.val_correlation:.3f} | "
                    f"ρ={metrics.val_rank_correlation:.3f} | "
                    f"{dt:.1f}s"
                )

            # Early stopping
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self.model

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
        loss_fn: nn.Module,
    ) -> dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        losses = []
        all_preds: list[float] = []
        all_targets: list[float] = []

        for ids, rewards in loader:
            ids = ids.to(self.device)
            rewards = rewards.to(self.device)
            mask = (ids != self.config.pad_id)

            pred = self.model(ids, attention_mask=mask)
            loss = loss_fn(pred, rewards)
            losses.append(loss.item())

            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(rewards.cpu().tolist())

        mae = sum(abs(p - t) for p, t in zip(all_preds, all_targets)) / max(
            len(all_preds), 1
        )

        correlation = _pearson(all_preds, all_targets)
        rank_corr = _spearman_approx(all_preds, all_targets)

        return {
            "loss": sum(losses) / max(len(losses), 1),
            "mae": mae,
            "correlation": correlation,
            "rank_correlation": rank_corr,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, config, and training history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
                "history": [asdict(m) for m in self.history],
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> tuple["RewardModelTrainer", CircuitRewardModel]:
        """Load a trained reward model from checkpoint."""
        device = torch.device(device) if isinstance(device, str) else device
        ckpt = torch.load(path, map_location=device, weights_only=True)
        config = RewardModelConfig.from_dict(ckpt["config"])
        trainer = cls(config, device=device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.history = [
            TrainingMetrics(**m) for m in ckpt.get("history", [])
        ]
        return trainer, trainer.model


# ---------------------------------------------------------------------------
# Best-of-N integration
# ---------------------------------------------------------------------------


class RewardModelRanker:
    """Rank Best-of-N candidates using a learned reward model.

    Drop-in replacement for confidence-based ranking.
    """

    def __init__(
        self,
        reward_model: CircuitRewardModel,
        tokenizer: CircuitTokenizer,
        device: torch.device | str = "cpu",
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.reward_model.to(self.device)
        self.reward_model.eval()

    @torch.no_grad()
    def score_candidates(
        self,
        candidates: list[Any],
    ) -> list[tuple[Any, float]]:
        """Score a list of ScoredCandidate objects with predicted reward.

        Args:
            candidates: ScoredCandidate objects from BestOfNGenerator.

        Returns:
            List of (candidate, predicted_reward) tuples, sorted by
            predicted reward descending.
        """
        if not candidates:
            return []

        max_len = self.reward_model.config.max_seq_len

        # Batch tokenize
        batch_ids = []
        for c in candidates:
            ids = c.tokens[:max_len]
            ids = ids + [0] * (max_len - len(ids))
            batch_ids.append(ids)

        ids_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        scores = self.reward_model.predict(ids_tensor)

        # Build scored list
        scored = list(zip(candidates, scores.cpu().tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update ranks
        for i, (c, _) in enumerate(scored):
            c.rank = i

        return scored

    def rank_candidates(
        self,
        candidates: list[Any],
    ) -> list[Any]:
        """Rank candidates by predicted reward (highest first).

        Returns ranked candidate list (same interface as
        bestofn.rank_candidates).
        """
        scored = self.score_candidates(candidates)
        return [c for c, _ in scored]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return cov / (sx * sy)


def _spearman_approx(x: list[float], y: list[float]) -> float:
    """Approximate Spearman rank correlation (no scipy dependency)."""
    if len(x) < 3:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda iv: iv[1])
        ranks = [0.0] * len(vals)
        for rank, (idx, _) in enumerate(indexed):
            ranks[idx] = float(rank)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    return _pearson(rx, ry)


def evaluate_reward_model(
    model: CircuitRewardModel,
    dataset: CircuitRewardDataset,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> dict[str, float]:
    """Evaluate a trained reward model on a dataset.

    Returns:
        Dict with 'mae', 'mse', 'correlation', 'rank_correlation',
        'within_1', 'within_0.5'.
    """
    device = torch.device(device) if isinstance(device, str) else device
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[float] = []
    targets: list[float] = []

    with torch.no_grad():
        for ids, rewards in loader:
            ids = ids.to(device)
            mask = (ids != model.config.pad_id)
            pred = model(ids, attention_mask=mask)
            preds.extend(pred.cpu().tolist())
            targets.extend(rewards.tolist())

    n = len(preds)
    mae = sum(abs(p - t) for p, t in zip(preds, targets)) / n
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / n
    within_1 = sum(1 for p, t in zip(preds, targets) if abs(p - t) < 1.0) / n
    within_05 = sum(1 for p, t in zip(preds, targets) if abs(p - t) < 0.5) / n

    return {
        "mae": mae,
        "mse": mse,
        "rmse": mse ** 0.5,
        "correlation": _pearson(preds, targets),
        "rank_correlation": _spearman_approx(preds, targets),
        "within_1": within_1,
        "within_0.5": within_05,
        "n_samples": n,
    }
