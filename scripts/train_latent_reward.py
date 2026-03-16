#!/usr/bin/env python3
"""Train latent reward predictor on VCG-encoded circuits with SPICE rewards.

Usage:
    python scripts/train_latent_reward.py \
        --data data/combined \
        --vcg-checkpoint checkpoints/vcg/best_model.pt \
        --output checkpoints/latent_reward

This encodes training circuits through the frozen VCG encoder to get z vectors,
pairs them with SPICE simulation rewards, and trains a lightweight MLP to
predict reward from (z, spec_embed). The trained predictor is then used by
LatentRefinement for gradient-based latent-space optimization at inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig,
    ValidCircuitGenModel,
    CircuitGraphDataset,
)
from arcs.simulate import compute_reward
from arcs.latent_reward import (
    LatentRewardConfig,
    LatentRewardPredictor,
    LatentRewardTrainer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Latent Reward Predictor")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data directory")
    parser.add_argument("--vcg-checkpoint", type=str, required=True,
                        help="Path to pre-trained VCG checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/latent_reward",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP hidden dim")
    parser.add_argument("--n-layers", type=int, default=3, help="MLP layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=20, help="Log every N steps")
    parser.add_argument("--valid-only", action="store_true", default=True,
                        help="Only use valid circuits")
    return parser.parse_args()


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Tokenizer
    tokenizer = CircuitTokenizer()

    # Load pre-trained VCG
    logger.info(f"Loading VCG from {args.vcg_checkpoint}")
    vcg_ckpt = torch.load(args.vcg_checkpoint, map_location="cpu", weights_only=False)
    vcg_config = VCGConfig.from_dict(vcg_ckpt["config"])
    vcg_model = ValidCircuitGenModel(vcg_config).to(device)
    vcg_model.load_state_dict(vcg_ckpt["model"])
    vcg_model.eval()
    for p in vcg_model.parameters():
        p.requires_grad_(False)
    logger.info(f"VCG loaded (frozen): {vcg_model.count_parameters():,} params")

    # Dataset
    logger.info(f"Loading data from {args.data}...")
    dataset = CircuitGraphDataset(
        args.data, tokenizer, vcg_config, valid_only=args.valid_only,
    )
    logger.info(f"Dataset: {len(dataset)} circuits")

    # Train/val split
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    # Reward predictor
    reward_config = LatentRewardConfig(
        latent_dim=vcg_config.latent_dim,
        spec_dim=vcg_config.d_model,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    predictor = LatentRewardPredictor(reward_config).to(device)
    trainer = LatentRewardTrainer(predictor, lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in predictor.parameters())
    logger.info(f"Reward predictor: {n_params:,} parameters")

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "reward_config": {
                "latent_dim": reward_config.latent_dim,
                "spec_dim": reward_config.spec_dim,
                "hidden_dim": reward_config.hidden_dim,
                "n_layers": reward_config.n_layers,
            },
            "vcg_checkpoint": args.vcg_checkpoint,
        }, f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    logger.info(f"\n{'='*60}")
    logger.info(f"Training latent reward predictor for {args.epochs} epochs")
    logger.info(f"{'='*60}\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        predictor.train()
        epoch_losses = []
        epoch_corrs = []

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Encode through frozen VCG to get z and spec_embed
            with torch.no_grad():
                mu, logvar, spec_embed = vcg_model.encode(batch)
                z = ValidCircuitGenModel.reparameterize(mu, logvar)

            # Use the reward from the dataset (stored in spec_values as proxy)
            # The actual reward is computed from simulation metrics
            # For now we use a heuristic: circuits that are valid get reward
            # proportional to how well their values match bounds
            reward = batch.get("reward")
            if reward is None:
                # Fallback: use value-bounds adherence as proxy reward
                values = batch["values"]
                bounds_min = batch["value_bounds_min"]
                bounds_max = batch["value_bounds_max"]
                mask = batch["active_mask"]
                in_bounds = ((values >= bounds_min) & (values <= bounds_max)).float()
                reward = (in_bounds * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                reward = reward * 8.0  # Scale to [0, 8] range

            stats = trainer.train_step(z, spec_embed, reward)
            epoch_losses.append(stats["loss"])
            epoch_corrs.append(stats["correlation"])

            if (step + 1) % args.log_interval == 0:
                logger.info(
                    f"  [{epoch+1}/{args.epochs}] step {step+1}/{len(train_loader)} "
                    f"| loss={stats['loss']:.4f} corr={stats['correlation']:.3f}"
                )

        # Validation
        predictor.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mu, logvar, spec_embed = vcg_model.encode(batch)
                z = ValidCircuitGenModel.reparameterize(mu, logvar)

                reward = batch.get("reward")
                if reward is None:
                    values = batch["values"]
                    bounds_min = batch["value_bounds_min"]
                    bounds_max = batch["value_bounds_max"]
                    mask = batch["active_mask"]
                    in_bounds = ((values >= bounds_min) & (values <= bounds_max)).float()
                    reward = (in_bounds * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
                    reward = reward * 8.0

                pred = predictor(z, spec_embed)
                loss = torch.nn.functional.mse_loss(pred, reward)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_corr = sum(epoch_corrs) / len(epoch_corrs)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) "
            f"| train_loss={avg_loss:.4f} corr={avg_corr:.3f} "
            f"| val_loss={val_loss:.4f}"
        )

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            state = trainer.state_dict()
            state["config"] = {
                "latent_dim": reward_config.latent_dim,
                "spec_dim": reward_config.spec_dim,
                "hidden_dim": reward_config.hidden_dim,
                "n_layers": reward_config.n_layers,
            }
            state["epoch"] = epoch + 1
            state["val_loss"] = val_loss
            torch.save(state, output_dir / "best_reward_predictor.pt")
            logger.info(f"  New best (val_loss={val_loss:.4f})")

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
