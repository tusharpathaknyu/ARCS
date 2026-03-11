#!/usr/bin/env python3
"""Train ValidCircuitGen: Constrained VAE for circuit generation.

Usage:
    python scripts/train_vcg.py --data data/combined --epochs 100

This trains the Direction 5 (ValidCircuitGen) architecture — a VAE that
generates entire circuit graphs in one shot with formal validity guarantees
via constraint projection and Lagrangian training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig,
    ValidCircuitGenModel,
    CircuitGraphDataset,
    LagrangianVAETrainer,
    check_circuit_validity,
    graph_to_token_sequence,
    TOPOLOGY_TO_IDX,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ValidCircuitGen (Direction 5)")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data directory")
    parser.add_argument("--output", type=str, default="checkpoints/vcg", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-lr", type=float, default=0.01, help="Lagrange multiplier LR")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent space dimension")
    parser.add_argument("--d-model", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-encoder-layers", type=int, default=4, help="Encoder layers")
    parser.add_argument("--beta-kl", type=float, default=0.1, help="KL weight (β-VAE)")
    parser.add_argument("--n-projection-steps", type=int, default=20, help="Projection steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--eval-samples", type=int, default=50, help="Samples for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--valid-only", action="store_true", help="Only use valid circuits")
    return parser.parse_args()


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate CircuitGraph dicts into batched tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def evaluate_generation(
    model: ValidCircuitGenModel,
    dataset: CircuitGraphDataset,
    tokenizer: CircuitTokenizer,
    n_samples: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Evaluate generation quality by sampling and checking validity."""
    model.eval()

    # Collect unique topologies from dataset
    topo_samples: dict[str, dict] = {}
    for i in range(min(len(dataset), 500)):
        item = dataset[i]
        topo_idx = item["topology_idx"].item()
        if topo_idx not in topo_samples:
            topo_samples[topo_idx] = item

    if not topo_samples:
        return {"validity_rate": 0.0}

    total_valid = 0
    total_generated = 0
    all_violations = []

    for topo_idx, template_item in topo_samples.items():
        # Generate circuits for this topology
        per_topo = max(1, n_samples // len(topo_samples))

        graphs, stats = model.generate(
            spec_types=template_item["spec_types"].unsqueeze(0).to(device),
            spec_values=template_item["spec_values"].unsqueeze(0).to(device),
            spec_mask=template_item["spec_mask"].unsqueeze(0).to(device),
            topology_idx=template_item["topology_idx"].unsqueeze(0).to(device),
            active_mask=template_item["active_mask"].unsqueeze(0).to(device),
            bounds_min=template_item["value_bounds_min"].unsqueeze(0).to(device),
            bounds_max=template_item["value_bounds_max"].unsqueeze(0).to(device),
            n_samples=per_topo,
            use_projection=True,
        )

        for graph in graphs:
            validity = check_circuit_validity(graph)
            if validity["valid"]:
                total_valid += 1
            total_generated += 1

    validity_rate = total_valid / max(total_generated, 1)
    return {
        "validity_rate": validity_rate,
        "total_generated": total_generated,
        "total_valid": total_valid,
    }


def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = CircuitTokenizer()

    # Config
    config = VCGConfig(
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        beta_kl=args.beta_kl,
        n_projection_steps=args.n_projection_steps,
    )

    # Dataset
    print(f"Loading data from {args.data}...")
    dataset = CircuitGraphDataset(
        args.data, tokenizer, config, valid_only=args.valid_only,
    )

    # Train/val split (90/10)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )

    # Model
    model = ValidCircuitGenModel(config).to(device)
    n_params = model.count_parameters()
    print(f"ValidCircuitGen: {n_params:,} parameters")
    print(f"  Parameter groups: {model.count_parameters_by_group()}")

    # Trainer
    steps_per_epoch = len(train_loader)
    max_steps = args.epochs * steps_per_epoch
    trainer = LagrangianVAETrainer(
        model,
        lr=args.lr,
        lambda_lr=args.lambda_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        max_steps=max_steps,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        trainer.load_state_dict(ckpt)
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at epoch {start_epoch}, step {trainer.step_count}")

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Training loop
    best_val_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"Training ValidCircuitGen for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()

        epoch_stats: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            stats = trainer.train_step(batch)

            # Accumulate stats
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    if k not in epoch_stats:
                        epoch_stats[k] = []
                    epoch_stats[k].append(v)

            # Log
            if (step + 1) % args.log_interval == 0:
                total = stats.get("loss/total", 0)
                recon = stats.get("loss/recon", 0)
                kl = stats.get("loss/kl", 0)
                constr = stats.get("loss/constraint", 0)
                lam = stats.get("lambda/mean", 0)
                print(
                    f"  [{epoch+1}/{args.epochs}] step {step+1}/{steps_per_epoch} "
                    f"| total={total:.4f} recon={recon:.4f} kl={kl:.4f} "
                    f"constr={constr:.4f} λ={lam:.3f}"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_stats = {k: sum(v) / len(v) for k, v in epoch_stats.items()}

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, _, _, total_loss, _ = model(batch)
                val_losses.append(total_loss.item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)

        print(
            f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) "
            f"| train_loss={avg_stats.get('loss/total', 0):.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| kl={avg_stats.get('loss/kl', 0):.4f} "
            f"| λ_mean={avg_stats.get('lambda/mean', 0):.3f}"
        )

        # Evaluate generation periodically
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            print("  Evaluating generation quality...")
            gen_stats = evaluate_generation(
                model, dataset, tokenizer,
                n_samples=args.eval_samples, device=device,
            )
            print(
                f"  Validity: {gen_stats['validity_rate']*100:.1f}% "
                f"({gen_stats['total_valid']}/{gen_stats['total_generated']})"
            )

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % args.save_interval == 0 or is_best or epoch == args.epochs - 1:
            ckpt = trainer.state_dict()
            ckpt["epoch"] = epoch + 1
            ckpt["val_loss"] = val_loss
            ckpt["config"] = config.to_dict()
            ckpt["n_params"] = n_params

            if is_best:
                torch.save(ckpt, output_dir / "best_model.pt")
                print(f"  ✓ New best model (val_loss={val_loss:.4f})")

            if (epoch + 1) % args.save_interval == 0:
                torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch+1}.pt")

        print()

    # Final save
    final_ckpt = trainer.state_dict()
    final_ckpt["epoch"] = args.epochs
    final_ckpt["val_loss"] = val_loss
    final_ckpt["config"] = config.to_dict()
    torch.save(final_ckpt, output_dir / "final_model.pt")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
