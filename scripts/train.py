#!/usr/bin/env python3
"""Train CircuitGenie model on Phase 1 power converter data."""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from circuitgenie.data.generator import load_mlentry_data
from circuitgenie.tokenizer.tokenizer import CircuitTokenizer
from circuitgenie.model.config import CircuitGenieConfig
from circuitgenie.model.transformer import CircuitGenieModel
from circuitgenie.training.dataset import create_dataloaders
from circuitgenie.training.trainer import Trainer


def main():
    print("=" * 60)
    print("CircuitGenie Phase 1: Training")
    print("=" * 60)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    samples = load_mlentry_data(samples_per_topology=2000, seed=42)

    # Tokenizer
    tokenizer = CircuitTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create dataloaders
    batch_size = 64
    train_loader, val_loader = create_dataloaders(
        samples, tokenizer,
        batch_size=batch_size,
        val_split=0.1,
        seed=42,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    config = CircuitGenieConfig()
    model = CircuitGenieModel(config)
    print(f"\nModel: {model.count_parameters():,} parameters")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, d_ff={config.d_ff}")

    # Trainer
    checkpoint_dir = project_root / "checkpoints"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
    )

    # Train
    print("\n" + "=" * 60)
    n_epochs = 100
    history = trainer.train(n_epochs=n_epochs, log_every=5)

    # Summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.1%}")
    print(f"Final val value accuracy: {history['val_value_accuracy'][-1]:.1%}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
