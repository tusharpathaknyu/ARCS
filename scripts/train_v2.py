#!/usr/bin/env python3
"""
CircuitGenie Phase 2: Two-stage training.

Stage 1: Cross-entropy with 5x value weight + spec-param consistency loss
Stage 2: REINFORCE fine-tuning with SPICE simulation reward
"""

import copy
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from circuitgenie.data.generator import load_mlentry_data
from circuitgenie.data.spice_templates import Topology
from circuitgenie.tokenizer.tokenizer import CircuitTokenizer
from circuitgenie.model.config import CircuitGenieConfig
from circuitgenie.model.transformer import CircuitGenieModel
from circuitgenie.training.dataset import create_dataloaders
from circuitgenie.training.trainer import Trainer
from circuitgenie.training.rl_trainer import RLTrainer
from circuitgenie.training.evaluate import (
    TEST_SPECS, evaluate_model, print_evaluation_report,
)


def main():
    print("=" * 60)
    print("CircuitGenie Phase 2: Two-Stage Training")
    print("=" * 60)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ================================================================
    # Stage 1: Cross-Entropy with Improved Loss
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Cross-Entropy Training (5x value weight + consistency)")
    print("=" * 60)

    # Load ALL data (5000 per topology = 35K total)
    print("\nLoading data (all 35K samples)...")
    samples = load_mlentry_data(samples_per_topology=5000, seed=42)

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

    # Trainer (5x value weight via default)
    checkpoint_dir = project_root / "checkpoints_v2"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        weight_decay=0.01,
        warmup_steps=200,  # More warmup for larger dataset
        max_grad_norm=1.0,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        value_weight=5.0,
    )

    # Train Stage 1
    print("\n" + "-" * 40)
    n_epochs_stage1 = 50
    history1 = trainer.train(n_epochs=n_epochs_stage1, log_every=5)

    print(f"\nStage 1 complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Final train loss: {history1['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history1['val_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history1['val_accuracy'][-1]:.1%}")
    print(f"Final val value accuracy: {history1['val_value_accuracy'][-1]:.1%}")

    # Save Stage 1 checkpoint explicitly
    trainer.save_checkpoint(checkpoint_dir / "stage1_final.pt")

    # Evaluate Stage 1
    print("\n--- Stage 1 Evaluation ---")
    metrics1 = evaluate_model(
        model, tokenizer,
        n_samples_per_spec=5,
        temperature=0.7,
        top_k=15,
        device=device,
        run_spice=True,
    )
    print_evaluation_report(metrics1)

    # ================================================================
    # Stage 2: REINFORCE Fine-Tuning
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: REINFORCE Fine-Tuning with SPICE Reward")
    print("=" * 60)

    # Create reference model (frozen copy of CE-trained model)
    ref_model = CircuitGenieModel(config)
    ref_model.load_state_dict(model.state_dict())

    # Build RL test specs (reuse evaluation TEST_SPECS)
    rl_test_specs = TEST_SPECS

    rl_trainer = RLTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        test_specs=rl_test_specs,
        lr=1e-5,
        kl_coeff=0.1,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        baseline_decay=0.99,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        use_spice=True,
        temperature=0.8,
        top_k=20,
    )

    # Train Stage 2
    n_steps_stage2 = 5000
    history2 = rl_trainer.train(
        n_steps=n_steps_stage2,
        log_every=50,
        eval_every=500,
        n_eval_samples=5,
    )

    # ================================================================
    # Final Evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION: CE + RL Model")
    print("=" * 60)

    # Load best RL model
    best_rl_path = checkpoint_dir / "best_rl_model.pt"
    if best_rl_path.exists():
        rl_trainer.load_checkpoint(best_rl_path)
        print("Loaded best RL model")

    metrics_final = evaluate_model(
        model, tokenizer,
        n_samples_per_spec=10,
        temperature=0.7,
        top_k=15,
        device=device,
        run_spice=True,
    )
    print_evaluation_report(metrics_final)

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Stage 1 (CE)':>15} {'Stage 2 (RL)':>15}")
    print("-" * 55)
    print(f"{'Decode rate':<25} {metrics1['decode_rate']:>14.1%} {metrics_final['decode_rate']:>14.1%}")
    print(f"{'Structural validity':<25} {metrics1['structural_validity']:>14.1%} {metrics_final['structural_validity']:>14.1%}")
    print(f"{'V_in match rate':<25} {metrics1['v_in_match_rate']:>14.1%} {metrics_final['v_in_match_rate']:>14.1%}")
    print(f"{'Duty reasonable rate':<25} {metrics1['duty_reasonable_rate']:>14.1%} {metrics_final['duty_reasonable_rate']:>14.1%}")

    if metrics1['spice_success_rate'] is not None:
        print(f"{'SPICE success rate':<25} {metrics1['spice_success_rate']:>14.1%} {metrics_final['spice_success_rate']:>14.1%}")
    if metrics1['mean_v_out_error'] is not None:
        s1_err = metrics1['mean_v_out_error']
    else:
        s1_err = float('nan')
    if metrics_final['mean_v_out_error'] is not None:
        s2_err = metrics_final['mean_v_out_error']
    else:
        s2_err = float('nan')
    print(f"{'Mean V_out error':<25} {s1_err:>14.1%} {s2_err:>14.1%}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
