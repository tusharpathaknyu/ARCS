#!/usr/bin/env python3
"""
CircuitGenie v3: Eulerian walk representation training.

Two-stage training with Eulerian walk augmented data:
  Stage 1: Cross-entropy with 5x value weight (175K augmented sequences)
  Stage 2: REINFORCE with SPICE reward
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from circuitgenie.data.generator import load_mlentry_data
from circuitgenie.data.spice_templates import Topology
from circuitgenie.tokenizer.tokenizer_v2 import CircuitTokenizerV2
from circuitgenie.model.config import CircuitGenieConfigV2
from circuitgenie.model.transformer import CircuitGenieModel
from circuitgenie.training.dataset_v2 import create_dataloaders_v2
from circuitgenie.training.trainer import Trainer


def main():
    print("=" * 60)
    print("CircuitGenie v3: Eulerian Walk Training")
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
    # Load data
    # ================================================================
    print("\nLoading data (all 35K base samples)...")
    samples = load_mlentry_data(samples_per_topology=5000, seed=42)

    # Tokenizer v2
    tokenizer = CircuitTokenizerV2(max_seq_len=64)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max seq len: {tokenizer.max_seq_len}")

    # Quick test encode
    s0 = samples[0]
    tokens = tokenizer.encode(s0, walk_seed=42)
    print(f"\nExample {s0.topology.name} sequence ({len(tokens)} tokens):")
    print(f"  {tokenizer.to_readable(tokens)[:120]}...")

    # Create dataloaders with 5x walk augmentation
    n_walks = 5
    batch_size = 64
    train_loader, val_loader = create_dataloaders_v2(
        samples, tokenizer,
        batch_size=batch_size,
        val_split=0.1,
        n_walks=n_walks,
        seed=42,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ================================================================
    # Stage 1: Cross-Entropy Training
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 1: Cross-Entropy Training (Eulerian walk + 5x value weight)")
    print("=" * 60)

    config = CircuitGenieConfigV2()
    model = CircuitGenieModel(config)
    n_params = model.count_parameters()
    print(f"\nModel: {n_params:,} parameters")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, d_ff={config.d_ff}")

    checkpoint_dir = project_root / "checkpoints_v3"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=3e-4,
        weight_decay=0.01,
        warmup_steps=300,
        max_grad_norm=1.0,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        value_weight=5.0,
    )

    # Train
    n_epochs = 30  # Fewer epochs because 5x more data from augmentation
    print(f"\nTraining: {n_epochs} epochs")
    history = trainer.train(n_epochs=n_epochs, log_every=5)

    print(f"\nStage 1 complete!")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.1%}")
    print(f"Final val value accuracy: {history['val_value_accuracy'][-1]:.1%}")

    # Save stage 1
    trainer.save_checkpoint(checkpoint_dir / "stage1_final.pt")

    # ================================================================
    # Quick evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    model.eval()
    from circuitgenie.tokenizer.vocabulary_v2 import (
        BOS_ID, SEP_ID, WALK_END_ID, EOS_ID as EOS_V2,
        TOKEN_TO_ID_V2, SPEC_KEY_TO_INFO_V2,
        value_to_token_id_v2,
    )
    from circuitgenie.tokenizer.sequence_v2 import (
        SPEC_ORDER, tokens_to_circuit_v2, _identify_topology_from_walk,
    )
    from circuitgenie.data.spice_templates import calculate_expected_vout
    from circuitgenie.data.simulator import run_simulation

    # Test specs
    test_specs = [
        {'name': 'Buck 12V→5V 2A', 'specs': {'v_in': 12, 'v_out': 5, 'i_out': 2, 'ripple_pct': 2.0, 'efficiency': 0.92}},
        {'name': 'Boost 5V→12V 0.5A', 'specs': {'v_in': 5, 'v_out': 12, 'i_out': 0.5, 'ripple_pct': 1.5, 'efficiency': 0.90}},
        {'name': 'Buck 24V→12V 1A', 'specs': {'v_in': 24, 'v_out': 12, 'i_out': 1, 'ripple_pct': 1.0, 'efficiency': 0.95}},
        {'name': 'Flyback 48V→5V 1A', 'specs': {'v_in': 48, 'v_out': 5, 'i_out': 1, 'ripple_pct': 3.0, 'efficiency': 0.85}},
        {'name': 'SEPIC 12V→15V 0.5A', 'specs': {'v_in': 12, 'v_out': 15, 'i_out': 0.5, 'ripple_pct': 2.0, 'efficiency': 0.88}},
    ]

    def build_spec_prefix_v2(specs):
        tokens = [BOS_ID]
        for spec_key in SPEC_ORDER:
            spec_token_name, qtype = SPEC_KEY_TO_INFO_V2[spec_key]
            spec_token_id = TOKEN_TO_ID_V2[spec_token_name]
            val_token_id = value_to_token_id_v2(specs[spec_key], qtype)
            tokens.extend([spec_token_id, val_token_id])
        tokens.append(SEP_ID)
        return tokens

    n_samples = 10
    total_decoded = 0
    total_generated = 0
    total_spice_ok = 0
    vout_errors = []

    for tc in test_specs:
        name = tc['name']
        specs = tc['specs']
        prefix = build_spec_prefix_v2(specs)
        prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

        decoded_count = 0
        spice_count = 0
        tc_vout_errors = []

        for _ in range(n_samples):
            total_generated += 1
            generated = model.generate(
                prefix_tensor, max_new_tokens=64 - len(prefix),
                temperature=0.7, top_k=15,
            )
            token_ids = generated[0].tolist()
            decoded = tokens_to_circuit_v2(token_ids)

            if decoded is None:
                continue

            # Refine topology
            walk_tokens = decoded.get('walk_tokens', [])
            if walk_tokens:
                refined = _identify_topology_from_walk(walk_tokens)
                if refined is not None:
                    decoded['topology'] = refined

            decoded_count += 1
            total_decoded += 1

            # V_out error
            topo = decoded['topology']
            params = decoded['params']
            try:
                expected_vout = calculate_expected_vout(topo, params)
                vout_error = abs(expected_vout - specs['v_out']) / max(specs['v_out'], 0.01)
                tc_vout_errors.append(vout_error)
                vout_errors.append(vout_error)
            except Exception:
                pass

            # SPICE
            try:
                waveform = run_simulation(topo, params, timeout=10)
                if waveform is not None:
                    spice_count += 1
                    total_spice_ok += 1
            except Exception:
                pass

        avg_err = f"{100*sum(tc_vout_errors)/len(tc_vout_errors):.1f}%" if tc_vout_errors else "N/A"
        print(f"  {name}: decoded={decoded_count}/{n_samples}, "
              f"SPICE={spice_count}/{decoded_count}, V_out_err={avg_err}")

    print(f"\n--- Aggregate ---")
    print(f"  Decode rate:     {total_decoded}/{total_generated} ({100*total_decoded/max(1,total_generated):.0f}%)")
    print(f"  SPICE success:   {total_spice_ok}/{total_decoded} ({100*total_spice_ok/max(1,total_decoded):.0f}%)")
    if vout_errors:
        import numpy as np
        print(f"  Mean V_out err:  {100*np.mean(vout_errors):.1f}%")
        print(f"  Median V_out err: {100*np.median(vout_errors):.1f}%")

    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
