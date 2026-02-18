"""Quick smoke test for ARCS model, dataset, and training pipeline."""

import torch
from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer


def test_model():
    print("=== Model Smoke Test ===")
    tokenizer = CircuitTokenizer()
    config = ARCSConfig.small()
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {tokenizer.vocab_size}")

    model = ARCSModel(config)
    n = model.count_parameters()
    print(f"Model parameters: {n:,} ({n / 1e6:.1f}M)")
    print("Parameter breakdown:")
    for k, v in model.count_parameters_by_group().items():
        if v > 0:
            print(f"  {k}: {v:,}")

    # Forward pass
    x = torch.randint(0, config.vocab_size, (2, 20))
    t = torch.zeros_like(x)
    logits, _ = model(x, token_types=t)
    print(f"Forward: input {x.shape} -> logits {logits.shape}")

    # Loss
    logits, loss = model(x, token_types=t, targets=x)
    print(f"Loss: {loss.item():.4f}")

    # Generation
    prefix = torch.tensor([[tokenizer.start_id]])
    gen = model.generate(prefix, max_new_tokens=15, temperature=0.8, top_k=50)
    print(f"Generated: {gen.shape[1]} tokens")
    seq_str = tokenizer.sequence_to_string(gen[0].tolist())
    print(f"  {seq_str[:150]}")

    print("Model smoke test PASSED\n")


def test_dataset_from_samples():
    """Test dataset with synthetic samples (no real data needed)."""
    import json
    import tempfile
    from pathlib import Path
    from arcs.datagen import CircuitSample

    print("=== Dataset Smoke Test ===")

    # Create a few synthetic samples
    samples = [
        CircuitSample(
            topology="buck",
            parameters={"inductance": 22e-6, "capacitance": 470e-6, "r_dson": 0.05, "esr": 0.01, "r_load": 5.0},
            operating_conditions={"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000.0},
            metrics={"efficiency": 92.5, "vout_avg": 4.95, "vout_ripple": 0.03, "vout_error_pct": 1.0, "ripple_ratio": 0.006},
            valid=True,
            sim_time=1.5,
        ),
        CircuitSample(
            topology="boost",
            parameters={"inductance": 47e-6, "capacitance": 220e-6, "r_dson": 0.05, "esr": 0.02, "r_load": 12.0},
            operating_conditions={"vin": 5.0, "vout": 12.0, "iout": 0.5, "fsw": 100000.0},
            metrics={"efficiency": 88.0, "vout_avg": 11.8, "vout_ripple": 0.1, "vout_error_pct": 1.7, "ripple_ratio": 0.008},
            valid=True,
            sim_time=2.0,
        ),
        CircuitSample(
            topology="buck_boost",
            parameters={"inductance": 33e-6, "capacitance": 330e-6, "r_dson": 0.05, "esr": 0.015, "r_load": 9.0},
            operating_conditions={"vin": 12.0, "vout": -9.0, "iout": 1.0, "fsw": 100000.0},
            metrics={},
            valid=False,
            sim_time=0.0,
            error_message="Sim failed",
        ),
    ]

    # Write to temp JSONL
    tmpdir = tempfile.mkdtemp()
    tmpfile = Path(tmpdir) / "test.jsonl"
    with open(tmpfile, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")

    tokenizer = CircuitTokenizer()
    from arcs.dataset import CircuitDataset
    ds = CircuitDataset(data_path=tmpfile, tokenizer=tokenizer, max_seq_len=64)
    print(f"Dataset size: {len(ds)}")

    # Check a batch
    item = ds[0]
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"targets shape:   {item['targets'].shape}")
    print(f"token_types:     {item['token_types'].shape}")
    print(f"value_mask sum:  {item['value_mask'].sum().item()}")
    print(f"valid:           {item['valid'].item()}")

    # Decode sequence
    full_seq = [tokenizer.start_id] + item["input_ids"].tolist()
    readable = tokenizer.sequence_to_string([t for t in full_seq if t != tokenizer.pad_id])
    print(f"Token sequence:  {readable[:200]}")

    print("Dataset smoke test PASSED\n")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


def test_training_step():
    """Test one training step end-to-end."""
    import json
    import tempfile
    from pathlib import Path
    from arcs.datagen import CircuitSample
    from arcs.dataset import create_dataloaders

    print("=== Training Step Test ===")

    # Create synthetic data
    tmpdir = tempfile.mkdtemp()
    tmpfile = Path(tmpdir) / "test.jsonl"
    samples = []
    for i in range(50):
        s = CircuitSample(
            topology="buck",
            parameters={
                "inductance": 22e-6 * (1 + i * 0.1),
                "capacitance": 470e-6,
                "r_dson": 0.05,
                "esr": 0.01,
                "r_load": 5.0,
            },
            operating_conditions={"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000.0},
            metrics={"efficiency": 90 + i * 0.1, "vout_avg": 4.9 + i * 0.01, "vout_ripple": 0.03},
            valid=True,
            sim_time=1.0,
        )
        samples.append(s)

    with open(tmpfile, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")

    tokenizer = CircuitTokenizer()
    config = ARCSConfig.small()
    config.vocab_size = tokenizer.vocab_size

    train_loader, val_loader = create_dataloaders(
        data_path=tmpfile,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        batch_size=8,
        val_split=0.2,
    )

    model = ARCSModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # One training step
    model.train()
    batch = next(iter(train_loader))
    logits, loss = model(
        batch["input_ids"],
        token_types=batch["token_types"],
        targets=batch["targets"],
        value_mask=batch["value_mask"],
    )
    loss.backward()
    optimizer.step()
    print(f"Training step loss: {loss.item():.4f}")

    # One eval step
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        logits, loss = model(
            batch["input_ids"],
            token_types=batch["token_types"],
            targets=batch["targets"],
            value_mask=batch["value_mask"],
        )
    print(f"Eval step loss: {loss.item():.4f}")
    print("Training step test PASSED\n")

    import shutil
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_model()
    test_dataset_from_samples()
    test_training_step()
    print("=" * 40)
    print("ALL TESTS PASSED")
