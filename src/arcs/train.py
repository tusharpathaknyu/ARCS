"""ARCS training script.

Phase 2 training pipeline:
  1. Pre-training: next-token prediction on all circuit sequences
  2. Spec-conditioning: learned naturally from prefix structure
  3. Evaluation: validity rate, spec compliance, diversity

Usage:
    cd CircuitGenie
    PYTHONPATH=src python -m arcs.train --data data/phase1 --config small --epochs 100

Features:
    - Value-weighted loss (5× weight on component value tokens)
    - Token-type embeddings (spec vs. component vs. value)
    - Cosine LR schedule with linear warmup
    - Periodic sample generation for qualitative monitoring
    - Best model + periodic checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer
from arcs.dataset import create_dataloaders


# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: ARCSModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    grad_clip: float = 1.0,
    value_weight: float = 5.0,
) -> dict[str, float]:
    """Train for one epoch, return metrics."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        targets = batch["targets"].to(device)
        value_mask = batch["value_mask"].to(device)

        logits, loss = model(
            input_ids,
            token_types=token_types,
            targets=targets,
            value_mask=value_mask,
            value_weight=value_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n_tok = (targets != model.config.pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20)),
        "n_batches": n_batches,
    }


@torch.no_grad()
def evaluate(
    model: ARCSModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    value_weight: float = 5.0,
) -> dict[str, float]:
    """Evaluate on validation set, return metrics."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    value_correct = 0
    value_total = 0
    struct_correct = 0
    struct_total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        targets = batch["targets"].to(device)
        value_mask = batch["value_mask"].to(device)

        logits, loss = model(
            input_ids,
            token_types=token_types,
            targets=targets,
            value_mask=value_mask,
            value_weight=value_weight,
        )

        n_tok = (targets != model.config.pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

        # Token-level accuracy
        preds = logits.argmax(dim=-1)
        non_pad = targets != model.config.pad_id
        correct += ((preds == targets) & non_pad).sum().item()

        # Value-token accuracy (how well model predicts component values)
        vm = value_mask & non_pad
        value_correct += ((preds == targets) & vm).sum().item()
        value_total += vm.sum().item()

        # Structural accuracy (topology, spec names, component types)
        sm = (~value_mask) & non_pad
        struct_correct += ((preds == targets) & sm).sum().item()
        struct_total += sm.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20)),
        "accuracy": correct / max(total_tokens, 1),
        "value_accuracy": value_correct / max(value_total, 1),
        "struct_accuracy": struct_correct / max(struct_total, 1),
    }


# ---------------------------------------------------------------------------
# Generation for qualitative monitoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model: ARCSModel,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_samples: int = 5,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[str]:
    """Generate unconditional circuit samples and return as readable strings."""
    model.eval()
    results = []

    for _ in range(n_samples):
        prefix = torch.tensor([[tokenizer.start_id]], device=device)
        output = model.generate(
            prefix,
            max_new_tokens=80,
            temperature=temperature,
            top_k=top_k,
        )
        text = tokenizer.sequence_to_string(output[0].tolist())
        results.append(text)

    return results


@torch.no_grad()
def generate_from_specs(
    model: ARCSModel,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    topology: str,
    specs: dict[str, float],
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Generate a circuit conditioned on topology and spec values.

    Args:
        topology: e.g. "buck", "boost"
        specs:    e.g. {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}

    Returns:
        Readable token sequence string
    """
    model.eval()

    # Build spec prefix: START, TOPO_X, SEP, SPEC_VIN, val, SPEC_VOUT, val, ..., SEP
    prefix_ids = [tokenizer.start_id]

    topo_key = f"TOPO_{topology.upper()}"
    if topo_key in tokenizer.name_to_id:
        prefix_ids.append(tokenizer.name_to_id[topo_key])
    prefix_ids.append(tokenizer.sep_id)

    for spec_name, spec_val in specs.items():
        spec_key = f"SPEC_{spec_name.upper()}"
        if spec_key in tokenizer.name_to_id:
            prefix_ids.append(tokenizer.name_to_id[spec_key])
            prefix_ids.append(tokenizer.encode_value(abs(spec_val)))
    prefix_ids.append(tokenizer.sep_id)

    prefix = torch.tensor([prefix_ids], device=device)
    output = model.generate(
        prefix,
        max_new_tokens=60,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.sequence_to_string(output[0].tolist())


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARCS Model Training")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data directory or JSONL file")
    parser.add_argument("--config", type=str, default="small",
                        choices=["small", "base", "large"],
                        help="Model size config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=5.0,
                        help="Loss multiplier for value tokens (default 5×)")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--valid-only", action="store_true",
                        help="Train only on valid circuit samples")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation via component shuffling")
    parser.add_argument("--n-augmentations", type=int, default=5,
                        help="Augmentation factor (N orderings per sample)")
    parser.add_argument("--output", type=str, default="checkpoints/",
                        help="Checkpoint output directory")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print metrics every N epochs")
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # --- Seed ---
    torch.manual_seed(args.seed)

    # --- Device ---
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Tokenizer ---
    tokenizer = CircuitTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # --- Model config ---
    config_map = {
        "small": ARCSConfig.small,
        "base": ARCSConfig.base,
        "large": ARCSConfig.large,
    }
    config = config_map[args.config]()
    config.vocab_size = tokenizer.vocab_size
    print(f"Model config: {args.config}")

    # --- Data ---
    train_loader, val_loader = create_dataloaders(
        data_path=args.data,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        valid_only=args.valid_only,
        augment=args.augment,
        n_augmentations=args.n_augmentations,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model ---
    model = ARCSModel(config).to(device)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    param_groups = model.count_parameters_by_group()
    for group, count in param_groups.items():
        if count > 0:
            print(f"  {group}: {count:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Linear warmup → cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # min LR = 10%

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Output ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Training loop ---
    best_val_loss = float("inf")
    history: list[dict] = []

    print(f"\n{'=' * 60}")
    print(f"Training ARCS ({args.config}) for {args.epochs} epochs")
    print(f"  Value weight: {args.value_weight}×")
    print(f"  Warmup: {args.warmup_epochs} epochs ({warmup_steps} steps)")
    print(f"  Total steps: {total_steps}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip,
            value_weight=args.value_weight,
        )

        val_metrics = evaluate(
            model, val_loader, device,
            value_weight=args.value_weight,
        )

        dt = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ppl": train_metrics["ppl"],
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
            "val_accuracy": val_metrics["accuracy"],
            "val_value_acc": val_metrics["value_accuracy"],
            "val_struct_acc": val_metrics["struct_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "time": dt,
        }
        history.append(record)

        # --- Log ---
        if epoch % args.log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_metrics['loss']:.4f} ppl={train_metrics['ppl']:.1f} | "
                f"val={val_metrics['loss']:.4f} ppl={val_metrics['ppl']:.1f} | "
                f"acc={val_metrics['accuracy']:.3f} "
                f"v_acc={val_metrics['value_accuracy']:.3f} "
                f"s_acc={val_metrics['struct_accuracy']:.3f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"{dt:.1f}s"
            )

        # --- Best model ---
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "val_loss": best_val_loss,
                },
                output_dir / "best_model.pt",
            )

        # --- Periodic checkpoint ---
        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.to_dict(),
                },
                output_dir / f"checkpoint_epoch{epoch}.pt",
            )

        # --- Generated samples ---
        if epoch % (args.log_interval * 5) == 0:
            print("\n--- Unconditional Samples ---")
            samples = generate_samples(model, tokenizer, device, n_samples=3)
            for i, s in enumerate(samples):
                print(f"  [{i + 1}] {s[:200]}")

            # Spec-conditioned sample
            print("--- Spec-Conditioned Sample (Buck 12→5V) ---")
            cond = generate_from_specs(
                model, tokenizer, device,
                topology="buck",
                specs={"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000},
            )
            print(f"  {cond[:200]}")
            print("--- End Samples ---\n")

    # --- Save history & final model ---
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
        },
        output_dir / "final_model.pt",
    )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
