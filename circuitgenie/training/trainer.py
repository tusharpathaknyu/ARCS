"""
Training loop for CircuitGenie.
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..model.config import CircuitGenieConfig
from ..model.transformer import CircuitGenieModel
from ..tokenizer.vocabulary import VALUE_BIN_OFFSET, NUM_VALUE_BINS, PAD_ID


class Trainer:
    """Training loop with cosine LR schedule and gradient clipping."""

    def __init__(
        self,
        model: CircuitGenieModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        value_weight: float = 5.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.value_weight = value_weight
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Track total steps for LR schedule
        self.total_steps = 0
        self.warmup_steps = warmup_steps
        self.base_lr = lr

        # Best val loss for checkpointing
        self.best_val_loss = float('inf')

    def _get_lr(self, step: int, total_training_steps: int) -> float:
        """Cosine schedule with linear warmup."""
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_training_steps - self.warmup_steps)
        return self.base_lr * 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def train(self, n_epochs: int = 100, log_every: int = 1) -> Dict:
        """
        Run training for n_epochs.

        Returns:
            Dict with training history
        """
        total_training_steps = n_epochs * len(self.train_loader)
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_value_accuracy': [],
            'lr': [],
        }

        print(f"Training: {n_epochs} epochs, {len(self.train_loader)} steps/epoch")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Device: {self.device}")
        print()

        for epoch in range(n_epochs):
            t0 = time.time()

            # ---- Train ----
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                value_mask = batch['value_mask'].to(self.device)
                spec_param_pairs = batch.get('spec_param_pairs')
                if spec_param_pairs is not None:
                    spec_param_pairs = spec_param_pairs.to(self.device)

                # LR schedule
                lr = self._get_lr(self.total_steps, total_training_steps)
                self._set_lr(lr)

                # Forward
                logits, loss = self.model(
                    input_ids, labels, value_mask,
                    spec_param_pairs=spec_param_pairs,
                    value_weight=self.value_weight,
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                self.total_steps += 1

            avg_train_loss = epoch_loss / max(1, n_batches)

            # ---- Validate ----
            val_metrics = self.evaluate()
            dt = time.time() - t0

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_value_accuracy'].append(val_metrics['value_accuracy'])
            history['lr'].append(lr)

            # Log
            if (epoch + 1) % log_every == 0:
                print(
                    f"Epoch {epoch+1:3d}/{n_epochs} | "
                    f"train_loss={avg_train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_acc={val_metrics['accuracy']:.1%} | "
                    f"val_val_acc={val_metrics['value_accuracy']:.1%} | "
                    f"lr={lr:.2e} | "
                    f"{dt:.1f}s"
                )

            # Checkpoint best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(self.checkpoint_dir / "best_model.pt")

            # Periodic checkpoint
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt"
                )

        # Save final
        self.save_checkpoint(self.checkpoint_dir / "final_model.pt")
        return history

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set. Returns loss and accuracies."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        value_correct = 0
        value_total = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            value_mask = batch['value_mask'].to(self.device)
            spec_param_pairs = batch.get('spec_param_pairs')
            if spec_param_pairs is not None:
                spec_param_pairs = spec_param_pairs.to(self.device)

            logits, loss = self.model(
                input_ids, labels, value_mask,
                spec_param_pairs=spec_param_pairs,
                value_weight=self.value_weight,
            )
            total_loss += loss.item()

            # Per-token accuracy (excluding PAD)
            preds = logits.argmax(dim=-1)  # (B, T)
            non_pad = labels != PAD_ID
            total_correct += (preds == labels)[non_pad].sum().item()
            total_tokens += non_pad.sum().item()

            # Value-token accuracy
            value_positions = value_mask & non_pad
            if value_positions.any():
                value_correct += (preds == labels)[value_positions].sum().item()
                value_total += value_positions.sum().item()

        n_batches = max(1, len(self.val_loader))
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_correct / max(1, total_tokens),
            'value_accuracy': value_correct / max(1, value_total),
        }

    def save_checkpoint(self, path: Path) -> None:
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'total_steps': self.total_steps,
            'best_val_loss': self.best_val_loss,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
