"""ARCS training dataset: loads generated circuit data and prepares batches.

Reads JSONL files from Phase 1 data generation, tokenizes using
CircuitTokenizer, and provides PyTorch Dataset/DataLoader for training.

Sequence format (from tokenizer.encode_circuit_sample):
    START, TOPO_X, SEP, SPEC_VIN, val, ..., SEP, COMP_X, VAL, ..., END

Each sample becomes an (input, target) pair shifted by one token for
teacher-forced next-token prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from arcs.tokenizer import CircuitTokenizer, TokenType
from arcs.datagen import CircuitSample


class CircuitDataset(Dataset):
    """Training dataset for autoregressive circuit generation.

    Loads JSONL data, tokenizes each sample, creates teacher-forced
    (input, target) pairs with value masks for weighted loss.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: CircuitTokenizer,
        max_seq_len: int = 128,
        valid_only: bool = False,
    ):
        """
        Args:
            data_path:   Path to directory of JSONL files, or a single JSONL file
            tokenizer:   CircuitTokenizer instance
            max_seq_len: Maximum sequence length (pad or truncate)
            valid_only:  If True, only include samples that passed validation
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.samples: list[CircuitSample] = []
        self.sequences: list[list[int]] = []
        self.token_types_list: list[list[int]] = []

        # --- Load raw data ---
        data_path = Path(data_path)
        if data_path.is_dir():
            files = sorted(data_path.glob("*.jsonl"))
        else:
            files = [data_path]

        for fpath in files:
            with open(fpath) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    sample = CircuitSample.from_dict(json.loads(line))
                    if valid_only and not sample.valid:
                        continue
                    self.samples.append(sample)

        # --- Tokenize ---
        for sample in self.samples:
            tokens = tokenizer.encode_circuit_sample(sample)
            token_types = self._compute_token_types(tokens)
            self.sequences.append(tokens)
            self.token_types_list.append(token_types)

        # --- Summary ---
        n_valid = sum(1 for s in self.samples if s.valid)
        n_invalid = len(self.samples) - n_valid
        print(f"Loaded {len(self.samples)} samples ({n_valid} valid, {n_invalid} invalid)")

        if self.sequences:
            lens = [len(s) for s in self.sequences]
            print(
                f"Sequence lengths: min={min(lens)}, max={max(lens)}, "
                f"mean={np.mean(lens):.1f}"
            )

    def _compute_token_types(self, tokens: list[int]) -> list[int]:
        """Map each token ID to its TokenType enum value (0-indexed)."""
        types = []
        for tid in tokens:
            if 0 <= tid < len(self.tokenizer.tokens):
                types.append(self.tokenizer.tokens[tid].token_type.value - 1)
            else:
                types.append(0)  # default to SPECIAL
        return types

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = list(self.sequences[idx])
        types = list(self.token_types_list[idx])

        # Truncate if needed (keep END token)
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len - 1] + [self.tokenizer.end_id]
            types = types[: self.max_seq_len - 1] + [0]

        # Teacher forcing: input = seq[:-1], target = seq[1:]
        input_ids = seq[:-1]
        target_ids = seq[1:]
        input_types = types[:-1]

        # Pad to (max_seq_len - 1)
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_id] * pad_len
            target_ids += [self.tokenizer.pad_id] * pad_len
            input_types += [0] * pad_len

        # Value mask: which target positions are VALUE tokens
        value_mask = []
        for tid in target_ids:
            if 0 <= tid < len(self.tokenizer.tokens):
                value_mask.append(
                    self.tokenizer.tokens[tid].token_type == TokenType.VALUE
                )
            else:
                value_mask.append(False)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_types": torch.tensor(input_types, dtype=torch.long),
            "targets": torch.tensor(target_ids, dtype=torch.long),
            "value_mask": torch.tensor(value_mask, dtype=torch.bool),
            "valid": torch.tensor(self.samples[idx].valid, dtype=torch.bool),
        }


class EulerianAugmentedDataset(CircuitDataset):
    """Extended dataset with Eulerian path data augmentation.

    For each circuit, generates multiple valid token orderings via
    different Eulerian paths through the circuit graph. This helps
    the model learn that circuits are about connections, not sequence order.
    (Following AnalogGenie's approach.)
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: CircuitTokenizer,
        max_seq_len: int = 128,
        valid_only: bool = False,
        n_augmentations: int = 5,
    ):
        # Load base data
        super().__init__(data_path, tokenizer, max_seq_len, valid_only)

        if n_augmentations > 1:
            # TODO: Implement Eulerian augmentation using euler.py
            # For now, use random component order shuffling as a simple augmentation
            original_len = len(self.sequences)
            self._augment_by_component_shuffle(n_augmentations - 1)
            print(
                f"Augmented: {original_len} → {len(self.sequences)} sequences "
                f"({n_augmentations}× augmentation)"
            )

    def _augment_by_component_shuffle(self, n_extra: int) -> None:
        """Simple augmentation: shuffle component order in the circuit body.

        The spec prefix (START → TOPO → SEP → specs → SEP) stays fixed.
        Only the component section (COMP → VAL pairs) is randomly reordered.
        """
        rng = np.random.default_rng(42)
        original_len = len(self.sequences)

        for orig_idx in range(original_len):
            seq = self.sequences[orig_idx]
            types = self.token_types_list[orig_idx]

            # Find the second SEP (end of spec section)
            sep_count = 0
            body_start = -1
            for i, tid in enumerate(seq):
                if tid == self.tokenizer.sep_id:
                    sep_count += 1
                    if sep_count == 2:
                        body_start = i + 1
                        break

            if body_start < 0:
                continue

            # Find END token
            body_end = len(seq) - 1  # Assume END is last token

            # Extract prefix and component pairs
            prefix = seq[:body_start]
            prefix_types = types[:body_start]

            # Components come in (TYPE, VALUE) pairs
            body = seq[body_start:body_end]
            body_types = types[body_start:body_end]
            suffix = seq[body_end:]
            suffix_types = types[body_end:]

            if len(body) < 4:  # Need at least 2 component pairs
                continue

            # Group into pairs
            pairs = []
            type_pairs = []
            for j in range(0, len(body) - 1, 2):
                pairs.append((body[j], body[j + 1]))
                type_pairs.append((body_types[j], body_types[j + 1]))

            for _ in range(n_extra):
                order = rng.permutation(len(pairs))
                new_body = []
                new_body_types = []
                for k in order:
                    new_body.extend(pairs[k])
                    new_body_types.extend(type_pairs[k])

                new_seq = prefix + new_body + suffix
                new_types = prefix_types + new_body_types + suffix_types
                self.sequences.append(new_seq)
                self.token_types_list.append(new_types)
                # Reuse original sample reference
                self.samples.append(self.samples[orig_idx])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_path: str | Path,
    tokenizer: CircuitTokenizer,
    max_seq_len: int = 128,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    valid_only: bool = False,
    augment: bool = False,
    n_augmentations: int = 5,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        data_path:        Path to data directory or JSONL file
        tokenizer:        CircuitTokenizer instance
        max_seq_len:      Max sequence length
        batch_size:       Training batch size
        val_split:        Fraction reserved for validation
        seed:             Random seed for split reproducibility
        num_workers:      DataLoader worker processes
        valid_only:       Only use valid circuit samples
        augment:          Enable Eulerian / shuffle augmentation
        n_augmentations:  Number of augmented copies per sample

    Returns:
        (train_loader, val_loader)
    """
    if augment:
        dataset = EulerianAugmentedDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            valid_only=valid_only,
            n_augmentations=n_augmentations,
        )
    else:
        dataset = CircuitDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            valid_only=valid_only,
        )

    # Train/val split
    n = len(dataset)
    n_val = max(int(n * val_split), 1)
    n_train = n - n_val

    rng = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=rng
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
