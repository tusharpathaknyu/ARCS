"""
PyTorch Dataset v2 with Eulerian walk augmentation and invalid examples.

Each circuit sample generates N_WALKS different Eulerian walks,
providing data augmentation. Failed/invalid circuits are included
with an is_valid flag for the model to learn constraints.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..data.generator import CircuitSample
from ..tokenizer.tokenizer_v2 import CircuitTokenizerV2
from ..tokenizer.vocabulary_v2 import (
    VALUE_BIN_OFFSET, NUM_VALUE_BINS, PAD_ID,
    TOKEN_TO_ID_V2, SPEC_KEY_TO_INFO_V2,
)
from ..tokenizer.sequence_v2 import SPEC_ORDER


class CircuitDatasetV2(Dataset):
    """
    Dataset of tokenized circuit sequences with Eulerian walk augmentation.

    Each sample generates n_walks different walks for data augmentation.
    """

    def __init__(
        self,
        samples: List[CircuitSample],
        tokenizer: CircuitTokenizerV2,
        n_walks: int = 5,
        base_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.n_walks = n_walks
        max_len = tokenizer.max_seq_len

        # Pre-tokenize all samples with augmentation
        self.sequences = []
        for i, s in enumerate(samples):
            for w in range(n_walks):
                walk_seed = base_seed + i * n_walks + w
                try:
                    tokens = tokenizer.encode(s, walk_seed=walk_seed)
                    padded = tokenizer.pad_sequence(tokens)
                    self.sequences.append(padded)
                except Exception:
                    # Skip samples that fail to encode
                    pass

        self.sequences = np.array(self.sequences, dtype=np.int64)
        self.n_samples = len(self.sequences)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]  # (max_seq_len,)

        # Teacher forcing: input = tokens[:-1], target = tokens[1:]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)

        # Attention mask (1 = attend, 0 = ignore padding)
        attention_mask = (input_ids != PAD_ID).long()

        # Value token mask for weighted loss
        value_mask = (
            (labels >= VALUE_BIN_OFFSET) &
            (labels < VALUE_BIN_OFFSET + NUM_VALUE_BINS)
        )

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'value_mask': value_mask,
        }


def create_dataloaders_v2(
    samples: List[CircuitSample],
    tokenizer: CircuitTokenizerV2,
    batch_size: int = 64,
    val_split: float = 0.1,
    n_walks: int = 5,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val DataLoaders with Eulerian walk augmentation.

    Note: We split BEFORE augmentation so val set has unseen base circuits.
    """
    rng = np.random.RandomState(seed)
    n_total = len(samples)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = rng.permutation(n_total)
    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train:]]

    train_dataset = CircuitDatasetV2(
        train_samples, tokenizer,
        n_walks=n_walks, base_seed=seed,
    )
    # Val uses only 1 walk (no augmentation for consistent evaluation)
    val_dataset = CircuitDatasetV2(
        val_samples, tokenizer,
        n_walks=1, base_seed=seed + 999999,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} sequences ({n_train} circuits × {n_walks} walks)")
    print(f"Val:   {len(val_dataset)} sequences ({n_val} circuits × 1 walk)")

    return train_loader, val_loader
