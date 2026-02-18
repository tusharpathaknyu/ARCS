"""
PyTorch Dataset and DataLoader for CircuitGenie training.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..data.generator import CircuitSample
from ..data.spice_templates import PARAM_NAMES
from ..tokenizer.tokenizer import CircuitTokenizer
from ..tokenizer.vocabulary import (
    VALUE_BIN_OFFSET, NUM_VALUE_BINS, PAD_ID,
    TOKEN_TO_ID, PARAM_NAME_TO_TOKEN, SPEC_KEY_TO_INFO,
)
from ..tokenizer.sequence import SPEC_ORDER


# Spec-param matching pairs: spec key -> param name with same physical quantity
# SPEC_V_IN should match PARAM_V_IN, etc.
SPEC_PARAM_MATCH = {
    'v_in': 'V_in',     # Input voltage must match
}


def _compute_spec_param_pairs(seq: np.ndarray) -> np.ndarray:
    """
    Find positions of (spec_value, param_value) pairs in the TARGET sequence
    that should have matching value bins.

    The target sequence is seq[1:] (teacher forcing shift).
    We return positions in the TARGET (shifted) indexing.

    Args:
        seq: Full padded token sequence (max_seq_len,)

    Returns:
        (N_pairs, 2) array of (spec_val_target_pos, param_val_target_pos).
        Padded with -1 if fewer pairs found.
    """
    max_pairs = len(SPEC_PARAM_MATCH)  # Currently 1 pair (V_in)
    pairs = np.full((max_pairs, 2), -1, dtype=np.int64)

    # Build target-indexed lookup: target[i] = seq[i+1]
    # So if a token is at seq position p, it's at target position p-1.

    # 1) Find spec value positions in seq
    # Seq: BOS(0) SPEC_V_IN(1) VAL(2) SPEC_V_OUT(3) VAL(4) ...
    spec_value_positions = {}  # spec_key -> seq position of value token
    for i, spec_key in enumerate(SPEC_ORDER):
        # Spec value is at seq position: 1 + i*2 + 1 = 2 + i*2
        spec_val_pos = 2 + i * 2
        if spec_val_pos < len(seq) and seq[spec_val_pos] != PAD_ID:
            spec_value_positions[spec_key] = spec_val_pos

    # 2) Find param value positions in seq
    # Params start after: BOS(0) + 5 spec pairs(10) + SEP(11) + TOPO(12) + SEP(13)
    # So first param name at seq position 14, first param value at 15
    param_start = 14  # Position of first PARAM_x token
    param_value_positions = {}  # param_name -> seq position of value token

    for pos in range(param_start, len(seq) - 1, 2):
        token_id = int(seq[pos])
        if token_id == PAD_ID or token_id == 2:  # PAD or EOS
            break
        # Check if this is a param name token
        for pname, ptok in PARAM_NAME_TO_TOKEN.items():
            if TOKEN_TO_ID.get(ptok) == token_id:
                # Value is at pos+1
                val_pos = pos + 1
                if val_pos < len(seq) and seq[val_pos] != PAD_ID:
                    param_value_positions[pname] = val_pos
                break

    # 3) Build pairs in target indexing (target_pos = seq_pos - 1)
    pair_idx = 0
    for spec_key, param_name in SPEC_PARAM_MATCH.items():
        if spec_key in spec_value_positions and param_name in param_value_positions:
            spec_target_pos = spec_value_positions[spec_key] - 1
            param_target_pos = param_value_positions[param_name] - 1
            if spec_target_pos >= 0 and param_target_pos >= 0:
                pairs[pair_idx, 0] = spec_target_pos
                pairs[pair_idx, 1] = param_target_pos
                pair_idx += 1

    return pairs


class CircuitDataset(Dataset):
    """Dataset of tokenized circuit sequences for next-token prediction."""

    def __init__(
        self,
        samples: List[CircuitSample],
        tokenizer: CircuitTokenizer,
    ):
        self.tokenizer = tokenizer
        max_len = tokenizer.max_seq_len

        # Pre-tokenize all samples
        self.sequences = []
        self.spec_param_pairs = []
        for s in samples:
            tokens = tokenizer.encode(s)
            padded = tokenizer.pad_sequence(tokens)
            self.sequences.append(padded)
            self.spec_param_pairs.append(_compute_spec_param_pairs(np.array(padded)))

        self.sequences = np.array(self.sequences, dtype=np.int64)
        self.spec_param_pairs = np.array(self.spec_param_pairs, dtype=np.int64)
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

        # Spec-param pair positions for consistency loss
        pairs = torch.tensor(self.spec_param_pairs[idx], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'value_mask': value_mask,
            'spec_param_pairs': pairs,
        }


def create_dataloaders(
    samples: List[CircuitSample],
    tokenizer: CircuitTokenizer,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        samples: List of CircuitSample objects
        tokenizer: CircuitTokenizer instance
        batch_size: Batch size
        val_split: Fraction of data for validation
        seed: Random seed for split

    Returns:
        (train_loader, val_loader)
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    n_val = int(len(samples) * val_split)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    train_dataset = CircuitDataset(train_samples, tokenizer)
    val_dataset = CircuitDataset(val_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader
