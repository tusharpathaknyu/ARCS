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

    Uses euler.py's CircuitGraph and find_euler_paths() when possible.
    Falls back to component-order shuffling for topologies without
    graph definitions.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: CircuitTokenizer,
        max_seq_len: int = 128,
        valid_only: bool = False,
        n_augmentations: int = 5,
        use_euler: bool = True,
    ):
        # Load base data
        super().__init__(data_path, tokenizer, max_seq_len, valid_only)

        if n_augmentations > 1:
            original_len = len(self.sequences)
            euler_count = 0
            shuffle_count = 0

            if use_euler:
                euler_count = self._augment_by_euler_walks(n_augmentations - 1)
                shuffle_count = (original_len * (n_augmentations - 1)) - euler_count
            else:
                # Shuffle-only augmentation (no Eulerian walks)
                for idx in range(original_len):
                    self._augment_single_by_shuffle(idx, n_augmentations - 1)
                shuffle_count = len(self.sequences) - original_len

            print(
                f"Augmented: {original_len} → {len(self.sequences)} sequences "
                f"({n_augmentations}× augmentation, "
                f"{euler_count} Eulerian + {shuffle_count} shuffle)"
            )

    def _augment_by_euler_walks(self, n_extra: int) -> int:
        """Augment using Eulerian walk orderings from euler.py.

        For each circuit sample, converts to a CircuitGraph, finds multiple
        Eulerian paths, and re-encodes the circuit with components in each
        path's ordering. Falls back to shuffle for samples where Euler
        augmentation fails.

        Returns:
            Number of successfully Euler-augmented sequences.
        """
        from arcs.euler import circuit_sample_to_graph

        original_len = len(self.sequences)
        euler_successes = 0

        for orig_idx in range(original_len):
            sample = self.samples[orig_idx]

            try:
                graph = circuit_sample_to_graph(sample)
                paths = graph.find_euler_paths(max_paths=n_extra + 1)

                if len(paths) > 1:
                    # Use Eulerian paths to reorder components
                    for path in paths[1 : n_extra + 1]:
                        seq = self._encode_with_euler_ordering(
                            sample, path
                        )
                        if seq is not None:
                            types = self._compute_token_types(seq)
                            self.sequences.append(seq)
                            self.token_types_list.append(types)
                            self.samples.append(sample)
                            euler_successes += 1

                    # If we didn't get enough Euler augmentations, fill with shuffles
                    remaining = n_extra - min(len(paths) - 1, n_extra)
                    if remaining > 0:
                        self._augment_single_by_shuffle(orig_idx, remaining)
                else:
                    # No diverse Euler paths found, fall back to shuffle
                    self._augment_single_by_shuffle(orig_idx, n_extra)

            except Exception:
                # Euler augmentation failed, fall back to shuffle
                self._augment_single_by_shuffle(orig_idx, n_extra)

        return euler_successes

    def _encode_with_euler_ordering(self, sample, euler_path) -> list[int] | None:
        """Re-encode a circuit sample with components ordered by Euler path.

        The spec prefix stays the same. Only the component body section
        (COMP_X VAL pairs) is reordered to match the Euler path traversal.
        """
        tokens = [self.tokenizer.start_id]

        # Topology token
        topo_key = f"TOPO_{sample.topology.upper()}"
        if topo_key in self.tokenizer.name_to_id:
            tokens.append(self.tokenizer.name_to_id[topo_key])
        tokens.append(self.tokenizer.sep_id)

        # Spec tokens (same as original)
        oc = sample.operating_conditions
        _OC_SPEC = {
            "vin": "SPEC_VIN", "vout": "SPEC_VOUT", "iout": "SPEC_IOUT",
            "fsw": "SPEC_FSW", "vin_amp": "SPEC_VIN",
            "freq_test": "SPEC_CUTOFF_FREQ", "vcc": "SPEC_VIN",
        }
        spec_map: dict[str, float | None] = {}
        for oc_key, oc_val in oc.items():
            spec_tok = _OC_SPEC.get(oc_key)
            if spec_tok and spec_tok not in spec_map:
                spec_map[spec_tok] = oc_val

        if sample.valid:
            m = sample.metrics
            _METRIC_SPEC = {
                "efficiency": "SPEC_EFFICIENCY",
                "vout_ripple": "SPEC_RIPPLE",
                "gain_db": "SPEC_GAIN",
                "bw_3db": "SPEC_BANDWIDTH",
                "fc_3db": "SPEC_CUTOFF_FREQ",
                "phase_rad": "SPEC_PHASE_MARGIN",
                "f_peak": "SPEC_CENTER_FREQ",
                "vosc_pp": "SPEC_VIN",
            }
            for mk, sk in _METRIC_SPEC.items():
                val = m.get(mk)
                if val is not None and sk not in spec_map:
                    spec_map[sk] = val

        for spec_key, spec_val in spec_map.items():
            if spec_val is not None and spec_key in self.tokenizer.name_to_id:
                tokens.append(self.tokenizer.name_to_id[spec_key])
                tokens.append(self.tokenizer.encode_value(abs(spec_val)))
        tokens.append(self.tokenizer.sep_id)

        # Component section: reordered by Euler path
        # Map euler path edges to (component_type, value) pairs
        comp_mapping = self.tokenizer._params_to_components(
            sample.topology, sample.parameters
        )

        # Build lookup from component_id in euler path → (comp_type, value)
        euler_order = []
        comp_lookup = {}
        for comp_type, comp_val in comp_mapping:
            comp_key = comp_type.upper()
            if comp_key not in comp_lookup:
                comp_lookup[comp_key] = []
            comp_lookup[comp_key].append(comp_val)

        # Try to match Euler path edges to component types
        used_counts: dict[str, int] = {}
        for edge in euler_path:
            ct = edge.component_type.upper()
            idx = used_counts.get(ct, 0)
            if ct in comp_lookup and idx < len(comp_lookup[ct]):
                euler_order.append((ct, comp_lookup[ct][idx]))
                used_counts[ct] = idx + 1

        # If Euler ordering doesn't cover all components, append remaining
        remaining = []
        for comp_type, comp_val in comp_mapping:
            ct = comp_type.upper()
            used = used_counts.get(ct, 0)
            if used > 0:
                used_counts[ct] = used - 1
            else:
                remaining.append((ct, comp_val))
        euler_order.extend(remaining)

        if not euler_order:
            return None

        for comp_type, comp_value in euler_order:
            comp_key = f"COMP_{comp_type}"
            if comp_key in self.tokenizer.name_to_id:
                tokens.append(self.tokenizer.name_to_id[comp_key])
                tokens.append(self.tokenizer.encode_value(comp_value))

        tokens.append(self.tokenizer.end_id)
        return tokens

    def _augment_single_by_shuffle(self, orig_idx: int, n_extra: int) -> None:
        """Shuffle-augment a single sample."""
        rng = np.random.default_rng(42 + orig_idx)
        seq = self.sequences[orig_idx]
        types = self.token_types_list[orig_idx]

        sep_count = 0
        body_start = -1
        for i, tid in enumerate(seq):
            if tid == self.tokenizer.sep_id:
                sep_count += 1
                if sep_count == 2:
                    body_start = i + 1
                    break

        if body_start < 0:
            return

        body_end = len(seq) - 1
        prefix = seq[:body_start]
        prefix_types = types[:body_start]
        body = seq[body_start:body_end]
        body_types = types[body_start:body_end]
        suffix = seq[body_end:]
        suffix_types = types[body_end:]

        if len(body) < 4:
            return

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
            self.sequences.append(prefix + new_body + suffix)
            self.token_types_list.append(prefix_types + new_body_types + suffix_types)
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
