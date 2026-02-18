"""
High-level tokenizer class for CircuitGenie.
"""

from typing import Dict, List, Optional

import numpy as np

from ..data.generator import CircuitSample
from .vocabulary import (
    VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID, SEP_ID,
    VALUE_BIN_OFFSET, NUM_VALUE_BINS, is_value_token,
)
from .sequence import (
    MAX_SEQ_LEN, circuit_to_tokens, tokens_to_circuit,
    tokens_to_netlist, tokens_to_readable,
)


class CircuitTokenizer:
    """Tokenizer for circuit samples."""

    def __init__(self, max_seq_len: int = MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def sep_id(self) -> int:
        return SEP_ID

    def encode(self, sample: CircuitSample) -> List[int]:
        """Encode a CircuitSample to token IDs."""
        return circuit_to_tokens(sample.topology, sample.params, sample.specs)

    def decode(self, token_ids: List[int]) -> Optional[Dict]:
        """Decode token IDs back to topology, params, specs."""
        return tokens_to_circuit(token_ids)

    def decode_to_netlist(self, token_ids: List[int]) -> Optional[str]:
        """Decode token IDs to a SPICE netlist string."""
        return tokens_to_netlist(token_ids)

    def to_readable(self, token_ids: List[int]) -> str:
        """Convert token IDs to human-readable string."""
        return tokens_to_readable(token_ids)

    def pad_sequence(self, token_ids: List[int]) -> List[int]:
        """Right-pad a token sequence to max_seq_len."""
        padded = token_ids[:self.max_seq_len]
        padded = padded + [PAD_ID] * (self.max_seq_len - len(padded))
        return padded

    def batch_encode(self, samples: List[CircuitSample]) -> np.ndarray:
        """Encode and pad a batch of samples. Returns (N, max_seq_len) int array."""
        batch = []
        for s in samples:
            tokens = self.encode(s)
            padded = self.pad_sequence(tokens)
            batch.append(padded)
        return np.array(batch, dtype=np.int64)

    def get_value_token_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """Return boolean mask of which positions are value tokens."""
        return (token_ids >= VALUE_BIN_OFFSET) & (
            token_ids < VALUE_BIN_OFFSET + NUM_VALUE_BINS
        )
