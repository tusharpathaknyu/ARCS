"""
CircuitTokenizer v2: wraps Eulerian walk encoding/decoding.
"""

from typing import Dict, List, Optional

import numpy as np

from ..data.generator import CircuitSample
from ..data.spice_templates import Topology
from .vocabulary_v2 import (
    VOCAB_SIZE_V2, PAD_ID, BOS_ID, EOS_ID,
    VALUE_BIN_OFFSET, NUM_VALUE_BINS,
)
from .sequence_v2 import (
    circuit_to_tokens_v2, tokens_to_circuit_v2,
    tokens_to_readable_v2, MAX_SEQ_LEN_V2,
    _identify_topology_from_walk,
)


class CircuitTokenizerV2:
    """Tokenizer for Eulerian walk circuit representation."""

    def __init__(self, max_seq_len: int = MAX_SEQ_LEN_V2):
        self.vocab_size = VOCAB_SIZE_V2
        self.max_seq_len = max_seq_len

    def encode(
        self,
        sample: CircuitSample,
        walk_seed: int = 42,
    ) -> List[int]:
        """Encode a CircuitSample to v2 token IDs."""
        return circuit_to_tokens_v2(
            sample.topology, sample.params, sample.specs,
            walk_seed=walk_seed,
        )

    def decode(self, token_ids: List[int]) -> Optional[Dict]:
        """Decode token IDs back to circuit dict."""
        decoded = tokens_to_circuit_v2(token_ids)
        if decoded is not None:
            walk_tokens = decoded.get('walk_tokens', [])
            if walk_tokens:
                refined = _identify_topology_from_walk(walk_tokens)
                if refined is not None:
                    decoded['topology'] = refined
        return decoded

    def to_readable(self, token_ids: List[int]) -> str:
        """Convert token IDs to human-readable string."""
        return tokens_to_readable_v2(token_ids)

    def pad_sequence(self, tokens: List[int]) -> List[int]:
        """Pad or truncate a token sequence to max_seq_len."""
        if len(tokens) >= self.max_seq_len:
            return tokens[:self.max_seq_len]
        return tokens + [PAD_ID] * (self.max_seq_len - len(tokens))

    def get_value_token_mask(self, token_ids: List[int]) -> List[bool]:
        """Return a mask where True = value token position."""
        return [
            VALUE_BIN_OFFSET <= t < VALUE_BIN_OFFSET + NUM_VALUE_BINS
            for t in token_ids
        ]
