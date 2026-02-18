"""Model hyperparameter configuration."""

from dataclasses import dataclass

from ..tokenizer.vocabulary import VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID
from ..tokenizer.vocabulary_v2 import (
    VOCAB_SIZE_V2,
    PAD_ID as PAD_ID_V2,
    BOS_ID as BOS_ID_V2,
    EOS_ID as EOS_ID_V2,
)


@dataclass
class CircuitGenieConfig:
    """V1 config: flat parameter representation, 157 tokens, 32 max length."""
    vocab_size: int = VOCAB_SIZE       # 157
    max_seq_len: int = 32
    d_model: int = 128                 # embedding dimension
    n_heads: int = 4                   # attention heads
    n_layers: int = 4                  # transformer blocks
    d_ff: int = 512                    # feedforward inner dim
    dropout: float = 0.1
    pad_token_id: int = PAD_ID         # 0
    bos_token_id: int = BOS_ID         # 1
    eos_token_id: int = EOS_ID         # 2


@dataclass
class CircuitGenieConfigV2:
    """V2 config: Eulerian walk representation, 161 tokens, 64 max length."""
    vocab_size: int = VOCAB_SIZE_V2    # 161
    max_seq_len: int = 64
    d_model: int = 256                 # 2× bigger embedding
    n_heads: int = 8                   # 2× more heads
    n_layers: int = 6                  # 1.5× deeper
    d_ff: int = 1024                   # 2× wider FFN
    dropout: float = 0.1
    pad_token_id: int = PAD_ID_V2     # 0
    bos_token_id: int = BOS_ID_V2     # 1
    eos_token_id: int = EOS_ID_V2     # 2
