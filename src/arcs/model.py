"""ARCS GPT-style decoder-only transformer for circuit generation.

Architecture:
  - Token embedding + learned positional embedding + token-type embedding
  - N transformer blocks (pre-norm, causal self-attention, SwiGLU FFN)
  - LM head (weight-tied with token embedding)

Configs:
  - Small  (~6M params): d_model=256, n_layers=6,  n_heads=4,  d_ff=1024
  - Base   (~50M params): d_model=512, n_layers=12, n_heads=8,  d_ff=2048
  - Large  (~100M params): d_model=768, n_layers=12, n_heads=12, d_ff=3072

The model generates circuits autoregressively, conditioned on a spec prefix:
    START → TOPO → SEP → specs → SEP → components → END

Spec conditioning is achieved naturally through the causal attention:
circuit tokens attend to all preceding spec tokens, learning the mapping
from performance requirements to circuit structure and component values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ARCSConfig:
    """Model hyperparameters."""

    vocab_size: int = 675        # CircuitTokenizer vocabulary
    max_seq_len: int = 128       # Maximum token sequence length
    d_model: int = 256           # Embedding / hidden dimension
    n_heads: int = 4             # Number of attention heads
    n_layers: int = 6            # Number of transformer blocks
    d_ff: int = 1024             # SwiGLU inner dimension
    dropout: float = 0.1        # Dropout rate
    pad_id: int = 0              # PAD token ID
    n_token_types: int = 7       # TokenType categories
    weight_tying: bool = True    # Tie LM head ↔ token embedding

    @classmethod
    def small(cls) -> ARCSConfig:
        """~6M parameter config — good for initial training on 14K samples."""
        return cls(d_model=256, n_heads=4, n_layers=6, d_ff=1024)

    @classmethod
    def base(cls) -> ARCSConfig:
        """~50M parameter config — full-scale training."""
        return cls(d_model=512, n_heads=8, n_layers=12, d_ff=2048)

    @classmethod
    def large(cls) -> ARCSConfig:
        """~100M parameter config."""
        return cls(d_model=768, n_heads=12, n_layers=12, d_ff=3072)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ARCSConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no bias, GPT-2/LLaMA style)."""

    def __init__(self, config: ARCSConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1
            ).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nh, hd)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (LLaMA-style, ~30% better than GELU FFN)."""

    def __init__(self, config: ARCSConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)  # gate
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm."""

    def __init__(self, config: ARCSConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ARCSModel(nn.Module):
    """GPT-style decoder-only transformer for autoregressive circuit generation.

    Generates circuits token-by-token, conditioned on a spec prefix:
        START → TOPO_X → SEP → SPEC_VIN → val → ... → SEP → COMP → VAL → ... → END

    Features:
        - Token type embeddings (spec prefix vs circuit body vs value tokens)
        - Value-weighted loss for better numerical prediction
        - Weight tying between token embedding and LM head
        - SwiGLU FFN + RMSNorm for modern transformer best practices
    """

    def __init__(self, config: ARCSConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.type_emb = nn.Embedding(config.n_token_types, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # --- Output ---
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.lm_head.weight = self.tok_emb.weight

        # --- Weight initialization ---
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) for stable training
        for block in self.blocks:
            nn.init.normal_(
                block.attn.proj.weight, std=0.02 / math.sqrt(2 * config.n_layers)
            )
            nn.init.normal_(
                block.ffn.w2.weight, std=0.02 / math.sqrt(2 * config.n_layers)
            )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_types: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        value_weight: float = 5.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional teacher-forced loss computation.

        Args:
            input_ids:   (B, T) token IDs
            token_types: (B, T) token type IDs (0=SPECIAL, 1=COMPONENT, ...)
            targets:     (B, T) target token IDs for loss
            value_mask:  (B, T) bool mask — True at value token positions in targets
            value_weight: Loss multiplier for value token positions (default 5×)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar or None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        # Build embeddings: token + position + type
        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        if token_types is not None:
            x = x + self.type_emb(token_types)
        x = self.emb_drop(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss computation
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.config.vocab_size)
            targets_flat = targets.view(-1)

            ce = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=self.config.pad_id,
                reduction="none",
            )

            if value_mask is not None:
                vm = value_mask.view(-1).float()
                weights = 1.0 + (value_weight - 1.0) * vm
                pad_mask = (targets_flat != self.config.pad_id).float()
                weights = weights * pad_mask
                loss = (ce * weights).sum() / weights.sum().clamp(min=1.0)
            else:
                non_pad = targets_flat != self.config.pad_id
                loss = ce[non_pad].mean()

        return logits, loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        token_types_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation from a prefix (e.g. spec tokens).

        Args:
            prefix:             (1, T_prefix) starting token IDs
            max_new_tokens:     Maximum new tokens to generate
            temperature:        Sampling temperature (1.0 = standard)
            top_k:              Top-k logit filtering
            top_p:              Nucleus (top-p) sampling threshold
            token_types_prefix: (1, T_prefix) types for prefix tokens

        Returns:
            (1, T_prefix + T_gen) full generated sequence
        """
        self.eval()
        seq = prefix.clone()
        tseq = token_types_prefix.clone() if token_types_prefix is not None else None

        end_id = 2  # END token

        for _ in range(max_new_tokens):
            # Crop to context window
            if seq.shape[1] > self.config.max_seq_len:
                s = seq[:, -self.config.max_seq_len :]
                t = tseq[:, -self.config.max_seq_len :] if tseq is not None else None
            else:
                s = seq
                t = tseq

            logits, _ = self.forward(s, token_types=t)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_tok], dim=1)

            # Extend type sequence (default to SPECIAL=0 for generated tokens)
            if tseq is not None:
                new_type = torch.zeros(1, 1, dtype=torch.long, device=seq.device)
                tseq = torch.cat([tseq, new_type], dim=1)

            if next_tok.item() == end_id:
                break

        return seq

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_group(self) -> dict[str, int]:
        """Parameter count breakdown."""
        groups = {
            "embeddings": 0,
            "attention": 0,
            "ffn": 0,
            "norm": 0,
            "lm_head": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if "emb" in name:
                groups["embeddings"] += n
            elif "attn" in name or "qkv" in name or "proj" in name:
                groups["attention"] += n
            elif "ffn" in name or "w1" in name or "w2" in name or "w3" in name:
                groups["ffn"] += n
            elif "ln" in name or "norm" in name:
                groups["norm"] += n
            elif "lm_head" in name:
                groups["lm_head"] += n
        return groups
