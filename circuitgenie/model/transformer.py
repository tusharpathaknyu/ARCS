"""
GPT-style decoder-only transformer for circuit generation.

Architecture:
  - Token embedding + learned positional embedding
  - N transformer blocks (pre-norm, causal self-attention)
  - LM head (weight-tied with token embedding)

~813K parameters with default config (4 layers, 128 dim, 4 heads).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CircuitGenieConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, config: CircuitGenieConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (upper triangular = masked)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feedforward with GELU activation."""

    def __init__(self, config: CircuitGenieConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (GPT-2 style)."""

    def __init__(self, config: CircuitGenieConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CircuitGenieModel(nn.Module):
    """
    GPT-style decoder-only transformer for circuit generation.

    Input: token IDs (B, T)
    Output: logits over vocabulary (B, T, vocab_size)
    """

    def __init__(self, config: CircuitGenieConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_final = nn.LayerNorm(config.d_model)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        spec_param_pairs: Optional[torch.Tensor] = None,
        value_weight: float = 5.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token IDs
            targets: (B, T) target token IDs for loss computation
            value_mask: (B, T) boolean mask where True = value token position
            spec_param_pairs: (B, N_pairs, 2) indices of (spec_val_pos, param_val_pos)
                              pairs in the TARGET sequence that should have matching
                              value bins (e.g., SPEC_V_IN val ↔ PARAM_V_IN val)
            value_weight: multiplier for value token loss (default 5.0)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        # Embeddings
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy loss with padding ignored
            logits_flat = logits.view(-1, self.config.vocab_size)
            targets_flat = targets.view(-1)

            # Base loss (ignore PAD)
            ce_loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=self.config.pad_token_id,
                reduction='none',
            )

            # Weighted loss: value_weight on value tokens, 1.0 on structural
            if value_mask is not None:
                weights = torch.ones_like(ce_loss)
                value_mask_flat = value_mask.view(-1).float()
                weights = weights + (value_weight - 1.0) * value_mask_flat
                loss = (ce_loss * weights).sum() / weights.sum()
            else:
                non_pad = (targets_flat != self.config.pad_token_id)
                loss = ce_loss[non_pad].mean()

            # Spec-param consistency loss: penalize when predicted value bins
            # for spec and corresponding param diverge
            if spec_param_pairs is not None and spec_param_pairs.shape[1] > 0:
                # Get predicted distributions at spec and param positions
                log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
                probs = F.softmax(logits, dim=-1)

                consistency_loss = torch.tensor(0.0, device=logits.device)
                n_pairs = 0

                for pair_idx in range(spec_param_pairs.shape[1]):
                    spec_pos = spec_param_pairs[:, pair_idx, 0]  # (B,)
                    param_pos = spec_param_pairs[:, pair_idx, 1]  # (B,)

                    # Skip invalid pairs (padded with -1)
                    valid = (spec_pos >= 0) & (param_pos >= 0) & (spec_pos < T) & (param_pos < T)
                    if not valid.any():
                        continue

                    # For each valid sample, the param position should predict
                    # close to the target value at the spec position.
                    # We use KL divergence between spec distribution and param distribution
                    # restricted to value token range
                    for b in range(B):
                        if not valid[b]:
                            continue
                        sp = spec_pos[b].item()
                        pp = param_pos[b].item()

                        # Target: the spec value token (ground truth)
                        spec_target = targets[b, sp].item()
                        # The param position should predict the same value
                        param_logits_at_pos = logits[b, pp, :]

                        # Add CE loss for the param position targeting the spec value
                        if spec_target != self.config.pad_token_id:
                            pair_loss = F.cross_entropy(
                                param_logits_at_pos.unsqueeze(0),
                                torch.tensor([spec_target], device=logits.device),
                            )
                            consistency_loss = consistency_loss + pair_loss
                            n_pairs += 1

                if n_pairs > 0:
                    consistency_loss = consistency_loss / n_pairs
                    loss = loss + 0.5 * consistency_loss  # 50% weight on consistency

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> torch.Tensor:
        """
        Autoregressive generation from a prefix.

        Args:
            prefix_ids: (1, T_prefix) starting token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_k: only sample from top-k logits

        Returns:
            (1, T_total) generated token IDs including prefix
        """
        self.eval()
        ids = prefix_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            ids_cond = ids[:, -self.config.max_seq_len:]

            logits, _ = self.forward(ids_cond)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-k filtering
            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, -1:]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
            ids = torch.cat([ids, next_id], dim=1)

            # Stop at EOS
            if next_id.item() == self.config.eos_token_id:
                break

        return ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
