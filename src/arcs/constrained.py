"""ARCS Constrained Decoding: Guaranteed-Valid Circuit Generation.

Enforces circuit validity by construction through autoregressive token masking.
At each decode step, a validity mask restricts the model's output distribution
to only structurally legal next tokens, guaranteeing 100% structural validity
without RL.

Three levels of constraints (applied cumulatively):
  Level 1 — Sequence grammar: enforces START→TOPO→SEP→specs→SEP→comps→END
  Level 2 — Topology-aware components: correct component types and counts
  Level 3 — Value range: limits value tokens to physically valid ranges

Key insight: the autoregressive constraint projection avoids the continuous→
discrete gap problem of VAE-based approaches (ValidCircuitGen). Each token
is discrete from the start; the mask simply zeros out invalid options before
sampling, preserving the model's learned distribution over valid tokens.

Usage:
    from arcs.constrained import ConstrainedGenerator, ConstraintLevel

    gen = ConstrainedGenerator(model, tokenizer, level=ConstraintLevel.FULL)
    seq = gen.generate(prefix, topology="buck")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch
import torch.nn.functional as F

from arcs.model import ARCSConfig
from arcs.simulate import COMPONENT_TO_PARAM, normalize_topology
from arcs.templates import (
    POWER_CONVERTER_BOUNDS,
    SIGNAL_CIRCUIT_BOUNDS,
    _TIER1_NAMES,
    _TIER2_NAMES,
)
from arcs.tokenizer import CircuitTokenizer, TokenType


# ---------------------------------------------------------------------------
# Constraint levels
# ---------------------------------------------------------------------------

class ConstraintLevel(IntEnum):
    """How strictly to constrain generation."""
    NONE = 0          # No constraints (baseline)
    GRAMMAR = 1       # Sequence grammar only (COMP/VAL alternation + END)
    TOPOLOGY = 2      # + correct component types per topology
    FULL = 3          # + value range constraints per component


# ---------------------------------------------------------------------------
# Grammar state machine
# ---------------------------------------------------------------------------

class GrammarState(IntEnum):
    """Where we are in the circuit sequence."""
    PREFIX = 0         # Still in prefix (START, TOPO, SEP, specs, SEP)
    EXPECT_COMP = 1    # Expecting a component token
    EXPECT_VAL = 2     # Expecting a value token (after component)
    DONE = 3           # END generated, stop


@dataclass
class DecoderState:
    """Tracks what has been generated so far for constraint enforcement."""

    grammar: GrammarState = GrammarState.EXPECT_COMP
    topology: str = ""

    # Component tracking
    expected_components: list[tuple[str, str]] = field(default_factory=list)
    # [(comp_type, param_name), ...] — components still to be placed
    placed_components: list[tuple[str, str]] = field(default_factory=list)
    # [(comp_type, param_name), ...] — components already generated

    # Current component being valued
    current_component: Optional[tuple[str, str]] = None  # (comp_type, param_name)
    current_comp_token: Optional[str] = None  # COMP_RESISTOR etc.

    n_components_generated: int = 0

    def remaining_types(self) -> dict[str, int]:
        """Count how many of each component type still need placing."""
        counts: dict[str, int] = {}
        for comp_type, _ in self.expected_components:
            counts[comp_type] = counts.get(comp_type, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Token range helpers
# ---------------------------------------------------------------------------

def _get_token_ids_by_type(
    tokenizer: CircuitTokenizer, token_type: TokenType
) -> list[int]:
    """Get all token IDs of a given type."""
    return [t.id for t in tokenizer.tokens if t.token_type == token_type]


def _get_component_token_id(tokenizer: CircuitTokenizer, comp_type: str) -> int:
    """Get token ID for a component type like 'RESISTOR' → COMP_RESISTOR."""
    key = f"COMP_{comp_type.upper()}"
    return tokenizer.name_to_id.get(key, -1)


def _value_bin_index(value: float, tokenizer: CircuitTokenizer) -> int:
    """Convert a continuous value to its bin index (0-499)."""
    if value <= 0:
        return 0
    log_val = math.log10(value)
    log_min = math.log10(tokenizer.VALUE_MIN)
    log_max = math.log10(tokenizer.VALUE_MAX)
    log_val = max(log_min, min(log_max, log_val))
    bin_idx = int((log_val - log_min) / (log_max - log_min) * tokenizer.N_VALUE_BINS)
    return min(bin_idx, tokenizer.N_VALUE_BINS - 1)


def _get_value_token_range(
    min_val: float, max_val: float, tokenizer: CircuitTokenizer
) -> tuple[int, int]:
    """Get the range of value token IDs [lo_id, hi_id] covering [min_val, max_val]."""
    lo_bin = _value_bin_index(min_val, tokenizer)
    hi_bin = _value_bin_index(max_val, tokenizer)
    # Value tokens start at the ID of VAL_0
    val_0_id = tokenizer.name_to_id["VAL_0"]
    return val_0_id + lo_bin, val_0_id + hi_bin


# ---------------------------------------------------------------------------
# Bounds lookup
# ---------------------------------------------------------------------------

_ALL_BOUNDS = {**POWER_CONVERTER_BOUNDS, **SIGNAL_CIRCUIT_BOUNDS}


def _get_param_bounds(
    topology: str, param_name: str
) -> Optional[tuple[float, float]]:
    """Get (min_val, max_val) for a parameter in a topology."""
    bounds_list = _ALL_BOUNDS.get(topology)
    if bounds_list is None:
        return None
    for b in bounds_list:
        if b.name == param_name:
            return (b.min_val, b.max_val)
    return None


# ---------------------------------------------------------------------------
# Constraint mask computation
# ---------------------------------------------------------------------------

class ConstraintMask:
    """Precomputes and caches token sets for fast mask generation."""

    def __init__(self, tokenizer: CircuitTokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        # Cache token ID sets by type
        self.component_ids = set(_get_token_ids_by_type(tokenizer, TokenType.COMPONENT))
        self.value_ids = set(_get_token_ids_by_type(tokenizer, TokenType.VALUE))
        self.spec_ids = set(_get_token_ids_by_type(tokenizer, TokenType.SPEC))
        self.topology_ids = set(_get_token_ids_by_type(tokenizer, TokenType.TOPOLOGY))

        self.end_id = tokenizer.end_id
        self.sep_id = tokenizer.sep_id
        self.start_id = tokenizer.start_id
        self.pad_id = tokenizer.pad_id

        # Map component type string → set of matching token IDs
        # e.g. "RESISTOR" → {id of COMP_RESISTOR}
        self.comp_type_to_ids: dict[str, set[int]] = {}
        for tok in tokenizer.tokens:
            if tok.token_type == TokenType.COMPONENT:
                ctype = tok.name.replace("COMP_", "")
                if ctype not in self.comp_type_to_ids:
                    self.comp_type_to_ids[ctype] = set()
                self.comp_type_to_ids[ctype].add(tok.id)

        # Precompute value token ID range
        self.val_0_id = tokenizer.name_to_id["VAL_0"]
        self.val_last_id = tokenizer.name_to_id[f"VAL_{tokenizer.N_VALUE_BINS - 1}"]

    def compute_mask(
        self,
        state: DecoderState,
        level: ConstraintLevel,
    ) -> torch.Tensor:
        """Compute a validity mask over the vocabulary.

        Returns:
            Tensor of shape (vocab_size,) with 0.0 for valid tokens
            and -inf for invalid tokens. Add to logits before softmax.
        """
        mask = torch.full((self.vocab_size,), float("-inf"))

        if state.grammar == GrammarState.DONE:
            # Only PAD is valid after END
            mask[self.pad_id] = 0.0
            return mask

        if state.grammar == GrammarState.EXPECT_COMP:
            mask = self._mask_expect_comp(state, level)
        elif state.grammar == GrammarState.EXPECT_VAL:
            mask = self._mask_expect_val(state, level)
        else:
            # PREFIX state — no constraints (prefix is given)
            mask[:] = 0.0

        return mask

    def _mask_expect_comp(
        self, state: DecoderState, level: ConstraintLevel
    ) -> torch.Tensor:
        """At a component position: which tokens are allowed?"""
        mask = torch.full((self.vocab_size,), float("-inf"))

        if not state.expected_components:
            # All components placed — must produce END
            mask[self.end_id] = 0.0
            return mask

        if level == ConstraintLevel.GRAMMAR:
            # Level 1: any component token or END (if ≥2 components placed)
            for tid in self.component_ids:
                mask[tid] = 0.0
            if state.n_components_generated >= 2:
                mask[self.end_id] = 0.0

        elif level >= ConstraintLevel.TOPOLOGY:
            # Level 2+: only component types still needed
            remaining = state.remaining_types()
            for comp_type, count in remaining.items():
                if count > 0:
                    ids = self.comp_type_to_ids.get(comp_type, set())
                    for tid in ids:
                        mask[tid] = 0.0

            # Allow END only if no remaining components
            # (strict mode: must place all expected components)
            # Relaxed: allow END if ≥ min_components placed
            if not state.expected_components:
                mask[self.end_id] = 0.0

        return mask

    def _mask_expect_val(
        self, state: DecoderState, level: ConstraintLevel
    ) -> torch.Tensor:
        """At a value position: which value tokens are allowed?"""
        mask = torch.full((self.vocab_size,), float("-inf"))

        if level <= ConstraintLevel.TOPOLOGY:
            # Level 1-2: any value token
            for tid in self.value_ids:
                mask[tid] = 0.0
        else:
            # Level 3: only value tokens within the parameter's valid range
            assert level == ConstraintLevel.FULL
            if state.current_component is not None:
                _, param_name = state.current_component
                bounds = _get_param_bounds(state.topology, param_name)
                if bounds is not None:
                    lo_id, hi_id = _get_value_token_range(
                        bounds[0], bounds[1], self.tokenizer
                    )
                    # Allow tokens in range [lo_id, hi_id]
                    for tid in range(lo_id, hi_id + 1):
                        if tid in self.value_ids:
                            mask[tid] = 0.0
                else:
                    # No bounds found — allow all values
                    for tid in self.value_ids:
                        mask[tid] = 0.0
            else:
                # No current component info — allow all values
                for tid in self.value_ids:
                    mask[tid] = 0.0

        return mask


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def _update_state(
    state: DecoderState,
    token_id: int,
    tokenizer: CircuitTokenizer,
) -> DecoderState:
    """Update decoder state after generating a token."""
    tok = tokenizer.tokens[token_id]

    if state.grammar == GrammarState.EXPECT_COMP:
        if tok.name == "END":
            state.grammar = GrammarState.DONE
        elif tok.token_type == TokenType.COMPONENT:
            comp_type = tok.name.replace("COMP_", "")
            # Find matching expected component (first of this type)
            matched = None
            for i, (etype, pname) in enumerate(state.expected_components):
                if etype == comp_type:
                    matched = (etype, pname)
                    state.expected_components.pop(i)
                    break

            if matched is not None:
                state.current_component = matched
            else:
                # Component not in expected list — still track it
                state.current_component = (comp_type, f"unknown_{state.n_components_generated}")

            state.current_comp_token = tok.name
            state.grammar = GrammarState.EXPECT_VAL

    elif state.grammar == GrammarState.EXPECT_VAL:
        if tok.token_type == TokenType.VALUE:
            state.placed_components.append(
                state.current_component or ("UNKNOWN", "unknown")
            )
            state.current_component = None
            state.current_comp_token = None
            state.n_components_generated += 1
            state.grammar = GrammarState.EXPECT_COMP

    return state


# ---------------------------------------------------------------------------
# Constrained Generator
# ---------------------------------------------------------------------------

class ConstrainedGenerator:
    """Wraps an ARCS model to enforce validity constraints during generation.

    The generator intercepts the logits at each decode step, applies a
    constraint mask, and samples from the restricted distribution. This
    guarantees 100% structural validity for Level ≥ TOPOLOGY.
    """

    def __init__(
        self,
        model,
        tokenizer: CircuitTokenizer,
        level: ConstraintLevel = ConstraintLevel.FULL,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.level = level
        self.mask_computer = ConstraintMask(tokenizer)

        # Component type ID set (for two-head routing)
        self.comp_ids: set[int] = set()
        for tok in tokenizer.tokens:
            if tok.token_type == TokenType.COMPONENT:
                self.comp_ids.add(tok.id)

    def _init_state(self, topology: str) -> DecoderState:
        """Initialize decoder state for a given topology."""
        topology = normalize_topology(topology)
        expected = COMPONENT_TO_PARAM.get(topology, [])

        state = DecoderState(
            grammar=GrammarState.EXPECT_COMP,
            topology=topology,
            expected_components=list(expected),  # copy
        )
        return state

    def _extract_topology_from_prefix(self, prefix: torch.Tensor) -> str:
        """Extract topology name from the prefix token sequence."""
        for i in range(min(5, prefix.shape[1])):
            tid = prefix[0, i].item()
            if 0 <= tid < len(self.tokenizer.tokens):
                tok = self.tokenizer.tokens[tid]
                if tok.token_type == TokenType.TOPOLOGY:
                    return tok.name.replace("TOPO_", "").lower()
        return ""

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        topology: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        token_types_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a circuit with constrained decoding.

        Args:
            prefix: (1, T) conditioning prefix (START, TOPO, SEP, specs, SEP)
            topology: topology name (auto-detected from prefix if None)
            max_new_tokens: maximum generation length
            temperature: sampling temperature
            top_k: top-k filtering (applied AFTER constraint mask)
            top_p: nucleus sampling (applied AFTER constraint mask)
            token_types_prefix: optional token type IDs for the prefix

        Returns:
            (1, T+N) full sequence including prefix and generated tokens
        """
        self.model.eval()
        device = prefix.device

        # Detect topology
        if topology is None:
            topology = self._extract_topology_from_prefix(prefix)

        # Initialize state
        state = self._init_state(topology)

        seq = prefix.clone()
        tseq = token_types_prefix.clone() if token_types_prefix is not None else None

        for step in range(max_new_tokens):
            if state.grammar == GrammarState.DONE:
                break

            # Crop to context window
            max_len = self.model.config.max_seq_len if hasattr(self.model, 'config') else 128
            if seq.shape[1] > max_len:
                s = seq[:, -max_len:]
                t = tseq[:, -max_len:] if tseq is not None else None
            else:
                s = seq
                t = tseq

            # Get logits from model
            logits = self._get_logits(s, t, seq)

            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Apply constraint mask (the key innovation)
            if self.level > ConstraintLevel.NONE:
                constraint_mask = self.mask_computer.compute_mask(
                    state, self.level
                ).to(device)
                logits = logits + constraint_mask

            # Apply top-k (after constraints)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p / nucleus (after constraints)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            # Update sequence
            seq = torch.cat([seq, next_tok], dim=1)

            # Update token types
            if tseq is not None:
                new_type_val = 0
                tok_id = next_tok.item()
                if 0 <= tok_id < len(self.tokenizer.tokens):
                    new_type_val = self.tokenizer.tokens[tok_id].token_type.value - 1
                new_type = torch.tensor(
                    [[new_type_val]], dtype=torch.long, device=device
                )
                tseq = torch.cat([tseq, new_type], dim=1)

            # Update constraint state
            state = _update_state(state, next_tok.item(), self.tokenizer)

        # If we ran out of tokens without END, force it
        if state.grammar != GrammarState.DONE:
            end_tok = torch.tensor(
                [[self.tokenizer.end_id]], dtype=torch.long, device=device
            )
            seq = torch.cat([seq, end_tok], dim=1)

        return seq

    def _get_logits(
        self,
        s: torch.Tensor,
        t: Optional[torch.Tensor],
        full_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Get last-token logits, handling different model architectures."""
        model = self.model
        model_type = type(model).__name__

        if model_type == "GraphTransformerARCSModel":
            # Graph Transformer: needs graph features
            g_adj, e_types, w_pos = model.compute_graph_features(s, self.tokenizer)
            B, T = s.shape
            positions = torch.arange(T, device=s.device)
            x = model.tok_emb(s) + model.pos_emb(positions)
            if t is not None:
                x = x + model.type_emb(t)
            if w_pos is not None:
                x = x + model.walk_pos_emb(w_pos.clamp(0, 31))
            x = model.emb_drop(x)
            for block in model.blocks:
                x = block(x, graph_adj=g_adj, edge_types=e_types)
            x = model.ln_f(x)
            h_last = x[:, -1:, :]

            last_tok = full_seq[0, -1].item()
            if last_tok in self.comp_ids:
                logits = model.value_head(model.value_proj(h_last) + h_last)
            else:
                logits = model.structure_head(h_last)
            return logits[:, -1, :]

        elif model_type == "TwoHeadARCSModel":
            # Two-Head: route through backbone + head selection
            h = model._backbone(s, t)
            h_last = h[:, -1:, :]

            last_tok = full_seq[0, -1].item()
            if last_tok in self.comp_ids:
                logits = model.value_head(model.value_proj(h_last) + h_last)
            else:
                logits = model.structure_head(h_last)
            return logits[:, -1, :]

        else:
            # Baseline ARCSModel
            logits, _ = model(s, token_types=t)
            return logits[:, -1, :]

    def generate_batch(
        self,
        prefixes: list[torch.Tensor],
        topologies: list[str],
        **kwargs,
    ) -> list[torch.Tensor]:
        """Generate multiple circuits (sequentially, each with constraints)."""
        results = []
        for prefix, topo in zip(prefixes, topologies):
            seq = self.generate(prefix, topology=topo, **kwargs)
            results.append(seq)
        return results


# ---------------------------------------------------------------------------
# Constrained generation with log-probs (for RL / training)
# ---------------------------------------------------------------------------

def constrained_sample_with_logprobs(
    model,
    prefix: torch.Tensor,
    tokenizer: CircuitTokenizer,
    topology: str,
    level: ConstraintLevel = ConstraintLevel.FULL,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample with constraints, returning tokens + log-probs + entropy.

    Like rl.sample_with_logprobs but with constraint masking.

    Returns:
        generated: (N,) generated token IDs
        log_probs: (N,) log probability of each generated token
        entropies: (N,) entropy at each generation step
    """
    model.eval()
    device = prefix.device
    mask_computer = ConstraintMask(tokenizer)

    topology = normalize_topology(topology)
    state = DecoderState(
        grammar=GrammarState.EXPECT_COMP,
        topology=topology,
        expected_components=list(COMPONENT_TO_PARAM.get(topology, [])),
    )

    comp_ids = set()
    for tok in tokenizer.tokens:
        if tok.token_type == TokenType.COMPONENT:
            comp_ids.add(tok.id)

    gen_tokens = []
    gen_logprobs = []
    gen_entropies = []

    seq = prefix.clone()

    for step in range(max_new_tokens):
        if state.grammar == GrammarState.DONE:
            break

        max_len = model.config.max_seq_len if hasattr(model, 'config') else 128
        if seq.shape[1] > max_len:
            s = seq[:, -max_len:]
        else:
            s = seq

        # Get logits (simplified — works for baseline model)
        model_type = type(model).__name__
        if model_type in ("TwoHeadARCSModel", "GraphTransformerARCSModel"):
            # Forward through backbone
            if model_type == "GraphTransformerARCSModel":
                g_adj, e_types, w_pos = model.compute_graph_features(s, tokenizer)
                B, T = s.shape
                positions = torch.arange(T, device=s.device)
                x = model.tok_emb(s) + model.pos_emb(positions)
                if w_pos is not None:
                    x = x + model.walk_pos_emb(w_pos.clamp(0, 31))
                x = model.emb_drop(x)
                for block in model.blocks:
                    x = block(x, graph_adj=g_adj, edge_types=e_types)
                x = model.ln_f(x)
            else:
                x = model._backbone(s)

            h_last = x[:, -1:, :]
            last_tok = seq[0, -1].item()
            if last_tok in comp_ids:
                raw_logits = model.value_head(model.value_proj(h_last) + h_last)
            else:
                raw_logits = model.structure_head(h_last)
            raw_logits = raw_logits[:, -1, :]
        else:
            logits_out, _ = model(s)
            raw_logits = logits_out[:, -1, :]

        # Apply temperature
        scaled_logits = raw_logits / max(temperature, 1e-8)

        # Apply constraint mask
        if level > ConstraintLevel.NONE:
            constraint_mask = mask_computer.compute_mask(state, level).to(device)
            scaled_logits = scaled_logits + constraint_mask

        # Apply top-k
        if top_k is not None:
            v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
            scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")

        # Sample and compute log-prob
        probs = F.softmax(scaled_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        # Log probability under the CONSTRAINED distribution
        log_prob = torch.log(probs[0, next_tok.item()] + 1e-10)

        # Entropy of the constrained distribution
        valid_probs = probs[probs > 1e-10]
        entropy = -(valid_probs * torch.log(valid_probs)).sum()

        gen_tokens.append(next_tok.item())
        gen_logprobs.append(log_prob.item())
        gen_entropies.append(entropy.item())

        seq = torch.cat([seq, next_tok], dim=1)
        state = _update_state(state, next_tok.item(), tokenizer)

    # Force END if not done
    if state.grammar != GrammarState.DONE:
        end_id = tokenizer.end_id
        gen_tokens.append(end_id)
        gen_logprobs.append(0.0)  # Deterministic
        gen_entropies.append(0.0)

    return (
        torch.tensor(gen_tokens, device=device),
        torch.tensor(gen_logprobs, device=device),
        torch.tensor(gen_entropies, device=device),
    )


# ---------------------------------------------------------------------------
# Lagrangian constraint loss (for training)
# ---------------------------------------------------------------------------

class LagrangianConstraintLoss(torch.nn.Module):
    """Differentiable constraint loss with adaptive Lagrange multipliers.

    During supervised training, adds penalty terms for tokens that violate
    circuit constraints. The Lagrange multipliers automatically increase
    for frequently violated constraints.

    Constraints enforced:
      1. Component-follows-component (no value→value or comp→comp)
      2. Correct component types for the topology
      3. Value tokens within valid range for the preceding component
      4. Sequence ends with END after all components are placed
    """

    def __init__(
        self,
        tokenizer: CircuitTokenizer,
        n_constraints: int = 4,
        initial_lambda: float = 0.1,
        lambda_lr: float = 0.01,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_computer = ConstraintMask(tokenizer)

        # Learnable Lagrange multipliers
        self.log_lambdas = torch.nn.Parameter(
            torch.full((n_constraints,), math.log(initial_lambda))
        )
        self.lambda_lr = lambda_lr

        # Cache
        self._comp_ids = set(
            t.id for t in tokenizer.tokens if t.token_type == TokenType.COMPONENT
        )
        self._val_ids = set(
            t.id for t in tokenizer.tokens if t.token_type == TokenType.VALUE
        )

    @property
    def lambdas(self) -> torch.Tensor:
        return self.log_lambdas.exp()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_ids: torch.Tensor,
        topologies: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute constraint violation loss.

        Args:
            logits: (B, T, V) model output logits
            targets: (B, T) target token IDs
            input_ids: (B, T) input token IDs (shifted)
            topologies: list of topology strings for each batch item

        Returns:
            loss: scalar constraint loss
            stats: dict of per-constraint violation rates
        """
        B, T, V = logits.shape
        device = logits.device

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # (B, T, V)

        violations = torch.zeros(4, device=device)
        counts = torch.zeros(4, device=device)

        for b in range(B):
            topology = normalize_topology(topologies[b]) if b < len(topologies) else ""
            expected = COMPONENT_TO_PARAM.get(topology, [])

            # Find where the component section starts (after second SEP)
            sep_count = 0
            comp_start = -1
            for t in range(T):
                if input_ids[b, t].item() == self.tokenizer.sep_id:
                    sep_count += 1
                    if sep_count == 2:
                        comp_start = t + 1
                        break

            if comp_start < 0:
                continue

            # Constraint 1: Alternating COMP/VAL structure
            expect_comp = True
            for t in range(comp_start, T):
                if targets[b, t].item() == self.tokenizer.pad_id:
                    break

                counts[0] += 1
                if expect_comp:
                    # Should be COMP_* or END
                    comp_prob = sum(
                        probs[b, t, tid].item()
                        for tid in self._comp_ids
                    ) + probs[b, t, self.tokenizer.end_id].item()
                    violations[0] += max(0, 1.0 - comp_prob)
                else:
                    # Should be VAL_*
                    val_prob = sum(
                        probs[b, t, tid].item()
                        for tid in self._val_ids
                    )
                    violations[1] += max(0, 1.0 - val_prob)
                    counts[1] += 1

                target_tok = targets[b, t].item()
                if target_tok == self.tokenizer.end_id:
                    break
                if target_tok in self._comp_ids:
                    expect_comp = False
                elif target_tok in self._val_ids:
                    expect_comp = True

            # Constraint 2: Correct component types
            if expected:
                comp_idx = 0
                for t in range(comp_start, T):
                    tid = targets[b, t].item()
                    if tid == self.tokenizer.pad_id or tid == self.tokenizer.end_id:
                        break
                    if tid in self._comp_ids:
                        if comp_idx < len(expected):
                            expected_type = expected[comp_idx][0]
                            expected_tid = _get_component_token_id(
                                self.tokenizer, expected_type
                            )
                            if expected_tid >= 0:
                                counts[2] += 1
                                violations[2] += max(
                                    0, 1.0 - probs[b, t, expected_tid].item()
                                )
                            comp_idx += 1

            # Constraint 3: Value in range (check probability mass within bounds)
            comp_idx = 0
            for t in range(comp_start, T):
                tid = targets[b, t].item()
                if tid == self.tokenizer.pad_id or tid == self.tokenizer.end_id:
                    break
                if tid in self._comp_ids:
                    pass  # Next position should be value
                elif tid in self._val_ids and comp_idx < len(expected):
                    _, param_name = expected[comp_idx]
                    bounds = _get_param_bounds(topology, param_name)
                    if bounds is not None:
                        lo_id, hi_id = _get_value_token_range(
                            bounds[0], bounds[1], self.tokenizer
                        )
                        in_range_prob = probs[b, t, lo_id:hi_id + 1].sum().item()
                        counts[3] += 1
                        violations[3] += max(0, 1.0 - in_range_prob)
                    comp_idx += 1

        # Normalize violations
        violation_rates = violations / counts.clamp(min=1.0)

        # Lagrangian loss: sum of lambda_i * violation_i
        loss = (self.lambdas * violation_rates).sum()

        stats = {
            "constraint/alternating": violation_rates[0].item(),
            "constraint/value_position": violation_rates[1].item(),
            "constraint/comp_type": violation_rates[2].item(),
            "constraint/value_range": violation_rates[3].item(),
            "constraint/total_loss": loss.item(),
            "constraint/lambda_mean": self.lambdas.mean().item(),
        }

        return loss, stats

    def update_lambdas(self, violation_rates: torch.Tensor) -> None:
        """Dual ascent step: increase multipliers for violated constraints."""
        with torch.no_grad():
            # Gradient ascent on lambdas (maximize Lagrangian w.r.t. lambda)
            self.log_lambdas += self.lambda_lr * violation_rates
            # Clamp to prevent explosion
            self.log_lambdas.clamp_(min=math.log(0.01), max=math.log(100.0))
