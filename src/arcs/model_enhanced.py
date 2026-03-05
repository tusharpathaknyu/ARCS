"""Enhanced ARCS model architectures.

Two additional architectures beyond the baseline GPT decoder:

1. TwoHeadARCSModel: Separate heads for component and value prediction
   - Structure head (weight-tied with token embedding): structural token prediction
   - Value head (independent weights + value-context MLP): value token prediction
   - During training, routes loss through the appropriate head per target position
   - During generation, dynamically selects head based on context

2. GraphTransformerARCSModel: Graph-structure-aware transformer
   - Attention bias from KNOWN circuit topology (not heuristic sequence position)
   - Per-topology adjacency tables define which components share circuit nets
   - Edge-type embedding for component-pair relationships
   - Walk position embeddings for Eulerian-augmented orderings
   - Two-head output (structure + value)

Key insight from studying AnalogGenie:
    AnalogGenie uses pin-level tokens where adjacency IS the walk sequence.
    ARCS uses component-level tokens, so adjacency must come from the KNOWN
    topology graph. This is actually an advantage — we inject the correct
    graph structure as an explicit inductive bias, rather than hoping the
    model discovers it from flat sequences.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arcs.model import (
    ARCSConfig,
    ARCSModel,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
)
from arcs.tokenizer import CircuitTokenizer, TokenType


# ---------------------------------------------------------------------------
# Topology-aware circuit adjacency tables
# ---------------------------------------------------------------------------

# For each topology: list of (component_index_i, component_index_j) pairs that
# are directly connected in the circuit (share a net/node).
# Component indices follow _params_to_components() ordering from tokenizer.py,
# which iterates params dict in insertion order (Python 3.7+ guaranteed).
#
# These adjacencies are derived from the actual SPICE circuit schematics in
# templates.py. Each pair means the two components are electrically connected.

TOPOLOGY_ADJACENCY: dict[str, list[tuple[int, int]]] = {
    # ==== Tier 1: Power Converters ====

    # Buck: INDUCTOR(0), CAPACITOR(1), RESISTOR/ESR(2), MOSFET_N(3)
    # VIN -> MOSFET -> switch_node -> INDUCTOR -> output -> CAP(+ESR) -> GND
    "buck": [(0, 3), (0, 1), (1, 2)],

    # Boost: INDUCTOR(0), CAPACITOR(1), RESISTOR/ESR(2), MOSFET_N(3)
    # VIN -> INDUCTOR -> switch_node -> MOSFET -> GND; switch_node -> D -> CAP
    "boost": [(0, 3), (0, 1), (1, 2)],

    # Buck-Boost: INDUCTOR(0), CAPACITOR(1), RESISTOR/ESR(2), MOSFET_N(3)
    "buck_boost": [(0, 3), (0, 1), (1, 2)],

    # Cuk: IND1(0), IND2(1), CAP_coupling(2), CAP_out(3), ESR(4), MOSFET(5)
    # VIN -> L1 -> MOSFET -> GND; L1 <-> Cc <-> L2 -> Cout(+ESR) -> load
    "cuk": [(0, 5), (0, 2), (2, 1), (1, 3), (3, 4)],

    # SEPIC: IND1(0), IND2(1), CAP_coupling(2), CAP_out(3), ESR(4), MOSFET(5)
    "sepic": [(0, 5), (0, 2), (2, 1), (1, 3), (3, 4)],

    # Flyback: IND_primary(0), TRANSFORMER(1), CAP(2), ESR(3), MOSFET(4)
    "flyback": [(0, 4), (0, 1), (1, 2), (2, 3)],

    # Forward: IND_primary(0), TRANSFORMER(1), IND_output(2), CAP(3), ESR(4), MOSFET(5)
    "forward": [(0, 5), (0, 1), (1, 2), (2, 3), (3, 4)],

    # ==== Tier 2: Signal Processing ====

    # Inverting amp: R_input(0), R_feedback(1) -- both at op-amp inverting input
    "inverting_amp": [(0, 1)],

    # Non-inverting amp: R_ground(0), R_feedback(1) -- both at inverting input
    "noninverting_amp": [(0, 1)],

    # Instrumentation: R1(0), R_gain(1), R2(2), R3(3)
    "instrumentation_amp": [(0, 1), (2, 3)],

    # Differential amp: R1(0), R2(1)
    "differential_amp": [(0, 1)],

    # Sallen-Key LP: R1(0), R2(1), C1(2), C2(3)
    # R1 -> node_A -> R2 -> node_B -> opamp; C1: A->out; C2: B->GND
    "sallen_key_lowpass": [(0, 1), (0, 2), (1, 2), (1, 3)],

    # Sallen-Key HP: R1(0), R2(1), C1(2), C2(3)
    # C1 -> node_A -> C2 -> node_B -> opamp; R1: A->GND; R2: B->out
    "sallen_key_highpass": [(2, 3), (0, 2), (1, 3), (0, 1)],

    # Sallen-Key BP (MFB): R1(0), R2(1), R3(2), C1(3), C2(4)
    "sallen_key_bandpass": [(0, 1), (0, 3), (1, 4), (2, 3), (3, 4)],

    # Wien bridge: R_freq(0), C_freq(1), R_feedback(2), R_ground(3)
    "wien_bridge": [(0, 1), (2, 3)],

    # Colpitts: L(0), C1(1), C2(2), Rb1(3), Rb2(4), Re(5), Rc(6)
    # Tank: L <-> C1 <-> C2; Bias divider: Rb1 <-> Rb2; BJT: Re, Rc
    "colpitts": [(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (4, 6)],
}


# Component type IDs for edge-type embedding
_COMP_TYPE_TO_IDX = {
    "COMP_MOSFET_N": 1, "COMP_MOSFET_P": 2,
    "COMP_RESISTOR": 3, "COMP_CAPACITOR": 4,
    "COMP_INDUCTOR": 5, "COMP_DIODE": 6,
    "COMP_BJT_NPN": 7, "COMP_BJT_PNP": 8,
    "COMP_OPAMP": 9, "COMP_TRANSFORMER": 10,
    "COMP_ZENER": 11, "COMP_LED": 12,
    "COMP_TRIAC": 13, "COMP_IGBT": 14,
    "COMP_JFET_N": 15,
}


def _extract_topology(
    input_ids: torch.Tensor,
    topo_name_to_id: dict[str, int],
) -> list[str | None]:
    """Extract topology name from TOPO_X token in each sequence."""
    id_to_topo: dict[int, str] = {}
    for topo_name, tid in topo_name_to_id.items():
        if topo_name.startswith("TOPO_"):
            id_to_topo[tid] = topo_name[5:].lower()

    B = input_ids.shape[0]
    results: list[str | None] = []
    for b in range(B):
        found = None
        for t in range(min(5, input_ids.shape[1])):  # TOPO token is near start
            tid = input_ids[b, t].item()
            if tid in id_to_topo:
                found = id_to_topo[tid]
                break
        results.append(found)
    return results


# ---------------------------------------------------------------------------
# 1. Two-Head Architecture
# ---------------------------------------------------------------------------


class TwoHeadARCSModel(nn.Module):
    """GPT with separate component and value prediction heads.

    The insight: predicting the next structural token (which component? which
    topology? which spec?) is fundamentally different from predicting component
    values (which discretized value bin?). Giving each task its own head lets
    each specialise.

    Architecture:
        Shared transformer backbone (embeddings + N blocks + final norm)
        +-- structure_head: Linear(d_model, vocab_size)  [weight-tied with tok_emb]
        +-- value_proj -> value_head: Linear(d_model, vocab_size)  [independent]

    During training:
        - value_mask tells which target positions are VALUE tokens
        - VALUE positions use value_head logits for loss
        - Non-VALUE positions use structure_head logits for loss

    During generation:
        - After a COMP_* token -> next is a value -> use value_head
        - Otherwise -> use structure_head
    """

    def __init__(self, config: ARCSConfig):
        super().__init__()
        self.config = config

        # --- Shared backbone (identical to ARCSModel) ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.type_emb = nn.Embedding(config.n_token_types, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.d_model)

        # --- Head 1: Structure head (weight-tied) ---
        self.structure_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # --- Head 2: Value head with context projection ---
        self.value_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model, bias=False),
        )
        self.value_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.structure_head.weight = self.tok_emb.weight

        # --- Initialization ---
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.proj.weight,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )
            nn.init.normal_(
                block.ffn.w2.weight,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _backbone(
        self,
        input_ids: torch.Tensor,
        token_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shared transformer backbone -> hidden states (B, T, d_model)."""
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        if token_types is not None:
            x = x + self.type_emb(token_types)
        x = self.emb_drop(x)

        for block in self.blocks:
            x = block(x)

        return self.ln_f(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_types: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        value_weight: float = 5.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with two-head routing."""
        h = self._backbone(input_ids, token_types)

        struct_logits = self.structure_head(h)
        value_h = self.value_proj(h) + h  # residual
        val_logits = self.value_head(value_h)

        if value_mask is not None:
            vm = value_mask.unsqueeze(-1)
            logits = torch.where(vm, val_logits, struct_logits)
        else:
            logits = struct_logits

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
                vm_flat = value_mask.view(-1).float()
                weights = 1.0 + (value_weight - 1.0) * vm_flat
                pad_mask = (targets_flat != self.config.pad_id).float()
                weights = weights * pad_mask
                loss = (ce * weights).sum() / weights.sum().clamp(min=1.0)
            else:
                non_pad = targets_flat != self.config.pad_id
                loss = ce[non_pad].mean()

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        token_types_prefix: Optional[torch.Tensor] = None,
        tokenizer: Optional[CircuitTokenizer] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with two-head routing."""
        self.eval()
        seq = prefix.clone()
        tseq = token_types_prefix.clone() if token_types_prefix is not None else None
        end_id = 2

        comp_ids: set[int] = set()
        if tokenizer is not None:
            for tok in tokenizer.tokens:
                if tok.token_type == TokenType.COMPONENT:
                    comp_ids.add(tok.id)

        for _ in range(max_new_tokens):
            if seq.shape[1] > self.config.max_seq_len:
                s = seq[:, -self.config.max_seq_len:]
                t = tseq[:, -self.config.max_seq_len:] if tseq is not None else None
            else:
                s = seq
                t = tseq

            h = self._backbone(s, t)
            h_last = h[:, -1:, :]

            last_tok = seq[0, -1].item()
            if last_tok in comp_ids:
                logits = self.value_head(self.value_proj(h_last) + h_last)
            else:
                logits = self.structure_head(h_last)

            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_tok], dim=1)

            if tseq is not None:
                new_type_val = 0
                if tokenizer is not None:
                    tok_id = next_tok.item()
                    if 0 <= tok_id < len(tokenizer.tokens):
                        new_type_val = tokenizer.tokens[tok_id].token_type.value - 1
                new_type = torch.tensor(
                    [[new_type_val]], dtype=torch.long, device=seq.device
                )
                tseq = torch.cat([tseq, new_type], dim=1)

            if next_tok.item() == end_id:
                break

        return seq

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_group(self) -> dict[str, int]:
        groups = {
            "embeddings": 0,
            "attention": 0,
            "ffn": 0,
            "norm": 0,
            "structure_head": 0,
            "value_proj": 0,
            "value_head": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if "emb" in name:
                groups["embeddings"] += n
            elif "attn" in name or "qkv" in name:
                groups["attention"] += n
            elif "ffn" in name or ("w1" in name or "w2" in name or "w3" in name):
                groups["ffn"] += n
            elif "ln" in name or "norm" in name:
                groups["norm"] += n
            elif "value_proj" in name:
                groups["value_proj"] += n
            elif "value_head" in name:
                groups["value_head"] += n
            elif "structure_head" in name:
                groups["structure_head"] += n
        return groups


# ---------------------------------------------------------------------------
# 2. Graph Transformer Architecture
# ---------------------------------------------------------------------------


class GraphAwareCausalAttention(nn.Module):
    """Causal self-attention with circuit graph structure bias.

    Uses topology-aware adjacency (from TOPOLOGY_ADJACENCY) to bias attention
    toward circuit-connected component pairs. This is the correct inductive bias
    for component-level tokenization, inspired by AnalogGenie's pin-level
    approach but adapted for our richer representation.
    """

    def __init__(self, config: ARCSConfig, n_edge_types: int = 16):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Per-head scalar for adjacent component pairs
        self.adj_bias = nn.Parameter(torch.zeros(config.n_heads))

        # Per-head bias for each component-type pair
        self.edge_type_bias = nn.Embedding(n_edge_types + 1, config.n_heads)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1
            ).bool(),
        )

    def forward(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if graph_adj is not None:
            adj_b = self.adj_bias.view(1, self.n_heads, 1, 1)
            attn = attn + adj_b * graph_adj.unsqueeze(1)

        if edge_types is not None:
            etype_b = self.edge_type_bias(edge_types)  # (B, T, T, nh)
            attn = attn + etype_b.permute(0, 3, 1, 2)

        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class GraphTransformerBlock(nn.Module):
    """Transformer block with graph-aware attention."""

    def __init__(self, config: ARCSConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = GraphAwareCausalAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), graph_adj=graph_adj, edge_types=edge_types)
        x = x + self.ffn(self.ln2(x))
        return x


class GraphTransformerARCSModel(nn.Module):
    """Graph Transformer for circuit generation with topology-aware attention.

    Key design informed by studying AnalogGenie:

    AnalogGenie: Pin-level tokens -> adjacency IS the walk sequence -> model
    learns structure implicitly from flat autoregressive sequence.

    ARCS: Component-level tokens -> adjacency must be INJECTED as structural
    bias. We use the KNOWN topology graph (from TOPO_X token) to build
    component-level adjacency matrices via TOPOLOGY_ADJACENCY. This gives
    the model an explicit inductive bias about which components interact.

    Architecture:
        Token embeddings + position + type + walk_position
        N x GraphTransformerBlock (graph-biased attention + SwiGLU FFN)
        Two-head output (structure + value)
    """

    def __init__(self, config: ARCSConfig):
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.type_emb = nn.Embedding(config.n_token_types, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # Walk position: meaningful for Eulerian-augmented sequences
        self.walk_pos_emb = nn.Embedding(32, config.d_model)

        # --- Graph Transformer blocks ---
        self.blocks = nn.ModuleList(
            [GraphTransformerBlock(config) for _ in range(config.n_layers)]
        )

        # --- Two-head output ---
        self.ln_f = RMSNorm(config.d_model)
        self.structure_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.value_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model, bias=False),
        )
        self.value_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.structure_head.weight = self.tok_emb.weight

        # --- Initialization ---
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.proj.weight,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )
            nn.init.normal_(
                block.ffn.w2.weight,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def compute_graph_features(
        input_ids: torch.Tensor,
        tokenizer: CircuitTokenizer,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute topology-aware graph features from token sequences.

        Uses TOPOLOGY_ADJACENCY to build correct component-level adjacency
        based on the actual circuit topology, not heuristic sequence position.

        Returns:
            graph_adj:  (B, T, T) float -- 1.0 where components are circuit-adjacent
            edge_types: (B, T, T) long  -- component type pair indices
            walk_pos:   (B, T) long     -- walk position (component index)
        """
        B, T = input_ids.shape
        device = input_ids.device

        graph_adj = torch.zeros(B, T, T, device=device)
        edge_types = torch.zeros(B, T, T, dtype=torch.long, device=device)
        walk_pos = torch.zeros(B, T, dtype=torch.long, device=device)

        # Build lookups
        comp_ids: set[int] = set()
        comp_id_to_name: dict[int, str] = {}
        topo_name_to_id: dict[str, int] = {}

        for tok in tokenizer.tokens:
            if tok.token_type == TokenType.COMPONENT:
                comp_ids.add(tok.id)
                comp_id_to_name[tok.id] = tok.name
            elif tok.token_type == TokenType.TOPOLOGY:
                topo_name_to_id[tok.name] = tok.id

        topologies = _extract_topology(input_ids, topo_name_to_id)

        for b in range(B):
            ids = input_ids[b].tolist()
            topo = topologies[b]

            # Find component positions and types
            comp_positions: list[int] = []
            comp_names: list[str] = []
            wp = 0

            for i, tid in enumerate(ids):
                if tid in comp_ids:
                    comp_positions.append(i)
                    comp_names.append(comp_id_to_name[tid])
                    walk_pos[b, i] = min(wp, 31)
                    wp += 1
                    # Mark following value token with same walk position
                    if i + 1 < T:
                        walk_pos[b, i + 1] = min(wp - 1, 31)

            n_comp = len(comp_positions)
            if n_comp == 0:
                continue

            # Build adjacency from topology graph
            adj_pairs = TOPOLOGY_ADJACENCY.get(topo, []) if topo else []

            for ci, cj in adj_pairs:
                if ci < n_comp and cj < n_comp:
                    pi, pj = comp_positions[ci], comp_positions[cj]
                    # Component positions are adjacent
                    graph_adj[b, pi, pj] = 1.0
                    graph_adj[b, pj, pi] = 1.0
                    # Value tokens (comp+1) get weaker adjacency
                    if pi + 1 < T:
                        graph_adj[b, pi + 1, pj] = 0.5
                        if pj + 1 < T:
                            graph_adj[b, pi + 1, pj + 1] = 0.5
                    if pj + 1 < T:
                        graph_adj[b, pj + 1, pi] = 0.5

            # Edge types from component name pairs
            for ci in range(n_comp):
                pi = comp_positions[ci]
                type_i = _COMP_TYPE_TO_IDX.get(comp_names[ci], 0)
                for cj in range(n_comp):
                    if ci == cj:
                        continue
                    pj = comp_positions[cj]
                    type_j = _COMP_TYPE_TO_IDX.get(comp_names[cj], 0)
                    pair_idx = (min(type_i, type_j) + max(type_i, type_j)) % 16
                    edge_types[b, pi, pj] = max(pair_idx, 1)

        return graph_adj, edge_types, walk_pos

    def forward(
        self,
        input_ids: torch.Tensor,
        token_types: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        value_mask: Optional[torch.Tensor] = None,
        value_weight: float = 5.0,
        graph_adj: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
        walk_pos: Optional[torch.Tensor] = None,
        tokenizer: Optional[CircuitTokenizer] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with graph-aware attention.

        Graph features can be pre-computed or computed on-the-fly from tokenizer.
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len

        if graph_adj is None and tokenizer is not None:
            graph_adj, edge_types, walk_pos = self.compute_graph_features(
                input_ids, tokenizer
            )

        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        if token_types is not None:
            x = x + self.type_emb(token_types)
        if walk_pos is not None:
            x = x + self.walk_pos_emb(walk_pos.clamp(0, 31))
        x = self.emb_drop(x)

        for block in self.blocks:
            x = block(x, graph_adj=graph_adj, edge_types=edge_types)

        x = self.ln_f(x)

        struct_logits = self.structure_head(x)
        val_h = self.value_proj(x) + x
        val_logits = self.value_head(val_h)

        if value_mask is not None:
            vm = value_mask.unsqueeze(-1)
            logits = torch.where(vm, val_logits, struct_logits)
        else:
            logits = struct_logits

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
                vm_flat = value_mask.view(-1).float()
                weights = 1.0 + (value_weight - 1.0) * vm_flat
                pad_mask = (targets_flat != self.config.pad_id).float()
                weights = weights * pad_mask
                loss = (ce * weights).sum() / weights.sum().clamp(min=1.0)
            else:
                non_pad = targets_flat != self.config.pad_id
                loss = ce[non_pad].mean()

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        token_types_prefix: Optional[torch.Tensor] = None,
        tokenizer: Optional[CircuitTokenizer] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with graph-aware attention."""
        self.eval()
        seq = prefix.clone()
        tseq = token_types_prefix.clone() if token_types_prefix is not None else None
        end_id = 2

        comp_ids: set[int] = set()
        if tokenizer is not None:
            for tok in tokenizer.tokens:
                if tok.token_type == TokenType.COMPONENT:
                    comp_ids.add(tok.id)

        for _ in range(max_new_tokens):
            if seq.shape[1] > self.config.max_seq_len:
                s = seq[:, -self.config.max_seq_len:]
                t = tseq[:, -self.config.max_seq_len:] if tseq is not None else None
            else:
                s = seq
                t = tseq

            # Compute graph features for current sequence
            if tokenizer is not None:
                g_adj, e_types, w_pos = self.compute_graph_features(s, tokenizer)
            else:
                g_adj = e_types = w_pos = None

            # Full forward through embedding + blocks
            B_s, T_s = s.shape
            positions = torch.arange(T_s, device=s.device)
            x = self.tok_emb(s) + self.pos_emb(positions)
            if t is not None:
                x = x + self.type_emb(t)
            if w_pos is not None:
                x = x + self.walk_pos_emb(w_pos.clamp(0, 31))
            x = self.emb_drop(x)

            for block in self.blocks:
                x = block(x, graph_adj=g_adj, edge_types=e_types)

            x = self.ln_f(x)
            h_last = x[:, -1:, :]

            last_tok = seq[0, -1].item()
            if last_tok in comp_ids:
                logits = self.value_head(self.value_proj(h_last) + h_last)
            else:
                logits = self.structure_head(h_last)

            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_tok], dim=1)

            if tseq is not None:
                new_type_val = 0
                if tokenizer is not None:
                    tok_id = next_tok.item()
                    if 0 <= tok_id < len(tokenizer.tokens):
                        new_type_val = tokenizer.tokens[tok_id].token_type.value - 1
                new_type = torch.tensor(
                    [[new_type_val]], dtype=torch.long, device=seq.device
                )
                tseq = torch.cat([tseq, new_type], dim=1)

            if next_tok.item() == end_id:
                break

        return seq

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_group(self) -> dict[str, int]:
        groups = {
            "embeddings": 0,
            "walk_pos_emb": 0,
            "attention": 0,
            "graph_bias": 0,
            "ffn": 0,
            "norm": 0,
            "structure_head": 0,
            "value_proj": 0,
            "value_head": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if "walk_pos_emb" in name:
                groups["walk_pos_emb"] += n
            elif "adj_bias" in name or "edge_type_bias" in name:
                groups["graph_bias"] += n
            elif "emb" in name:
                groups["embeddings"] += n
            elif "attn" in name or "qkv" in name:
                groups["attention"] += n
            elif "ffn" in name or ("w1" in name or "w2" in name or "w3" in name):
                groups["ffn"] += n
            elif "ln" in name or "norm" in name:
                groups["norm"] += n
            elif "value_proj" in name:
                groups["value_proj"] += n
            elif "value_head" in name:
                groups["value_head"] += n
            elif "structure_head" in name:
                groups["structure_head"] += n
        return groups


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_TYPES = {
    "baseline": ARCSModel,
    "two_head": TwoHeadARCSModel,
    "graph_transformer": GraphTransformerARCSModel,
}


def create_model(
    model_type: str,
    config: ARCSConfig,
) -> nn.Module:
    """Factory function to create model by type."""
    cls = MODEL_TYPES.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_TYPES.keys())}"
        )
    return cls(config)


def load_model(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    model_type: Optional[str] = None,
) -> Tuple[nn.Module, ARCSConfig, str]:
    """Load a model from checkpoint, auto-detecting model type.

    Returns:
        (model, config, model_type) tuple
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ARCSConfig.from_dict(ckpt["config"])

    # Auto-detect model type from state dict keys when not stored in checkpoint
    mt = ckpt.get("model_type", model_type)
    if mt is None:
        state_keys = set(ckpt.get("model_state_dict", {}).keys())
        if "walk_pos_emb.weight" in state_keys:
            mt = "graph_transformer"
        elif "value_head.weight" in state_keys:
            mt = "two_head"
        else:
            mt = "baseline"

    model = create_model(mt, config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, mt
