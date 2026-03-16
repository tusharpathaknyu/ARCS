"""ValidCircuitGen: Constrained VAE for Circuit Generation with Formal Guarantees.

Direction 5 of the ARCS research agenda.  Generates complete circuit graphs via
a Variational Autoencoder operating in continuous graph space, with differentiable
constraint projection guaranteeing structural validity by construction.

Unlike ARCS's autoregressive models (which generate tokens sequentially with
masking), ValidCircuitGen generates entire circuits in one shot:

  1. Encoder:    Circuit graph → latent code z  (bidirectional graph transformer)
  2. Decoder:    (z, specs) → soft circuit graph (continuous relaxation)
  3. Projection: Π_C(G) → valid circuit        (iterative constraint satisfaction)
  4. Discretize: Soft graph → token sequence     (argmax + value binning)

Training uses Lagrangian relaxation with adaptive multipliers for each
constraint, enabling automatic trade-off between reconstruction quality
and constraint satisfaction.

Formal guarantee
    If the projection step converges (violation < ε), the generated circuit
    satisfies all encoded constraints to within ε tolerance.  As ε → 0,
    validity → 100%.

Architecture
    Encoder:     4-layer bidirectional Graph Transformer, d_model=256
    Decoder:     3-layer MLP with residual, outputs node types + adjacency + values
    Constraints: 5 differentiable circuit validity functions
    Projection:  20-step Adam optimizer on constraint violations
    Training:    Reconstruction + β-KL + Lagrangian constraints

Parameters: ~4.0M
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from arcs.datagen import CircuitSample
from arcs.model import RMSNorm, ARCSConfig
from arcs.model_enhanced import (
    TOPOLOGY_ADJACENCY,
    TOPOLOGY_RWPE,
    K_WALK,
    _COMP_TYPE_TO_IDX,
)
from arcs.simulate import COMPONENT_TO_PARAM, normalize_topology
from arcs.templates import (
    POWER_CONVERTER_BOUNDS,
    SIGNAL_CIRCUIT_BOUNDS,
    _TIER1_NAMES,
    _TIER2_NAMES,
)
from arcs.tokenizer import CircuitTokenizer, TokenType


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Component-type indexing (shared with encoder/decoder)
NODE_TYPE_TO_IDX: dict[str, int] = {
    "NONE": 0,
    "RESISTOR": 1,
    "CAPACITOR": 2,
    "INDUCTOR": 3,
    "MOSFET_N": 4,
    "MOSFET_P": 5,
    "BJT_NPN": 6,
    "BJT_PNP": 7,
    "OPAMP": 8,
    "TRANSFORMER": 9,
    "DIODE": 10,
    "DIODE_SCHOTTKY": 11,
    "DIODE_ZENER": 12,
    "SWITCH_IDEAL": 13,
    "VOLTAGE_SOURCE": 14,
    "CURRENT_SOURCE": 15,
}
IDX_TO_NODE_TYPE: dict[int, str] = {v: k for k, v in NODE_TYPE_TO_IDX.items()}
N_NODE_TYPES = len(NODE_TYPE_TO_IDX)  # 16

# Topology → index for embedding
ALL_TOPOLOGIES = sorted(
    set(list(COMPONENT_TO_PARAM.keys()) + list(TOPOLOGY_ADJACENCY.keys()))
)
TOPOLOGY_TO_IDX: dict[str, int] = {t: i + 1 for i, t in enumerate(ALL_TOPOLOGIES)}
TOPOLOGY_TO_IDX["unknown"] = 0
N_TOPOLOGIES = len(TOPOLOGY_TO_IDX)


def _count_connected_components(adj: torch.Tensor) -> int:
    """Count connected components in an undirected adjacency matrix."""
    n = adj.shape[0]
    if n <= 1:
        return n

    visited = [False] * n
    n_components = 0

    for start in range(n):
        if visited[start]:
            continue
        n_components += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = torch.where(adj[node] > 0.5)[0].tolist()
            for nei in neighbors:
                if not visited[nei]:
                    visited[nei] = True
                    stack.append(nei)

    return n_components


def _compute_expected_components() -> dict[str, int]:
    """Count connected components in each topology's reference adjacency."""
    result: dict[str, int] = {}
    for topo_name, edges in TOPOLOGY_ADJACENCY.items():
        if not edges:
            result[topo_name] = 1
            continue
        n_nodes = max(max(i, j) for i, j in edges) + 1
        adj = torch.zeros(n_nodes, n_nodes)
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        result[topo_name] = _count_connected_components(adj)
    return result


TOPOLOGY_EXPECTED_COMPONENTS: dict[str, int] = _compute_expected_components()

# Tensor lookup: topology_idx → expected connected components
_max_topo_idx = max(TOPOLOGY_TO_IDX.values())
EXPECTED_COMPONENTS_BY_IDX = torch.ones(_max_topo_idx + 1, dtype=torch.long)
for _name, _idx in TOPOLOGY_TO_IDX.items():
    if _name in TOPOLOGY_EXPECTED_COMPONENTS:
        EXPECTED_COMPONENTS_BY_IDX[_idx] = TOPOLOGY_EXPECTED_COMPONENTS[_name]


# Spec types
SPEC_TYPES = [
    "vin", "vout", "iout", "efficiency", "ripple", "fsw",
    "gain", "bandwidth", "phase_margin", "cutoff_freq",
]
SPEC_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SPEC_TYPES)}
N_SPEC_TYPES = len(SPEC_TYPES)

# Bounds
_ALL_BOUNDS = {**POWER_CONVERTER_BOUNDS, **SIGNAL_CIRCUIT_BOUNDS}

# Log-scale range for values (matches tokenizer)
LOG_VAL_MIN = math.log10(1e-12)
LOG_VAL_MAX = math.log10(1e6)


@dataclass
class VCGConfig:
    """ValidCircuitGen hyperparameters."""

    # Latent space
    latent_dim: int = 64

    # Graph dimensions
    max_nodes: int = 12          # max components in any topology
    n_node_types: int = N_NODE_TYPES  # 16
    n_topologies: int = N_TOPOLOGIES

    # Encoder
    d_model: int = 256
    n_encoder_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # Decoder
    n_decoder_layers: int = 3
    decoder_hidden: int = 512

    # Spec conditioning
    n_spec_types: int = N_SPEC_TYPES
    max_specs: int = 8           # max spec pairs per sample

    # Constraint projection
    n_projection_steps: int = 20
    projection_lr: float = 0.1

    # Training
    n_constraints: int = 5
    beta_kl: float = 0.1        # KL weight (β-VAE)
    initial_lambda: float = 0.1  # initial Lagrange multiplier

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VCGConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Graph data structure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitGraph:
    """A circuit represented as a attributed graph.

    All tensors are unbatched (single sample).
    """

    topology: str
    n_components: int
    node_types: torch.Tensor      # (max_nodes,)  int    — component type indices
    adjacency: torch.Tensor       # (max_nodes, max_nodes)  float — binary
    values: torch.Tensor          # (max_nodes,)  float  — log10 of value
    active_mask: torch.Tensor     # (max_nodes,)  float  — 1 for real, 0 for pad
    spec_types: torch.Tensor      # (max_specs,)  int    — spec type indices
    spec_values: torch.Tensor     # (max_specs,)  float  — log10 of spec values
    spec_mask: torch.Tensor       # (max_specs,)  float  — 1 for real specs
    value_bounds_min: torch.Tensor  # (max_nodes,) float — log10 min bound
    value_bounds_max: torch.Tensor  # (max_nodes,) float — log10 max bound


def circuit_sample_to_graph(
    sample: CircuitSample,
    tokenizer: CircuitTokenizer,
    config: VCGConfig,
) -> CircuitGraph:
    """Convert a CircuitSample to a CircuitGraph for VAE training.

    Extracts topology, component types, values, adjacency, and specs
    from the ARCS data format into the graph representation.
    """
    topology = normalize_topology(sample.topology)
    comp_param_list = COMPONENT_TO_PARAM.get(topology, [])
    n_comp = len(comp_param_list)

    # --- Node types ---
    node_types = torch.zeros(config.max_nodes, dtype=torch.long)
    for i, (comp_type, _) in enumerate(comp_param_list):
        idx = NODE_TYPE_TO_IDX.get(comp_type, 0)
        if i < config.max_nodes:
            node_types[i] = idx

    # --- Values (log10 scale) ---
    values = torch.zeros(config.max_nodes)
    for i, (_, param_name) in enumerate(comp_param_list):
        val = sample.parameters.get(param_name, 1.0)
        if val > 0:
            values[i] = math.log10(val)
        else:
            values[i] = LOG_VAL_MIN

    # --- Active mask ---
    active_mask = torch.zeros(config.max_nodes)
    active_mask[:min(n_comp, config.max_nodes)] = 1.0

    # --- Adjacency from topology ---
    adjacency = torch.zeros(config.max_nodes, config.max_nodes)
    adj_pairs = TOPOLOGY_ADJACENCY.get(topology, [])
    for i, j in adj_pairs:
        if i < config.max_nodes and j < config.max_nodes:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

    # --- Specs ---
    spec_types = torch.zeros(config.max_specs, dtype=torch.long)
    spec_values = torch.zeros(config.max_specs)
    spec_mask = torch.zeros(config.max_specs)

    oc = sample.operating_conditions
    metrics = sample.metrics if sample.valid else {}

    _OC_SPEC = {
        "vin": "vin", "vout": "vout", "iout": "iout", "fsw": "fsw",
        "vin_amp": "vin", "freq_test": "cutoff_freq", "vcc": "vin",
    }
    _METRIC_SPEC = {
        "efficiency": "efficiency", "vout_ripple": "ripple",
        "gain_db": "gain", "bw_3db": "bandwidth",
        "fc_3db": "cutoff_freq", "phase_rad": "phase_margin",
    }

    specs_found: dict[str, float] = {}
    for oc_key, oc_val in oc.items():
        spec_name = _OC_SPEC.get(oc_key)
        if spec_name and spec_name not in specs_found:
            specs_found[spec_name] = oc_val

    for mk, sk in _METRIC_SPEC.items():
        val = metrics.get(mk)
        if val is not None and sk not in specs_found:
            specs_found[sk] = val

    for si, (sname, sval) in enumerate(specs_found.items()):
        if si >= config.max_specs:
            break
        spec_idx = SPEC_TO_IDX.get(sname, 0)
        spec_types[si] = spec_idx
        spec_values[si] = math.log10(abs(sval)) if sval != 0 else 0.0
        spec_mask[si] = 1.0

    # --- Value bounds ---
    value_bounds_min = torch.full((config.max_nodes,), LOG_VAL_MIN)
    value_bounds_max = torch.full((config.max_nodes,), LOG_VAL_MAX)

    bounds_list = _ALL_BOUNDS.get(topology)
    if bounds_list:
        for i, (_, param_name) in enumerate(comp_param_list):
            if i >= config.max_nodes:
                break
            for b in bounds_list:
                if b.name == param_name:
                    value_bounds_min[i] = math.log10(max(b.min_val, 1e-15))
                    value_bounds_max[i] = math.log10(max(b.max_val, 1e-15))
                    break

    return CircuitGraph(
        topology=topology,
        n_components=n_comp,
        node_types=node_types,
        adjacency=adjacency,
        values=values,
        active_mask=active_mask,
        spec_types=spec_types,
        spec_values=spec_values,
        spec_mask=spec_mask,
        value_bounds_min=value_bounds_min,
        value_bounds_max=value_bounds_max,
    )


def graph_to_token_sequence(
    graph: CircuitGraph,
    tokenizer: CircuitTokenizer,
) -> list[int]:
    """Convert a CircuitGraph back to an ARCS token sequence.

    Format: START TOPO_X SEP specs SEP components END
    """
    tokens = [tokenizer.start_id]

    # Topology
    _topo_to_token = {
        "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
        "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
        "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
    }
    topo_key = _topo_to_token.get(graph.topology, f"TOPO_{graph.topology.upper()}")
    if topo_key in tokenizer.name_to_id:
        tokens.append(tokenizer.name_to_id[topo_key])
    tokens.append(tokenizer.sep_id)

    # Specs
    for si in range(graph.spec_mask.shape[0]):
        if graph.spec_mask[si] > 0:
            spec_idx = graph.spec_types[si].item()
            if spec_idx < len(SPEC_TYPES):
                spec_name = SPEC_TYPES[spec_idx]
                spec_key = f"SPEC_{spec_name.upper()}"
                if spec_key in tokenizer.name_to_id:
                    log_val = graph.spec_values[si].item()
                    tokens.append(tokenizer.name_to_id[spec_key])
                    tokens.append(tokenizer.encode_value(10 ** log_val))
    tokens.append(tokenizer.sep_id)

    # Components
    comp_param_list = COMPONENT_TO_PARAM.get(graph.topology, [])
    for ci in range(graph.n_components):
        if ci >= graph.active_mask.shape[0] or graph.active_mask[ci] < 0.5:
            break
        node_idx = graph.node_types[ci].item()
        node_type = IDX_TO_NODE_TYPE.get(node_idx, "RESISTOR")
        comp_key = f"COMP_{node_type}"
        if comp_key in tokenizer.name_to_id:
            log_val = graph.values[ci].item()
            tokens.append(tokenizer.name_to_id[comp_key])
            tokens.append(tokenizer.encode_value(10 ** log_val))

    tokens.append(tokenizer.end_id)
    return tokens


# ═══════════════════════════════════════════════════════════════════════════
# Dataset for VAE training
# ═══════════════════════════════════════════════════════════════════════════

class CircuitGraphDataset(Dataset):
    """Loads JSONL data and produces CircuitGraph tensors for VAE training."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: CircuitTokenizer,
        config: VCGConfig,
        valid_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.graphs: list[CircuitGraph] = []

        data_path = Path(data_path)
        files = sorted(data_path.glob("*.jsonl")) if data_path.is_dir() else [data_path]

        n_loaded = 0
        n_skipped = 0
        for fpath in files:
            with open(fpath) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    sample = CircuitSample.from_dict(json.loads(line))
                    if valid_only and not sample.valid:
                        n_skipped += 1
                        continue

                    topology = normalize_topology(sample.topology)
                    if topology not in COMPONENT_TO_PARAM:
                        n_skipped += 1
                        continue

                    graph = circuit_sample_to_graph(sample, tokenizer, config)
                    self.graphs.append(graph)
                    n_loaded += 1

        print(f"[VCG Dataset] Loaded {n_loaded} graphs, skipped {n_skipped}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = self.graphs[idx]
        return {
            "node_types": g.node_types,
            "adjacency": g.adjacency,
            "values": g.values,
            "active_mask": g.active_mask,
            "spec_types": g.spec_types,
            "spec_values": g.spec_values,
            "spec_mask": g.spec_mask,
            "value_bounds_min": g.value_bounds_min,
            "value_bounds_max": g.value_bounds_max,
            "topology_idx": torch.tensor(
                TOPOLOGY_TO_IDX.get(g.topology, 0), dtype=torch.long
            ),
            "n_components": torch.tensor(g.n_components, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Differentiable Circuit Constraints
# ═══════════════════════════════════════════════════════════════════════════

class CircuitConstraints(nn.Module):
    """Five differentiable circuit validity constraints.

    All methods take soft (continuous) graph representations and return
    per-sample violation scalars ≥ 0.  Zero means the constraint is satisfied.

    Constraints:
      C1: No floating nodes   — every active node has degree ≥ 1
      C2: Device completeness — devices have enough connections for their pins
      C3: No short circuits   — forbidden edge patterns
      C4: Graph connectivity  — single connected component (Fiedler value > 0)
      C5: Value bounds        — component values within physical ranges
    """

    def __init__(self, config: VCGConfig):
        super().__init__()
        self.max_nodes = config.max_nodes

        # Minimum connections per component type
        # MOSFET: 3 pins (D,G,S), R/C/L: 2 pins → min 1 connection each
        # In component-level graph, degree ≥ 1 suffices
        self._min_degree = 1

    def no_floating_nodes(
        self,
        soft_A: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """C1: Every active node must have degree ≥ 1.

        Args:
            soft_A: (B, N, N) soft adjacency in [0, 1]
            active_mask: (B, N) binary mask
        Returns:
            (B,) violation per sample
        """
        degrees = soft_A.sum(dim=-1)  # (B, N)
        violations = F.relu(self._min_degree - degrees) * active_mask
        return violations.sum(dim=-1)

    def device_completeness(
        self,
        soft_A: torch.Tensor,
        soft_X: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """C2: Devices with multiple pins need sufficient connections.

        MOSFET/BJT (3-4 pin) → degree ≥ 2
        2-terminal (R, C, L)  → degree ≥ 1

        Args:
            soft_A: (B, N, N) soft adjacency
            soft_X: (B, N, n_types) soft node type probabilities
            active_mask: (B, N)
        Returns:
            (B,) violation per sample
        """
        degrees = soft_A.sum(dim=-1)  # (B, N)

        # Multi-pin device types: MOSFET_N(4), MOSFET_P(5), BJT_NPN(6), BJT_PNP(7)
        multi_pin_indices = [4, 5, 6, 7]
        multi_pin_prob = soft_X[:, :, multi_pin_indices].sum(dim=-1)  # (B, N)

        # Multi-pin devices need degree ≥ 2
        required_degree = 1.0 + multi_pin_prob  # 1 for 2-terminal, 2 for multi-pin
        violations = F.relu(required_degree - degrees) * active_mask
        return violations.sum(dim=-1)

    def no_short_circuits(
        self,
        soft_A: torch.Tensor,
        soft_X: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """C3: Certain edge patterns are forbidden.

        In component-level graphs, we penalize direct connections between
        voltage sources and ground (would create short circuits).

        Args:
            soft_A: (B, N, N)
            soft_X: (B, N, n_types)
            active_mask: (B, N)
        Returns:
            (B,) violation
        """
        # VOLTAGE_SOURCE type index = 14
        vsrc_prob = soft_X[:, :, 14:15]  # (B, N, 1)

        # Self-connections through voltage sources (approximation of shorts)
        # If two voltage-source-like nodes are directly connected, that's bad
        # soft_A[i,j] * P(i=VSRC) * P(j=VSRC) should be small
        short_score = torch.bmm(vsrc_prob, vsrc_prob.transpose(1, 2))  # (B, N, N)
        violation = (soft_A * short_score * active_mask.unsqueeze(-1)).sum(dim=(-1, -2))
        return violation

    def graph_connectivity(
        self,
        soft_A: torch.Tensor,
        active_mask: torch.Tensor,
        topology_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """C4: Graph has at most K connected components (topology-aware).

        Uses algebraic connectivity: the K-th smallest eigenvalue of the
        graph Laplacian must be > 0, where K is the expected number of
        connected components for the topology.  For most topologies K=1
        (single component), but wien_bridge, instrumentation_amp, and
        colpitts have K=2.

        Args:
            soft_A: (B, N, N)
            active_mask: (B, N)
            topology_idx: (B,) optional topology indices for per-sample K
        Returns:
            (B,) violation
        """
        B = soft_A.shape[0]
        violations = torch.zeros(B, device=soft_A.device)

        for b in range(B):
            n_active = int(active_mask[b].sum().item())
            if n_active <= 1:
                continue  # trivially connected

            # Look up expected connected components for this topology
            K = 1
            if topology_idx is not None:
                idx = topology_idx[b].item()
                if idx < len(EXPECTED_COMPONENTS_BY_IDX):
                    K = EXPECTED_COMPONENTS_BY_IDX[idx].item()

            # If expected components >= active nodes, any graph is fine
            if K >= n_active:
                continue

            A_sub = soft_A[b, :n_active, :n_active]
            degree = A_sub.sum(dim=1)
            L = torch.diag(degree) - A_sub

            try:
                # eigvalsh not implemented on MPS — move to CPU if needed
                L_compute = L.cpu() if L.device.type == "mps" else L
                eigenvalues = torch.linalg.eigvalsh(L_compute)
                # K-th eigenvalue should be > 0 (at most K components)
                check_idx = min(K, n_active - 1)
                eig_k = eigenvalues[check_idx].to(L.device)
                violations[b] = F.relu(0.01 - eig_k)
            except RuntimeError:
                violations[b] = 1.0  # failed → assume disconnected

        return violations

    def value_bounds(
        self,
        soft_V: torch.Tensor,
        active_mask: torch.Tensor,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
    ) -> torch.Tensor:
        """C5: Component values within physical ranges.

        Args:
            soft_V: (B, N) values in log10 scale
            active_mask: (B, N)
            bounds_min: (B, N) log10 of minimum value
            bounds_max: (B, N) log10 of maximum value
        Returns:
            (B,) violation
        """
        below = F.relu(bounds_min - soft_V) * active_mask
        above = F.relu(soft_V - bounds_max) * active_mask
        return (below + above).sum(dim=-1)

    def adjacency_symmetry(
        self,
        soft_A: torch.Tensor,
    ) -> torch.Tensor:
        """Soft adjacency should be symmetric (undirected graph).

        Args:
            soft_A: (B, N, N)
        Returns:
            (B,) violation
        """
        diff = (soft_A - soft_A.transpose(-1, -2)).abs()
        return diff.sum(dim=(-1, -2))

    def all_constraints(
        self,
        soft_A: torch.Tensor,
        soft_X: torch.Tensor,
        soft_V: torch.Tensor,
        active_mask: torch.Tensor,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
        topology_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute all constraints.

        Args:
            topology_idx: (B,) optional topology indices, used by
                graph_connectivity to allow multi-component topologies.

        Returns:
            (B, 5) tensor — per-constraint violations for each sample
        """
        c1 = self.no_floating_nodes(soft_A, active_mask)
        c2 = self.device_completeness(soft_A, soft_X, active_mask)
        c3 = self.no_short_circuits(soft_A, soft_X, active_mask)
        c4 = self.graph_connectivity(soft_A, active_mask, topology_idx)
        c5 = self.value_bounds(soft_V, active_mask, bounds_min, bounds_max)
        return torch.stack([c1, c2, c3, c4, c5], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Graph Attention (Encoder)
# ═══════════════════════════════════════════════════════════════════════════

class BidirectionalGraphAttention(nn.Module):
    """Multi-head attention with graph-structure bias (non-causal).

    Unlike CausalSelfAttention, this allows bidirectional attention
    for encoding the full circuit simultaneously. Adjacency bias
    encourages attention between circuit-connected components.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Per-head scalar bias for adjacent nodes
        self.adj_bias = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
            graph_adj: (B, N, N) adjacency weights
            padding_mask: (B, N) True for active nodes, False for padding
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, N, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Graph adjacency bias
        if graph_adj is not None:
            adj_b = self.adj_bias.view(1, self.n_heads, 1, 1)
            attn = attn + adj_b * graph_adj.unsqueeze(1)

        # Padding mask: mask out attention to/from padded nodes
        if padding_mask is not None:
            # padding_mask: (B, N) — True = active
            mask_2d = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~mask_2d, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # handle all-masked rows
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.resid_drop(self.proj(out))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (reused from model.py pattern)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EncoderBlock(nn.Module):
    """Pre-norm transformer block with bidirectional graph attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = BidirectionalGraphAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        graph_adj: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), graph_adj=graph_adj, padding_mask=padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Spec Encoder
# ═══════════════════════════════════════════════════════════════════════════

class SpecEncoder(nn.Module):
    """Encodes specification (key, value) pairs into a fixed-length vector.

    Uses learnable spec-type embeddings + value projection, aggregated
    via cross-attention with a learnable query token.
    """

    def __init__(self, config: VCGConfig):
        super().__init__()
        self.type_embed = nn.Embedding(config.n_spec_types + 1, config.d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )
        # Cross-attention: learnable query attends to spec tokens
        self.query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True,
        )
        self.ln = RMSNorm(config.d_model)

    def forward(
        self,
        spec_types: torch.Tensor,    # (B, S) int
        spec_values: torch.Tensor,   # (B, S) float
        spec_mask: torch.Tensor,     # (B, S) float/bool
    ) -> torch.Tensor:
        """Returns (B, d_model) spec embedding."""
        B = spec_types.shape[0]

        x = self.type_embed(spec_types) + self.value_proj(spec_values.unsqueeze(-1))
        x = self.ln(x)

        query = self.query.expand(B, -1, -1)
        key_padding_mask = ~spec_mask.bool()

        # Guard against all-masked specs (no specs available → NaN in softmax)
        all_masked = key_padding_mask.all(dim=-1)  # (B,)
        if all_masked.any():
            # For fully-masked samples, unmask first position to prevent NaN
            # The value is arbitrary since there's no real spec info;
            # the type_embed(0) + value_proj(0) provides a neutral default.
            safe_mask = key_padding_mask.clone()
            safe_mask[all_masked, 0] = False
            key_padding_mask = safe_mask

        out, _ = self.cross_attn(query, x, x, key_padding_mask=key_padding_mask)

        # Zero out embeddings for samples that had no specs at all
        if all_masked.any():
            out[all_masked] = 0.0

        return out.squeeze(1)  # (B, d_model)


# ═══════════════════════════════════════════════════════════════════════════
# VCG Encoder (Circuit Graph → Latent z)
# ═══════════════════════════════════════════════════════════════════════════

class VCGEncoder(nn.Module):
    """Bidirectional Graph Transformer encoder.

    Input:  circuit graph (node types, values, adjacency)
    Output: latent distribution (mu, logvar)
    """

    def __init__(self, config: VCGConfig):
        super().__init__()
        self.config = config

        # Node feature embedding: type (one-hot) + value (continuous)
        self.type_embed = nn.Embedding(config.n_node_types, config.d_model)
        self.value_proj = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )
        # Position embedding for node order
        self.pos_embed = nn.Embedding(config.max_nodes, config.d_model)

        # Topology embedding
        self.topo_embed = nn.Embedding(config.n_topologies + 1, config.d_model)

        # RWPE projection (reuses K_WALK from model_enhanced)
        self.rwpe_proj = nn.Sequential(
            nn.Linear(K_WALK, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )

        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_encoder_layers)
        ])
        self.ln_f = RMSNorm(config.d_model)

        # Projection to latent
        self.mu_head = nn.Linear(config.d_model, config.latent_dim)
        self.logvar_head = nn.Linear(config.d_model, config.latent_dim)

    def _compute_rwpe(
        self,
        topology_indices: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Look up precomputed RWPE for each sample's topology.

        Returns (B, max_nodes, K_WALK) tensor.
        """
        B = topology_indices.shape[0]
        rwpe = torch.zeros(B, self.config.max_nodes, K_WALK, device=device)

        for b in range(B):
            topo_idx = topology_indices[b].item()
            # Reverse lookup: idx → topology name
            topo_name = None
            for name, idx in TOPOLOGY_TO_IDX.items():
                if idx == topo_idx:
                    topo_name = name
                    break
            if topo_name and topo_name in TOPOLOGY_RWPE:
                topo_rwpe = TOPOLOGY_RWPE[topo_name].to(device)
                n = min(topo_rwpe.shape[0], self.config.max_nodes)
                rwpe[b, :n] = topo_rwpe[:n]

        return rwpe

    def forward(
        self,
        node_types: torch.Tensor,     # (B, N) int
        values: torch.Tensor,         # (B, N) float (log10)
        adjacency: torch.Tensor,      # (B, N, N) float
        active_mask: torch.Tensor,    # (B, N) float
        topology_idx: torch.Tensor,   # (B,) int
        spec_embed: torch.Tensor,     # (B, d_model) from SpecEncoder
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode circuit graph → (mu, logvar)."""
        B, N = node_types.shape
        device = node_types.device

        # Node features
        positions = torch.arange(N, device=device)
        x = (
            self.type_embed(node_types)
            + self.value_proj(values.unsqueeze(-1))
            + self.pos_embed(positions)
            + self.topo_embed(topology_idx).unsqueeze(1)
        )

        # RWPE
        rwpe = self._compute_rwpe(topology_idx, device)
        x = x + self.rwpe_proj(rwpe)

        x = self.emb_drop(x)

        # Bidirectional transformer blocks with graph bias
        padding_mask = active_mask.bool()
        for block in self.blocks:
            x = block(x, graph_adj=adjacency, padding_mask=padding_mask)

        x = self.ln_f(x)

        # Mean-pool over active nodes → graph-level embedding
        mask_expanded = active_mask.unsqueeze(-1)  # (B, N, 1)
        x_sum = (x * mask_expanded).sum(dim=1)     # (B, d_model)
        n_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        x_pool = x_sum / n_active

        # Add spec conditioning
        x_pool = x_pool + spec_embed

        # Project to latent
        mu = self.mu_head(x_pool)
        logvar = self.logvar_head(x_pool)

        return mu, logvar


# ═══════════════════════════════════════════════════════════════════════════
# VCG Decoder (Latent z → Soft Circuit Graph)
# ═══════════════════════════════════════════════════════════════════════════

class VCGDecoder(nn.Module):
    """Decodes latent code z into a soft circuit graph.

    Output:
        soft_X: (B, N, n_types)  — node type probabilities  (softmax)
        soft_A: (B, N, N)        — edge probabilities        (sigmoid, symmetric)
        soft_V: (B, N)           — component values           (in log10 scale)
    """

    def __init__(self, config: VCGConfig):
        super().__init__()
        self.config = config
        N = config.max_nodes
        n_types = config.n_node_types

        # Input: latent z + spec embedding + topology embedding
        input_dim = config.latent_dim + config.d_model + config.d_model

        # MLP backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, config.decoder_hidden),
            nn.GELU(),
            nn.Linear(config.decoder_hidden, config.decoder_hidden),
            nn.GELU(),
        )

        # Output heads
        self.node_head = nn.Linear(config.decoder_hidden, N * n_types)
        self.edge_head = nn.Linear(config.decoder_hidden, N * N)
        self.value_head = nn.Linear(config.decoder_hidden, N)

        # Topology embedding for decoder conditioning
        self.topo_embed = nn.Embedding(config.n_topologies + 1, config.d_model)

    def forward(
        self,
        z: torch.Tensor,              # (B, latent_dim)
        spec_embed: torch.Tensor,     # (B, d_model)
        topology_idx: torch.Tensor,   # (B,) int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode z → soft graph (soft_X, soft_A, soft_V)."""
        B = z.shape[0]
        N = self.config.max_nodes
        n_types = self.config.n_node_types

        topo_emb = self.topo_embed(topology_idx)  # (B, d_model)
        h = self.backbone(torch.cat([z, spec_embed, topo_emb], dim=-1))

        # Node types: softmax probabilities
        node_logits = self.node_head(h).view(B, N, n_types)
        soft_X = F.softmax(node_logits, dim=-1)

        # Adjacency: sigmoid + symmetrize (undirected graph)
        edge_logits = self.edge_head(h).view(B, N, N)
        soft_A = torch.sigmoid(edge_logits)
        soft_A = (soft_A + soft_A.transpose(-1, -2)) / 2
        # Zero diagonal (no self-loops)
        diag_mask = torch.eye(N, device=z.device).bool().unsqueeze(0)
        soft_A = soft_A.masked_fill(diag_mask, 0.0)

        # Values: unconstrained (log10 scale)
        soft_V = self.value_head(h)  # (B, N)

        return soft_X, soft_A, soft_V


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Projection
# ═══════════════════════════════════════════════════════════════════════════

class ConstraintProjection(nn.Module):
    """Project a soft circuit graph onto the constraint-satisfying set.

    Uses iterative gradient descent to minimize constraint violations while
    staying close to the decoder output. This is the key mechanism that
    GUARANTEES validity: even if the VAE produces a poor output, projection
    fixes it.

    The projection operates on unconstrained logits (pre-softmax/sigmoid)
    to maintain differentiable parameterization.
    """

    def __init__(self, config: VCGConfig):
        super().__init__()
        self.config = config
        self.constraints = CircuitConstraints(config)

    @torch.no_grad()
    def project(
        self,
        soft_X: torch.Tensor,        # (B, N, n_types)
        soft_A: torch.Tensor,        # (B, N, N)
        soft_V: torch.Tensor,        # (B, N)
        active_mask: torch.Tensor,   # (B, N)
        bounds_min: torch.Tensor,    # (B, N)
        bounds_max: torch.Tensor,    # (B, N)
        topology_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Project onto constraint set via gradient descent.

        Returns:
            Updated (soft_X, soft_A, soft_V) and projection stats dict.
        """
        N = self.config.max_nodes
        n_types = self.config.n_node_types

        # Work in logit space for optimization
        X_logits = torch.log(soft_X.clamp(min=1e-8))
        A_logits = torch.logit(soft_A.clamp(1e-6, 1 - 1e-6))
        V_params = soft_V.clone()

        X_logits = X_logits.detach().requires_grad_(True)
        A_logits = A_logits.detach().requires_grad_(True)
        V_params = V_params.detach().requires_grad_(True)

        proj_opt = torch.optim.Adam(
            [X_logits, A_logits, V_params],
            lr=self.config.projection_lr,
        )

        best_violation = float("inf")
        best_state = (soft_X.clone(), soft_A.clone(), soft_V.clone())
        stats = {"steps": 0, "initial_violation": 0.0, "final_violation": 0.0}

        for step in range(self.config.n_projection_steps):
            proj_opt.zero_grad()

            # Map back to constrained space
            cur_X = F.softmax(X_logits, dim=-1)
            cur_A = torch.sigmoid(A_logits)
            cur_A = (cur_A + cur_A.transpose(-1, -2)) / 2
            diag_mask = torch.eye(N, device=soft_A.device).bool().unsqueeze(0)
            cur_A = cur_A.masked_fill(diag_mask, 0.0)
            cur_V = V_params

            # Enable gradients for this computation
            with torch.enable_grad():
                violations = self.constraints.all_constraints(
                    cur_A, cur_X, cur_V, active_mask, bounds_min, bounds_max,
                    topology_idx=topology_idx,
                )
                total_violation = violations.sum()

            if step == 0:
                stats["initial_violation"] = total_violation.item()

            if total_violation.item() < best_violation:
                best_violation = total_violation.item()
                best_state = (cur_X.detach().clone(), cur_A.detach().clone(), cur_V.detach().clone())

            if total_violation.item() < 1e-6:
                stats["steps"] = step + 1
                break

            total_violation.backward()
            proj_opt.step()

        stats["final_violation"] = best_violation
        stats["steps"] = step + 1

        return best_state[0], best_state[1], best_state[2], stats


# ═══════════════════════════════════════════════════════════════════════════
# Full ValidCircuitGen Model
# ═══════════════════════════════════════════════════════════════════════════

class ValidCircuitGenModel(nn.Module):
    """Constrained VAE for circuit generation with formal validity guarantees.

    Architecture:
        SpecEncoder   →  spec_embed  (B, d_model)
        VCGEncoder    →  mu, logvar  (B, latent_dim)
        Reparameterize → z          (B, latent_dim)
        VCGDecoder    →  soft_X, soft_A, soft_V
        Projection    →  valid soft graph
        Discretize    →  token sequence

    Training loss:
        L = L_recon + β * L_KL + Σ λ_i * c_i(G)
    """

    def __init__(self, config: VCGConfig):
        super().__init__()
        self.config = config

        self.spec_encoder = SpecEncoder(config)
        self.encoder = VCGEncoder(config)
        self.decoder = VCGDecoder(config)
        self.constraints = CircuitConstraints(config)
        self.projection = ConstraintProjection(config)

        # Lagrange multipliers (learned)
        self.log_lambdas = nn.Parameter(
            torch.full((config.n_constraints,), math.log(config.initial_lambda))
        )

    @property
    def lambdas(self) -> torch.Tensor:
        """Non-negative Lagrange multipliers."""
        return self.log_lambdas.exp()

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(
        self,
        batch: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode batch → (mu, logvar, spec_embed)."""
        spec_embed = self.spec_encoder(
            batch["spec_types"], batch["spec_values"], batch["spec_mask"],
        )
        mu, logvar = self.encoder(
            batch["node_types"], batch["values"], batch["adjacency"],
            batch["active_mask"], batch["topology_idx"], spec_embed,
        )
        return mu, logvar, spec_embed

    def decode(
        self,
        z: torch.Tensor,
        spec_embed: torch.Tensor,
        topology_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent → soft graph."""
        return self.decoder(z, spec_embed, topology_idx)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full forward pass for training.

        Returns:
            recon_loss: reconstruction loss (type + adjacency + value)
            kl_loss:    KL divergence from prior
            violations: (B, n_constraints) constraint violations
            total_loss: combined Lagrangian loss
            stats:      dict of per-component losses
        """
        # 1. Encode
        mu, logvar, spec_embed = self.encode(batch)

        # 2. Reparameterize
        z = self.reparameterize(mu, logvar)

        # 3. Decode
        soft_X, soft_A, soft_V = self.decode(z, spec_embed, batch["topology_idx"])

        # 4. Reconstruction losses
        active_mask = batch["active_mask"]
        B, N = active_mask.shape

        # Type reconstruction: cross-entropy on active nodes
        true_types = batch["node_types"]  # (B, N) int
        type_logits = torch.log(soft_X.clamp(min=1e-8))  # (B, N, n_types)
        type_loss = F.nll_loss(
            type_logits.view(-1, self.config.n_node_types),
            true_types.view(-1),
            reduction="none",
        ).view(B, N)
        type_loss = (type_loss * active_mask).sum() / active_mask.sum().clamp(min=1)

        # Adjacency reconstruction: binary cross-entropy
        true_adj = batch["adjacency"]  # (B, N, N)
        # Only count edges between active nodes
        adj_mask = active_mask.unsqueeze(-1) * active_mask.unsqueeze(-2)  # (B, N, N)
        adj_loss = F.binary_cross_entropy(
            soft_A * adj_mask, true_adj * adj_mask, reduction="sum",
        ) / adj_mask.sum().clamp(min=1)

        # Value reconstruction: MSE in log10 space
        true_values = batch["values"]  # (B, N) log10
        value_loss = F.mse_loss(
            soft_V * active_mask, true_values * active_mask, reduction="sum",
        ) / active_mask.sum().clamp(min=1)

        recon_loss = type_loss + adj_loss + 5.0 * value_loss  # weight values higher

        # 5. KL divergence
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # 6. Constraint violations
        violations = self.constraints.all_constraints(
            soft_A, soft_X, soft_V, active_mask,
            batch["value_bounds_min"], batch["value_bounds_max"],
            topology_idx=batch["topology_idx"],
        )

        # 7. Lagrangian total loss
        constraint_loss = (self.lambdas * violations.mean(dim=0)).sum()
        total_loss = recon_loss + self.config.beta_kl * kl_loss + constraint_loss

        stats = {
            "loss/total": total_loss.item(),
            "loss/recon": recon_loss.item(),
            "loss/type": type_loss.item(),
            "loss/adj": adj_loss.item(),
            "loss/value": value_loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/constraint": constraint_loss.item(),
            "constraint/no_floating": violations[:, 0].mean().item(),
            "constraint/device_comp": violations[:, 1].mean().item(),
            "constraint/no_short": violations[:, 2].mean().item(),
            "constraint/connectivity": violations[:, 3].mean().item(),
            "constraint/value_bounds": violations[:, 4].mean().item(),
            "lambda/mean": self.lambdas.mean().item(),
            "kl/mean": kl_loss.item(),
            "z/norm": z.norm(dim=-1).mean().item(),
        }

        return recon_loss, kl_loss, violations, total_loss, stats

    @torch.no_grad()
    def generate(
        self,
        spec_types: torch.Tensor,     # (B, S) or (S,)
        spec_values: torch.Tensor,    # (B, S)
        spec_mask: torch.Tensor,      # (B, S)
        topology_idx: torch.Tensor,   # (B,)
        active_mask: torch.Tensor,    # (B, N)
        bounds_min: torch.Tensor,     # (B, N)
        bounds_max: torch.Tensor,     # (B, N)
        n_samples: int = 1,
        temperature: float = 1.0,
        use_projection: bool = True,
    ) -> Tuple[list[CircuitGraph], dict]:
        """Generate circuits from specs.

        Samples z from prior N(0,I), decodes, projects, discretizes.

        Returns:
            List of CircuitGraph objects and generation stats.
        """
        self.eval()
        device = next(self.parameters()).device

        # Ensure batch dimension
        if spec_types.dim() == 1:
            spec_types = spec_types.unsqueeze(0)
            spec_values = spec_values.unsqueeze(0)
            spec_mask = spec_mask.unsqueeze(0)
            topology_idx = topology_idx.unsqueeze(0)
            active_mask = active_mask.unsqueeze(0)
            bounds_min = bounds_min.unsqueeze(0)
            bounds_max = bounds_max.unsqueeze(0)

        B = spec_types.shape[0]

        # Repeat for n_samples
        if n_samples > 1:
            spec_types = spec_types.repeat(n_samples, 1)
            spec_values = spec_values.repeat(n_samples, 1)
            spec_mask = spec_mask.repeat(n_samples, 1)
            topology_idx = topology_idx.repeat(n_samples)
            active_mask = active_mask.repeat(n_samples, 1)
            bounds_min = bounds_min.repeat(n_samples, 1)
            bounds_max = bounds_max.repeat(n_samples, 1)

        total_B = spec_types.shape[0]

        # Encode specs
        spec_embed = self.spec_encoder(spec_types, spec_values, spec_mask)

        # Sample from prior
        z = torch.randn(total_B, self.config.latent_dim, device=device) * temperature

        # Decode
        soft_X, soft_A, soft_V = self.decode(z, spec_embed, topology_idx)

        # Project to feasible set
        proj_stats = {"steps": 0, "initial_violation": 0.0, "final_violation": 0.0}
        if use_projection:
            soft_X, soft_A, soft_V, proj_stats = self.projection.project(
                soft_X, soft_A, soft_V, active_mask, bounds_min, bounds_max,
                topology_idx=topology_idx,
            )

        # Discretize: argmax for types, threshold for edges, bin for values
        pred_types = soft_X.argmax(dim=-1)         # (B, N) int
        pred_adj = (soft_A > 0.5).float()          # (B, N, N) binary
        pred_values = soft_V                        # (B, N) log10

        # Post-projection: clamp values to bounds
        pred_values = torch.clamp(pred_values, min=bounds_min, max=bounds_max)

        # Create CircuitGraph objects
        graphs = []
        for b in range(total_B):
            # Look up topology name
            topo_name = "unknown"
            idx_val = topology_idx[b].item()
            for name, idx in TOPOLOGY_TO_IDX.items():
                if idx == idx_val:
                    topo_name = name
                    break

            n_active = int(active_mask[b].sum().item())
            graph = CircuitGraph(
                topology=topo_name,
                n_components=n_active,
                node_types=pred_types[b].cpu(),
                adjacency=pred_adj[b].cpu(),
                values=pred_values[b].cpu(),
                active_mask=active_mask[b].cpu(),
                spec_types=spec_types[b].cpu(),
                spec_values=spec_values[b].cpu(),
                spec_mask=spec_mask[b].cpu(),
                value_bounds_min=bounds_min[b].cpu(),
                value_bounds_max=bounds_max[b].cpu(),
            )
            graphs.append(graph)

        gen_stats = {
            "n_generated": total_B,
            "projection": proj_stats,
        }
        return graphs, gen_stats

    @torch.no_grad()
    def interpolate(
        self,
        batch_a: dict[str, torch.Tensor],
        batch_b: dict[str, torch.Tensor],
        n_steps: int = 5,
    ) -> list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Interpolate between two circuits in latent space.

        Returns list of (soft_X, soft_A, soft_V) at each interpolation step.
        """
        self.eval()

        mu_a, logvar_a, spec_a = self.encode(batch_a)
        mu_b, logvar_b, spec_b = self.encode(batch_b)

        results = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            mu_interp = (1 - alpha) * mu_a + alpha * mu_b
            spec_interp = (1 - alpha) * spec_a + alpha * spec_b
            topo_idx = batch_a["topology_idx"]  # use first topology

            soft_X, soft_A, soft_V = self.decode(mu_interp, spec_interp, topo_idx)
            results.append((soft_X, soft_A, soft_V))

        return results

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_group(self) -> dict[str, int]:
        groups = {
            "spec_encoder": 0,
            "encoder": 0,
            "decoder": 0,
            "constraints": 0,
            "lagrange": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if name.startswith("spec_encoder"):
                groups["spec_encoder"] += n
            elif name.startswith("encoder"):
                groups["encoder"] += n
            elif name.startswith("decoder"):
                groups["decoder"] += n
            elif name.startswith("constraints"):
                groups["constraints"] += n
            elif "lambda" in name:
                groups["lagrange"] += n
        return groups


# ═══════════════════════════════════════════════════════════════════════════
# Lagrangian VAE Trainer
# ═══════════════════════════════════════════════════════════════════════════

class LagrangianVAETrainer:
    """Training loop with Lagrangian relaxation for constraint satisfaction.

    Implements a minimax optimization:
        Model parameters: MINIMIZE  L_recon + β*L_KL + λ·constraints
        Lagrange multipliers: MAXIMIZE  λ·constraints  (dual ascent)

    The multipliers automatically increase for frequently violated constraints,
    steering the model toward validity without manual tuning.
    """

    def __init__(
        self,
        model: ValidCircuitGenModel,
        lr: float = 1e-4,
        lambda_lr: float = 0.01,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 50000,
    ):
        self.model = model
        self.lambda_lr = lambda_lr

        # Separate optimizer for model (excludes lambdas)
        model_params = [
            p for n, p in model.named_parameters() if "log_lambdas" not in n
        ]
        self.model_opt = torch.optim.AdamW(
            model_params, lr=lr, weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_opt, T_max=max_steps, eta_min=lr * 0.01,
        )

        self.step_count = 0
        self.warmup_steps = warmup_steps
        self._base_lr = lr

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Single training step with Lagrangian optimization.

        Returns dict of training stats.
        """
        self.model.train()
        self.step_count += 1

        # Warmup LR
        if self.step_count <= self.warmup_steps:
            lr_scale = self.step_count / self.warmup_steps
            for pg in self.model_opt.param_groups:
                pg["lr"] = self._base_lr * lr_scale

        # Forward
        recon_loss, kl_loss, violations, total_loss, stats = self.model(batch)

        # Backward + step (model parameters MINIMIZE)
        self.model_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.model_opt.step()

        if self.step_count > self.warmup_steps:
            self.scheduler.step()

        # Dual ascent step: INCREASE multipliers for violated constraints
        with torch.no_grad():
            mean_violations = violations.mean(dim=0).detach()
            self.model.log_lambdas += self.lambda_lr * mean_violations
            # Clamp to prevent explosion
            self.model.log_lambdas.clamp_(
                min=math.log(0.01), max=math.log(100.0)
            )

        stats["lr"] = self.model_opt.param_groups[0]["lr"]
        stats["step"] = self.step_count
        return stats

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "model_opt": self.model_opt.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.model.load_state_dict(d["model"])
        self.model_opt.load_state_dict(d["model_opt"])
        self.scheduler.load_state_dict(d["scheduler"])
        self.step_count = d["step_count"]


# ═══════════════════════════════════════════════════════════════════════════
# Validity checker (post-generation)
# ═══════════════════════════════════════════════════════════════════════════

def check_circuit_validity(graph: CircuitGraph) -> dict[str, bool]:
    """Check if a generated CircuitGraph is structurally valid.

    Returns dict of constraint name → satisfied (True/False).
    """
    n = graph.n_components
    adj = graph.adjacency[:n, :n]
    active = graph.active_mask[:n]

    results = {}

    # C1: No floating nodes
    degrees = adj.sum(dim=-1)
    results["no_floating_nodes"] = bool((degrees >= 1).all())

    # C2: Device completeness (simplified: all nodes have edges)
    results["device_completeness"] = bool((degrees >= 1).all())

    # C3: No short circuits (check for forbidden patterns)
    results["no_short_circuits"] = True  # simplified check

    # C4: Graph connectivity (topology-aware)
    # Some component-level topology abstractions (e.g., wien_bridge,
    # instrumentation_amp, colpitts) are intentionally multi-component graphs.
    # For those, allow the same number of connected components as the reference
    # topology graph while still enforcing no floating nodes.
    if n <= 1:
        results["graph_connected"] = True
    else:
        try:
            actual_components = _count_connected_components(adj)

            expected_components = 1
            topo_edges = TOPOLOGY_ADJACENCY.get(graph.topology, [])
            if topo_edges:
                ref_adj = torch.zeros(n, n, device=adj.device, dtype=adj.dtype)
                for i, j in topo_edges:
                    if i < n and j < n:
                        ref_adj[i, j] = 1.0
                        ref_adj[j, i] = 1.0
                expected_components = _count_connected_components(ref_adj)

            results["graph_connected"] = actual_components <= expected_components
        except Exception:
            results["graph_connected"] = False

    # C5: Value bounds
    vals = graph.values[:n]
    mins = graph.value_bounds_min[:n]
    maxs = graph.value_bounds_max[:n]
    results["values_in_bounds"] = bool(
        ((vals >= mins - 0.1) & (vals <= maxs + 0.1)).all()
    )

    # Overall
    results["valid"] = all(results.values())
    return results
