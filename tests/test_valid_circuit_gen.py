"""Tests for ValidCircuitGen (Direction 5): Constrained VAE for circuit generation.

Covers:
  - Configuration
  - CircuitGraph data structure and conversions
  - CircuitConstraints (all 5 differentiable constraints)
  - Encoder (bidirectional graph transformer)
  - Decoder (MLP → soft graph)
  - SpecEncoder (spec conditioning)
  - ConstraintProjection (iterative feasibility projection)
  - Full ValidCircuitGenModel (forward, generate, interpolate)
  - LagrangianVAETrainer (training step, dual ascent)
  - Dataset (CircuitGraphDataset)
  - Validity checker
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig,
    ValidCircuitGenModel,
    CircuitGraph,
    CircuitGraphDataset,
    CircuitConstraints,
    BidirectionalGraphAttention,
    EncoderBlock,
    SwiGLUFFN,
    VCGEncoder,
    VCGDecoder,
    SpecEncoder,
    ConstraintProjection,
    LagrangianVAETrainer,
    check_circuit_validity,
    circuit_sample_to_graph,
    graph_to_token_sequence,
    NODE_TYPE_TO_IDX,
    IDX_TO_NODE_TYPE,
    TOPOLOGY_TO_IDX,
    SPEC_TO_IDX,
    N_NODE_TYPES,
    LOG_VAL_MIN,
    LOG_VAL_MAX,
)
from arcs.datagen import CircuitSample
from arcs.simulate import COMPONENT_TO_PARAM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Small config for fast tests."""
    return VCGConfig(
        latent_dim=16,
        max_nodes=8,
        d_model=64,
        n_encoder_layers=2,
        n_heads=2,
        d_ff=128,
        decoder_hidden=128,
        n_decoder_layers=2,
        n_projection_steps=5,
        beta_kl=0.1,
    )


@pytest.fixture
def tokenizer():
    return CircuitTokenizer()


@pytest.fixture
def sample_batch(config):
    """Create a fake batch for testing."""
    B = 4
    N = config.max_nodes
    return {
        "node_types": torch.randint(0, N_NODE_TYPES, (B, N)),
        "adjacency": torch.zeros(B, N, N),
        "values": torch.randn(B, N) * 2 - 5,  # log10 values
        "active_mask": torch.ones(B, N),
        "spec_types": torch.randint(0, len(SPEC_TO_IDX), (B, config.max_specs)),
        "spec_values": torch.randn(B, config.max_specs),
        "spec_mask": torch.ones(B, config.max_specs),
        "value_bounds_min": torch.full((B, N), LOG_VAL_MIN),
        "value_bounds_max": torch.full((B, N), LOG_VAL_MAX),
        "topology_idx": torch.ones(B, dtype=torch.long),  # index 1
        "n_components": torch.full((B,), 4, dtype=torch.long),
    }


@pytest.fixture
def sample_batch_with_adj(sample_batch):
    """Batch with simple adjacency (chain graph)."""
    B, N = sample_batch["active_mask"].shape
    adj = torch.zeros(B, N, N)
    for b in range(B):
        for i in range(N - 1):
            adj[b, i, i + 1] = 1.0
            adj[b, i + 1, i] = 1.0
    sample_batch["adjacency"] = adj
    return sample_batch


# ═══════════════════════════════════════════════════════════════════════════
# Configuration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVCGConfig:
    def test_defaults(self):
        cfg = VCGConfig()
        assert cfg.latent_dim == 64
        assert cfg.max_nodes == 12
        assert cfg.n_node_types == N_NODE_TYPES
        assert cfg.d_model == 256
        assert cfg.n_constraints == 5

    def test_to_from_dict(self):
        cfg = VCGConfig(latent_dim=32, d_model=128)
        d = cfg.to_dict()
        cfg2 = VCGConfig.from_dict(d)
        assert cfg2.latent_dim == 32
        assert cfg2.d_model == 128

    def test_node_type_mapping(self):
        assert NODE_TYPE_TO_IDX["NONE"] == 0
        assert NODE_TYPE_TO_IDX["RESISTOR"] == 1
        assert NODE_TYPE_TO_IDX["MOSFET_N"] == 4
        assert IDX_TO_NODE_TYPE[0] == "NONE"
        assert len(NODE_TYPE_TO_IDX) == N_NODE_TYPES

    def test_topology_mapping(self):
        assert "buck" in TOPOLOGY_TO_IDX
        assert "boost" in TOPOLOGY_TO_IDX
        assert "unknown" in TOPOLOGY_TO_IDX
        assert TOPOLOGY_TO_IDX["unknown"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Graph tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitGraph:
    def test_circuit_sample_to_graph(self, tokenizer):
        """Test conversion from CircuitSample to CircuitGraph."""
        sample = CircuitSample(
            topology="buck",
            parameters={
                "inductance": 47e-6,
                "capacitance": 100e-6,
                "esr": 0.05,
                "r_dson": 0.02,
            },
            operating_conditions={"vin": 12.0, "vout": 5.0, "iout": 2.0, "fsw": 500e3},
            metrics={"efficiency": 0.92},
            valid=True,
            sim_time=0.1,
        )

        config = VCGConfig()
        graph = circuit_sample_to_graph(sample, tokenizer, config)

        assert graph.topology == "buck"
        assert graph.n_components == 4
        assert graph.node_types[0].item() == NODE_TYPE_TO_IDX["INDUCTOR"]
        assert graph.node_types[1].item() == NODE_TYPE_TO_IDX["CAPACITOR"]
        assert graph.node_types[2].item() == NODE_TYPE_TO_IDX["RESISTOR"]
        assert graph.node_types[3].item() == NODE_TYPE_TO_IDX["MOSFET_N"]

        # Active mask
        assert graph.active_mask[:4].sum() == 4.0
        assert graph.active_mask[4:].sum() == 0.0

        # Values (log10)
        assert abs(graph.values[0].item() - math.log10(47e-6)) < 0.01
        assert abs(graph.values[1].item() - math.log10(100e-6)) < 0.01

        # Adjacency (from TOPOLOGY_ADJACENCY["buck"])
        assert graph.adjacency[0, 3] == 1.0  # inductor-mosfet
        assert graph.adjacency[3, 0] == 1.0  # symmetric
        assert graph.adjacency[0, 1] == 1.0  # inductor-capacitor

        # Specs
        assert graph.spec_mask.sum() >= 3  # at least vin, vout, iout

    def test_graph_to_token_sequence(self, tokenizer):
        """Test conversion from CircuitGraph back to token sequence."""
        config = VCGConfig()

        sample = CircuitSample(
            topology="buck",
            parameters={
                "inductance": 47e-6,
                "capacitance": 100e-6,
                "esr": 0.05,
                "r_dson": 0.02,
            },
            operating_conditions={"vin": 12.0, "vout": 5.0, "iout": 2.0},
            metrics={},
            valid=True,
            sim_time=0.1,
        )

        graph = circuit_sample_to_graph(sample, tokenizer, config)
        tokens = graph_to_token_sequence(graph, tokenizer)

        # Check structure: START TOPO SEP specs SEP comps END
        assert tokens[0] == tokenizer.start_id
        assert tokens[-1] == tokenizer.end_id

        # Should contain TOPO_BUCK
        topo_id = tokenizer.name_to_id.get("TOPO_BUCK")
        assert topo_id in tokens

        # Should contain component tokens
        comp_inductor = tokenizer.name_to_id.get("COMP_INDUCTOR")
        assert comp_inductor in tokens

    def test_roundtrip_topology(self, tokenizer):
        """Verify sample→graph→tokens preserves topology info."""
        for topology in ["buck", "boost", "inverting_amp"]:
            if topology not in COMPONENT_TO_PARAM:
                continue

            comp_list = COMPONENT_TO_PARAM[topology]
            params = {pname: 1e-3 for _, pname in comp_list}

            sample = CircuitSample(
                topology=topology, parameters=params,
                operating_conditions={"vin": 5.0},
                metrics={}, valid=True, sim_time=0.01,
            )

            config = VCGConfig()
            graph = circuit_sample_to_graph(sample, tokenizer, config)
            assert graph.topology == topology
            assert graph.n_components == len(comp_list)

    def test_sallen_key_token_aliases(self, tokenizer):
        """Sallen-Key long topology names map to LP/HP/BP topology tokens."""
        for topology, expected_token in [
            ("sallen_key_lowpass", "TOPO_SALLEN_KEY_LP"),
            ("sallen_key_highpass", "TOPO_SALLEN_KEY_HP"),
            ("sallen_key_bandpass", "TOPO_SALLEN_KEY_BP"),
        ]:
            comp_list = COMPONENT_TO_PARAM[topology]
            params = {pname: 1e-3 for _, pname in comp_list}
            sample = CircuitSample(
                topology=topology,
                parameters=params,
                operating_conditions={"cutoff_freq": 1000.0},
                metrics={},
                valid=True,
                sim_time=0.01,
            )
            graph = circuit_sample_to_graph(sample, tokenizer, VCGConfig())
            tokens = graph_to_token_sequence(graph, tokenizer)
            assert tokenizer.name_to_id[expected_token] in tokens


# ═══════════════════════════════════════════════════════════════════════════
# Constraint tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitConstraints:
    def test_no_floating_nodes_satisfied(self, config):
        """Chain graph: all nodes connected → no violation."""
        constraints = CircuitConstraints(config)
        B, N = 2, config.max_nodes

        # Chain adjacency: 0-1-2-3-...
        soft_A = torch.zeros(B, N, N)
        for i in range(N - 1):
            soft_A[:, i, i + 1] = 1.0
            soft_A[:, i + 1, i] = 1.0

        active_mask = torch.ones(B, N)
        violations = constraints.no_floating_nodes(soft_A, active_mask)
        assert violations.shape == (B,)
        assert violations.sum().item() == 0.0

    def test_no_floating_nodes_violated(self, config):
        """Isolated nodes → violation."""
        constraints = CircuitConstraints(config)
        B, N = 2, config.max_nodes

        soft_A = torch.zeros(B, N, N)  # no edges
        active_mask = torch.ones(B, N)

        violations = constraints.no_floating_nodes(soft_A, active_mask)
        assert violations.sum().item() > 0

    def test_device_completeness(self, config):
        """Multi-pin devices need higher degree."""
        constraints = CircuitConstraints(config)
        B, N = 1, config.max_nodes

        # Node 0 is MOSFET_N (index 4) — needs degree >= 2
        soft_X = torch.zeros(B, N, N_NODE_TYPES)
        soft_X[0, 0, 4] = 1.0  # MOSFET_N
        soft_X[0, 1, 1] = 1.0  # RESISTOR

        # Only one edge from MOSFET → violation
        soft_A = torch.zeros(B, N, N)
        soft_A[0, 0, 1] = 1.0
        soft_A[0, 1, 0] = 1.0

        active_mask = torch.zeros(B, N)
        active_mask[0, :2] = 1.0

        violation = constraints.device_completeness(soft_A, soft_X, active_mask)
        assert violation.item() > 0  # MOSFET has degree 1, needs 2

    def test_graph_connectivity_connected(self, config):
        """Connected graph → no violation."""
        constraints = CircuitConstraints(config)
        B, N = 1, 4

        soft_A = torch.zeros(B, config.max_nodes, config.max_nodes)
        # Simple chain: 0-1-2-3
        for i in range(3):
            soft_A[0, i, i + 1] = 1.0
            soft_A[0, i + 1, i] = 1.0

        active_mask = torch.zeros(B, config.max_nodes)
        active_mask[0, :4] = 1.0

        violation = constraints.graph_connectivity(soft_A, active_mask)
        assert violation.item() < 0.05

    def test_graph_connectivity_disconnected(self, config):
        """Disconnected graph → violation."""
        constraints = CircuitConstraints(config)
        B = 1

        soft_A = torch.zeros(B, config.max_nodes, config.max_nodes)
        # Two isolated pairs: 0-1 and 2-3
        soft_A[0, 0, 1] = 1.0
        soft_A[0, 1, 0] = 1.0
        soft_A[0, 2, 3] = 1.0
        soft_A[0, 3, 2] = 1.0

        active_mask = torch.zeros(B, config.max_nodes)
        active_mask[0, :4] = 1.0

        violation = constraints.graph_connectivity(soft_A, active_mask)
        assert violation.item() > 0

    def test_value_bounds_satisfied(self, config):
        """Values within bounds → no violation."""
        constraints = CircuitConstraints(config)
        B, N = 2, config.max_nodes

        soft_V = torch.zeros(B, N)  # log10(1) = 0
        active_mask = torch.ones(B, N)
        bounds_min = torch.full((B, N), -5.0)
        bounds_max = torch.full((B, N), 5.0)

        violation = constraints.value_bounds(soft_V, active_mask, bounds_min, bounds_max)
        assert violation.sum().item() == 0.0

    def test_value_bounds_violated(self, config):
        """Values outside bounds → violation."""
        constraints = CircuitConstraints(config)
        B, N = 1, config.max_nodes

        soft_V = torch.full((B, N), 10.0)  # way over max
        active_mask = torch.ones(B, N)
        bounds_min = torch.full((B, N), -5.0)
        bounds_max = torch.full((B, N), 5.0)

        violation = constraints.value_bounds(soft_V, active_mask, bounds_min, bounds_max)
        assert violation.item() > 0

    def test_all_constraints_shape(self, config, sample_batch_with_adj):
        """all_constraints returns (B, 5) tensor."""
        constraints = CircuitConstraints(config)
        batch = sample_batch_with_adj

        # Need soft_X for device_completeness and no_short_circuits
        B, N = batch["active_mask"].shape
        soft_X = F.one_hot(batch["node_types"], N_NODE_TYPES).float()

        violations = constraints.all_constraints(
            batch["adjacency"], soft_X, batch["values"],
            batch["active_mask"], batch["value_bounds_min"], batch["value_bounds_max"],
        )
        assert violations.shape == (B, 5)
        assert (violations >= 0).all()

    def test_adjacency_symmetry(self, config):
        """Symmetric adjacency → no violation."""
        constraints = CircuitConstraints(config)
        B, N = 2, config.max_nodes

        soft_A = torch.randn(B, N, N).abs()
        soft_A = (soft_A + soft_A.transpose(-1, -2)) / 2  # make symmetric

        violation = constraints.adjacency_symmetry(soft_A)
        assert violation.sum().item() < 1e-5


# ═══════════════════════════════════════════════════════════════════════════
# Module tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBidirectionalGraphAttention:
    def test_output_shape(self, config):
        attn = BidirectionalGraphAttention(config.d_model, config.n_heads, config.dropout)
        B, N = 2, config.max_nodes
        x = torch.randn(B, N, config.d_model)
        out = attn(x)
        assert out.shape == (B, N, config.d_model)

    def test_with_graph_adj(self, config):
        attn = BidirectionalGraphAttention(config.d_model, config.n_heads, config.dropout)
        B, N = 2, config.max_nodes
        x = torch.randn(B, N, config.d_model)
        adj = torch.zeros(B, N, N)
        adj[:, 0, 1] = 1.0
        adj[:, 1, 0] = 1.0
        out = attn(x, graph_adj=adj)
        assert out.shape == (B, N, config.d_model)

    def test_with_padding_mask(self, config):
        attn = BidirectionalGraphAttention(config.d_model, config.n_heads, config.dropout)
        B, N = 2, config.max_nodes
        x = torch.randn(B, N, config.d_model)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :4] = True  # only first 4 active
        out = attn(x, padding_mask=mask)
        assert out.shape == (B, N, config.d_model)


class TestSwiGLUFFN:
    def test_output_shape(self, config):
        ffn = SwiGLUFFN(config.d_model, config.d_ff, config.dropout)
        x = torch.randn(2, 8, config.d_model)
        out = ffn(x)
        assert out.shape == (2, 8, config.d_model)


class TestEncoderBlock:
    def test_forward(self, config):
        block = EncoderBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
        B, N = 2, config.max_nodes
        x = torch.randn(B, N, config.d_model)
        out = block(x)
        assert out.shape == (B, N, config.d_model)


class TestSpecEncoder:
    def test_output_shape(self, config):
        enc = SpecEncoder(config)
        B, S = 2, config.max_specs
        spec_types = torch.randint(0, config.n_spec_types, (B, S))
        spec_values = torch.randn(B, S)
        spec_mask = torch.ones(B, S)
        out = enc(spec_types, spec_values, spec_mask)
        assert out.shape == (B, config.d_model)

    def test_with_partial_mask(self, config):
        enc = SpecEncoder(config)
        B, S = 2, config.max_specs
        spec_types = torch.randint(0, config.n_spec_types, (B, S))
        spec_values = torch.randn(B, S)
        spec_mask = torch.zeros(B, S)
        spec_mask[:, :3] = 1.0  # only 3 specs
        out = enc(spec_types, spec_values, spec_mask)
        assert out.shape == (B, config.d_model)

    def test_all_zero_spec_mask_no_nan(self, config):
        """All-zero spec_mask must NOT produce NaN (regression test)."""
        enc = SpecEncoder(config)
        B, S = 4, config.max_specs
        spec_types = torch.zeros(B, S, dtype=torch.long)
        spec_values = torch.zeros(B, S)
        spec_mask = torch.zeros(B, S)  # all masked → previously caused NaN
        out = enc(spec_types, spec_values, spec_mask)
        assert out.shape == (B, config.d_model)
        assert not torch.isnan(out).any(), "SpecEncoder produced NaN for all-zero spec_mask"
        assert (out == 0.0).all(), "All-masked samples should get zero embeddings"

    def test_mixed_masked_batch(self, config):
        """Batch where some samples have specs and some don't."""
        enc = SpecEncoder(config)
        B, S = 4, config.max_specs
        spec_types = torch.randint(0, config.n_spec_types, (B, S))
        spec_values = torch.randn(B, S)
        spec_mask = torch.zeros(B, S)
        spec_mask[0, :3] = 1.0   # sample 0 has 3 specs
        spec_mask[2, :1] = 1.0   # sample 2 has 1 spec
        # samples 1, 3 have NO specs → all-zero mask
        out = enc(spec_types, spec_values, spec_mask)
        assert out.shape == (B, config.d_model)
        assert not torch.isnan(out).any(), "Mixed batch produced NaN"
        # Samples with no specs should be zero
        assert (out[1] == 0.0).all()
        assert (out[3] == 0.0).all()
        # Samples with specs should be non-zero (almost certainly)
        assert out[0].abs().sum() > 0
        assert out[2].abs().sum() > 0


class TestVCGEncoder:
    def test_output_shape(self, config, sample_batch):
        encoder = VCGEncoder(config)
        spec_embed = torch.randn(4, config.d_model)
        mu, logvar = encoder(
            sample_batch["node_types"], sample_batch["values"],
            sample_batch["adjacency"], sample_batch["active_mask"],
            sample_batch["topology_idx"], spec_embed,
        )
        assert mu.shape == (4, config.latent_dim)
        assert logvar.shape == (4, config.latent_dim)


class TestVCGDecoder:
    def test_output_shape(self, config):
        decoder = VCGDecoder(config)
        B = 4
        z = torch.randn(B, config.latent_dim)
        spec_embed = torch.randn(B, config.d_model)
        topo_idx = torch.ones(B, dtype=torch.long)
        soft_X, soft_A, soft_V = decoder(z, spec_embed, topo_idx)

        assert soft_X.shape == (B, config.max_nodes, config.n_node_types)
        assert soft_A.shape == (B, config.max_nodes, config.max_nodes)
        assert soft_V.shape == (B, config.max_nodes)

    def test_soft_X_is_probability(self, config):
        decoder = VCGDecoder(config)
        z = torch.randn(2, config.latent_dim)
        spec_embed = torch.randn(2, config.d_model)
        topo_idx = torch.ones(2, dtype=torch.long)
        soft_X, _, _ = decoder(z, spec_embed, topo_idx)

        # Each row should sum to 1
        row_sums = soft_X.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_soft_A_is_symmetric(self, config):
        decoder = VCGDecoder(config)
        z = torch.randn(2, config.latent_dim)
        spec_embed = torch.randn(2, config.d_model)
        topo_idx = torch.ones(2, dtype=torch.long)
        _, soft_A, _ = decoder(z, spec_embed, topo_idx)

        # Should be symmetric
        diff = (soft_A - soft_A.transpose(-1, -2)).abs().max()
        assert diff.item() < 1e-5

    def test_soft_A_no_self_loops(self, config):
        decoder = VCGDecoder(config)
        z = torch.randn(2, config.latent_dim)
        spec_embed = torch.randn(2, config.d_model)
        topo_idx = torch.ones(2, dtype=torch.long)
        _, soft_A, _ = decoder(z, spec_embed, topo_idx)

        # Diagonal should be zero
        diag = torch.diagonal(soft_A, dim1=-2, dim2=-1)
        assert diag.sum().item() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Projection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintProjection:
    def test_projection_reduces_violations(self, config):
        """Projection should reduce constraint violations."""
        proj = ConstraintProjection(config)
        B, N = 2, config.max_nodes

        # Start with random (likely invalid) graph
        soft_X = F.softmax(torch.randn(B, N, N_NODE_TYPES), dim=-1)
        soft_A = torch.sigmoid(torch.randn(B, N, N))
        soft_A = (soft_A + soft_A.transpose(-1, -2)) / 2
        soft_V = torch.randn(B, N) * 5  # some out of bounds

        active_mask = torch.ones(B, N)
        bounds_min = torch.full((B, N), -8.0)
        bounds_max = torch.full((B, N), 3.0)

        new_X, new_A, new_V, stats = proj.project(
            soft_X, soft_A, soft_V, active_mask, bounds_min, bounds_max,
        )

        assert stats["final_violation"] <= stats["initial_violation"] + 1e-3
        assert stats["steps"] >= 1

    def test_projection_output_shapes(self, config):
        proj = ConstraintProjection(config)
        B, N = 2, config.max_nodes

        soft_X = F.softmax(torch.randn(B, N, N_NODE_TYPES), dim=-1)
        soft_A = torch.sigmoid(torch.randn(B, N, N))
        soft_V = torch.randn(B, N)
        active_mask = torch.ones(B, N)
        bounds_min = torch.full((B, N), LOG_VAL_MIN)
        bounds_max = torch.full((B, N), LOG_VAL_MAX)

        new_X, new_A, new_V, _ = proj.project(
            soft_X, soft_A, soft_V, active_mask, bounds_min, bounds_max,
        )

        assert new_X.shape == soft_X.shape
        assert new_A.shape == soft_A.shape
        assert new_V.shape == soft_V.shape


# ═══════════════════════════════════════════════════════════════════════════
# Full Model tests
# ═══════════════════════════════════════════════════════════════════════════

class TestValidCircuitGenModel:
    def test_creation(self, config):
        model = ValidCircuitGenModel(config)
        n_params = model.count_parameters()
        assert n_params > 0
        print(f"VCG params: {n_params:,}")
        print(f"  Groups: {model.count_parameters_by_group()}")

    def test_forward(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)
        recon, kl, violations, total, stats = model(sample_batch_with_adj)

        assert recon.shape == ()  # scalar
        assert kl.shape == ()
        assert violations.shape == (4, 5)
        assert total.shape == ()
        assert total.item() > 0  # non-trivial loss
        assert "loss/total" in stats
        assert "loss/recon" in stats
        assert "loss/kl" in stats
        assert "constraint/no_floating" in stats

    def test_forward_backward(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)
        _, _, _, total, _ = model(sample_batch_with_adj)
        total.backward()

        # Check gradients exist
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_generate(self, config):
        model = ValidCircuitGenModel(config)

        B, S, N = 1, config.max_specs, config.max_nodes
        graphs, stats = model.generate(
            spec_types=torch.randint(0, config.n_spec_types, (B, S)),
            spec_values=torch.randn(B, S),
            spec_mask=torch.ones(B, S),
            topology_idx=torch.ones(B, dtype=torch.long),
            active_mask=torch.ones(B, N),
            bounds_min=torch.full((B, N), LOG_VAL_MIN),
            bounds_max=torch.full((B, N), LOG_VAL_MAX),
            n_samples=3,
            use_projection=True,
        )

        assert len(graphs) == 3
        assert stats["n_generated"] == 3

        # Each graph should have valid structure
        for g in graphs:
            assert isinstance(g, CircuitGraph)
            assert g.node_types.shape == (config.max_nodes,)
            assert g.adjacency.shape == (config.max_nodes, config.max_nodes)
            assert g.values.shape == (config.max_nodes,)

    def test_generate_without_projection(self, config):
        model = ValidCircuitGenModel(config)

        B, S, N = 1, config.max_specs, config.max_nodes
        graphs, stats = model.generate(
            spec_types=torch.randint(0, config.n_spec_types, (B, S)),
            spec_values=torch.randn(B, S),
            spec_mask=torch.ones(B, S),
            topology_idx=torch.ones(B, dtype=torch.long),
            active_mask=torch.ones(B, N),
            bounds_min=torch.full((B, N), LOG_VAL_MIN),
            bounds_max=torch.full((B, N), LOG_VAL_MAX),
            n_samples=2,
            use_projection=False,
        )

        assert len(graphs) == 2

    def test_interpolate(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)

        # Split batch into two halves
        batch_a = {k: v[:2] for k, v in sample_batch_with_adj.items()}
        batch_b = {k: v[2:] for k, v in sample_batch_with_adj.items()}

        results = model.interpolate(batch_a, batch_b, n_steps=3)
        assert len(results) == 4  # 3 steps + endpoints
        for soft_X, soft_A, soft_V in results:
            assert soft_X.shape[1] == config.max_nodes
            assert soft_A.shape[1] == config.max_nodes

    def test_lambdas(self, config):
        model = ValidCircuitGenModel(config)
        assert model.lambdas.shape == (config.n_constraints,)
        assert (model.lambdas > 0).all()

    def test_reparameterize(self):
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)
        z = ValidCircuitGenModel.reparameterize(mu, logvar)
        assert z.shape == (4, 16)
        # With zero mean/var, samples should be close to standard normal
        assert z.abs().mean() < 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Trainer tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLagrangianVAETrainer:
    def test_train_step(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)
        trainer = LagrangianVAETrainer(model, lr=1e-4, lambda_lr=0.01)

        stats = trainer.train_step(sample_batch_with_adj)

        assert "loss/total" in stats
        assert "lr" in stats
        assert stats["step"] == 1

    def test_multiple_steps(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)
        trainer = LagrangianVAETrainer(model, lr=1e-4)

        losses = []
        for _ in range(5):
            stats = trainer.train_step(sample_batch_with_adj)
            losses.append(stats["loss/total"])

        assert trainer.step_count == 5
        # Loss should generally decrease (or at least not explode)
        assert losses[-1] < losses[0] * 10

    def test_lambda_update(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)
        initial_lambdas = model.lambdas.clone()

        trainer = LagrangianVAETrainer(model, lr=1e-4, lambda_lr=0.1)
        trainer.train_step(sample_batch_with_adj)

        # Lambdas should have changed (dual ascent)
        final_lambdas = model.lambdas
        assert not torch.allclose(initial_lambdas, final_lambdas)

    def test_state_dict_roundtrip(self, config, sample_batch_with_adj):
        model = ValidCircuitGenModel(config)
        trainer = LagrangianVAETrainer(model, lr=1e-4)

        # Take a step
        trainer.train_step(sample_batch_with_adj)
        state = trainer.state_dict()

        # Create new trainer and load state
        model2 = ValidCircuitGenModel(config)
        trainer2 = LagrangianVAETrainer(model2, lr=1e-4)
        trainer2.load_state_dict(state)

        assert trainer2.step_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Dataset tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCircuitGraphDataset:
    def test_load_from_jsonl(self, tokenizer, tmp_path):
        """Test loading JSONL data into CircuitGraphDataset."""
        config = VCGConfig()

        # Create sample JSONL
        samples = []
        for i in range(10):
            sample = {
                "topology": "buck",
                "parameters": {
                    "inductance": 47e-6 * (1 + i * 0.1),
                    "capacitance": 100e-6,
                    "esr": 0.05,
                    "r_dson": 0.02,
                },
                "operating_conditions": {"vin": 12.0, "vout": 5.0, "iout": 2.0, "fsw": 500e3},
                "metrics": {"efficiency": 0.92},
                "valid": True,
                "sim_time": 0.1,
            }
            samples.append(sample)

        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        dataset = CircuitGraphDataset(tmp_path, tokenizer, config, valid_only=True)
        assert len(dataset) == 10

        item = dataset[0]
        assert "node_types" in item
        assert "adjacency" in item
        assert "values" in item
        assert "active_mask" in item
        assert item["node_types"].shape == (config.max_nodes,)

    def test_topology_diversity(self, tokenizer, tmp_path):
        """Test with multiple topologies."""
        config = VCGConfig()

        samples = []
        for topo, comp_list in COMPONENT_TO_PARAM.items():
            params = {pname: 1e-3 for _, pname in comp_list}
            sample = {
                "topology": topo,
                "parameters": params,
                "operating_conditions": {"vin": 5.0},
                "metrics": {},
                "valid": True,
                "sim_time": 0.01,
            }
            samples.append(sample)

        jsonl_path = tmp_path / "multi_topo.jsonl"
        with open(jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        dataset = CircuitGraphDataset(tmp_path, tokenizer, config, valid_only=False)
        assert len(dataset) >= len(COMPONENT_TO_PARAM)


# ═══════════════════════════════════════════════════════════════════════════
# Validity checker tests
# ═══════════════════════════════════════════════════════════════════════════

class TestValidityChecker:
    def test_valid_circuit(self):
        """A well-formed circuit should pass all checks."""
        config = VCGConfig()
        graph = CircuitGraph(
            topology="buck",
            n_components=4,
            node_types=torch.tensor([3, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0]),  # L,C,R,MOSFET
            adjacency=torch.zeros(12, 12),
            values=torch.tensor([-4.3, -4.0, -1.3, -1.7, 0, 0, 0, 0, 0, 0, 0, 0]),
            active_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            spec_types=torch.zeros(8, dtype=torch.long),
            spec_values=torch.zeros(8),
            spec_mask=torch.zeros(8),
            value_bounds_min=torch.full((12,), LOG_VAL_MIN),
            value_bounds_max=torch.full((12,), LOG_VAL_MAX),
        )
        # Chain connectivity: 0-1-2-3
        graph.adjacency[0, 1] = 1.0
        graph.adjacency[1, 0] = 1.0
        graph.adjacency[1, 2] = 1.0
        graph.adjacency[2, 1] = 1.0
        graph.adjacency[2, 3] = 1.0
        graph.adjacency[3, 2] = 1.0

        result = check_circuit_validity(graph)
        assert result["no_floating_nodes"]
        assert result["graph_connected"]
        assert result["values_in_bounds"]

    def test_floating_node_detected(self):
        """An isolated node should trigger violation."""
        config = VCGConfig()
        graph = CircuitGraph(
            topology="buck",
            n_components=4,
            node_types=torch.tensor([3, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
            adjacency=torch.zeros(12, 12),  # NO edges
            values=torch.zeros(12),
            active_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            spec_types=torch.zeros(8, dtype=torch.long),
            spec_values=torch.zeros(8),
            spec_mask=torch.zeros(8),
            value_bounds_min=torch.full((12,), LOG_VAL_MIN),
            value_bounds_max=torch.full((12,), LOG_VAL_MAX),
        )

        result = check_circuit_validity(graph)
        assert not result["no_floating_nodes"]
        assert not result["valid"]

    def test_disconnected_template_topology_is_valid(self):
        """Topologies with multi-component reference graphs should not fail C4."""
        graph = CircuitGraph(
            topology="wien_bridge",
            n_components=4,
            node_types=torch.tensor([1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            adjacency=torch.zeros(12, 12),
            values=torch.tensor([-3.0, -6.0, -3.0, -3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
            active_mask=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
            spec_types=torch.zeros(8, dtype=torch.long),
            spec_values=torch.zeros(8),
            spec_mask=torch.zeros(8),
            value_bounds_min=torch.full((12,), LOG_VAL_MIN),
            value_bounds_max=torch.full((12,), LOG_VAL_MAX),
        )
        # Expected template structure for wien_bridge is two connected components:
        # (0-1) and (2-3)
        graph.adjacency[0, 1] = graph.adjacency[1, 0] = 1.0
        graph.adjacency[2, 3] = graph.adjacency[3, 2] = 1.0

        result = check_circuit_validity(graph)
        assert result["no_floating_nodes"]
        assert result["graph_connected"]
        assert result["valid"]


# ═══════════════════════════════════════════════════════════════════════════
# Integration test
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_end_to_end_pipeline(self, config, tokenizer, tmp_path):
        """Full pipeline: data → model → train → generate → validate."""
        # 1. Create sample data
        samples = []
        for _ in range(20):
            import random
            sample = {
                "topology": "buck",
                "parameters": {
                    "inductance": random.uniform(10e-6, 100e-6),
                    "capacitance": random.uniform(10e-6, 500e-6),
                    "esr": random.uniform(0.01, 0.1),
                    "r_dson": random.uniform(0.01, 0.05),
                },
                "operating_conditions": {"vin": 12.0, "vout": 5.0, "iout": 2.0},
                "metrics": {},
                "valid": True,
                "sim_time": 0.01,
            }
            samples.append(sample)

        jsonl_path = tmp_path / "train.jsonl"
        with open(jsonl_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # 2. Create dataset
        dataset = CircuitGraphDataset(tmp_path, tokenizer, config, valid_only=False)
        assert len(dataset) == 20

        # 3. Create model + trainer
        model = ValidCircuitGenModel(config)
        trainer = LagrangianVAETrainer(model, lr=1e-3, lambda_lr=0.01)

        # 4. Train for a few steps
        from torch.utils.data import DataLoader

        def collate(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

        loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)
        for batch in loader:
            stats = trainer.train_step(batch)
            assert stats["loss/total"] < 1e6  # not exploding
            break

        # 5. Generate
        item = dataset[0]
        graphs, gen_stats = model.generate(
            spec_types=item["spec_types"].unsqueeze(0),
            spec_values=item["spec_values"].unsqueeze(0),
            spec_mask=item["spec_mask"].unsqueeze(0),
            topology_idx=item["topology_idx"].unsqueeze(0),
            active_mask=item["active_mask"].unsqueeze(0),
            bounds_min=item["value_bounds_min"].unsqueeze(0),
            bounds_max=item["value_bounds_max"].unsqueeze(0),
            n_samples=2,
            use_projection=True,
        )
        assert len(graphs) == 2

        # 6. Check validity
        for g in graphs:
            validity = check_circuit_validity(g)
            # bounds should be satisfied after projection
            assert validity["values_in_bounds"]

        # 7. Convert to token sequence
        for g in graphs:
            tokens = graph_to_token_sequence(g, tokenizer)
            assert tokens[0] == tokenizer.start_id
            assert tokens[-1] == tokenizer.end_id
