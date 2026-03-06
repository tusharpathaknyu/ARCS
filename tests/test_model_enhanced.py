"""Tests for enhanced ARCS model architectures (TwoHead + GraphTransformer).

Validates:
  - Model instantiation and parameter counts
  - Forward pass with/without value_mask
  - Loss computation
  - Generation (autoregressive)
  - Graph feature computation (topology-aware adjacency)
  - Model factory (create_model / load_model)
  - Interface compatibility with baseline model
"""

import tempfile
from pathlib import Path

import pytest
import torch

from arcs.model import ARCSConfig, ARCSModel
from arcs.model_enhanced import (
    TOPOLOGY_ADJACENCY,
    GraphTransformerARCSModel,
    TwoHeadARCSModel,
    create_model,
    load_model,
)
from arcs.tokenizer import CircuitTokenizer, TokenType


@pytest.fixture
def tokenizer():
    return CircuitTokenizer()


@pytest.fixture
def config(tokenizer):
    cfg = ARCSConfig.small()
    cfg.vocab_size = tokenizer.vocab_size
    return cfg


@pytest.fixture
def buck_ids(tokenizer):
    """Token IDs for a buck converter sample."""
    return [
        tokenizer.name_to_id["START"],
        tokenizer.name_to_id["TOPO_BUCK"],
        tokenizer.sep_id,
        tokenizer.name_to_id["SPEC_VIN"], tokenizer.encode_value(12.0),
        tokenizer.name_to_id["SPEC_VOUT"], tokenizer.encode_value(5.0),
        tokenizer.sep_id,
        tokenizer.name_to_id["COMP_INDUCTOR"], tokenizer.encode_value(100e-6),
        tokenizer.name_to_id["COMP_CAPACITOR"], tokenizer.encode_value(47e-6),
        tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(0.01),
        tokenizer.name_to_id["COMP_MOSFET_N"], tokenizer.encode_value(0.05),
        tokenizer.name_to_id["END"],
    ]


class TestTopologyAdjacency:
    """Test that topology adjacency tables are correctly defined."""

    def test_all_topologies_present(self):
        expected = [
            "buck", "boost", "buck_boost", "cuk", "sepic", "flyback", "forward",
            "inverting_amp", "noninverting_amp", "instrumentation_amp",
            "differential_amp", "sallen_key_lowpass", "sallen_key_highpass",
            "sallen_key_bandpass", "wien_bridge", "colpitts",
        ]
        for topo in expected:
            assert topo in TOPOLOGY_ADJACENCY, f"Missing topology: {topo}"

    def test_adjacency_pairs_are_tuples(self):
        for topo, pairs in TOPOLOGY_ADJACENCY.items():
            assert isinstance(pairs, list), f"{topo}: expected list"
            for pair in pairs:
                assert isinstance(pair, tuple) and len(pair) == 2, (
                    f"{topo}: bad pair {pair}"
                )
                assert pair[0] != pair[1], f"{topo}: self-loop {pair}"

    def test_adjacency_indices_in_range(self):
        """Component indices should be within the topology's component count."""
        # Known component counts per topology
        comp_counts = {
            "buck": 4, "boost": 4, "buck_boost": 4,
            "cuk": 6, "sepic": 6, "flyback": 5, "forward": 6,
            "inverting_amp": 2, "noninverting_amp": 2,
            "instrumentation_amp": 4, "differential_amp": 2,
            "sallen_key_lowpass": 4, "sallen_key_highpass": 4,
            "sallen_key_bandpass": 5, "wien_bridge": 4, "colpitts": 7,
        }
        for topo, pairs in TOPOLOGY_ADJACENCY.items():
            n = comp_counts.get(topo, 20)
            for i, j in pairs:
                assert 0 <= i < n, f"{topo}: index {i} out of range (n_comp={n})"
                assert 0 <= j < n, f"{topo}: index {j} out of range (n_comp={n})"


class TestTwoHeadModel:
    """Test TwoHeadARCSModel."""

    def test_instantiation(self, config):
        model = TwoHeadARCSModel(config)
        n_params = model.count_parameters()
        assert n_params > 0
        # Should have more params than baseline (extra value_proj + value_head)
        baseline = ARCSModel(config)
        assert n_params > baseline.count_parameters()

    def test_parameter_groups(self, config):
        model = TwoHeadARCSModel(config)
        groups = model.count_parameters_by_group()
        assert groups["value_proj"] > 0
        assert groups["value_head"] > 0
        assert groups["attention"] > 0
        assert groups["ffn"] > 0

    def test_forward_no_targets(self, config, buck_ids):
        model = TwoHeadARCSModel(config)
        model.eval()
        ids = torch.tensor([buck_ids])
        logits, loss = model(ids)
        assert logits.shape == (1, len(buck_ids), config.vocab_size)
        assert loss is None

    def test_forward_with_targets(self, config, buck_ids):
        model = TwoHeadARCSModel(config)
        ids = torch.tensor([buck_ids[:-1]])
        targets = torch.tensor([buck_ids[1:]])
        value_mask = torch.zeros(1, len(buck_ids) - 1, dtype=torch.bool)
        # Mark value positions
        for i in range(len(buck_ids) - 1):
            tid = buck_ids[1:][i]
            if 0 <= tid < len(model.config.vocab_size if isinstance(model.config.vocab_size, list) else [0] * 700):
                pass
        logits, loss = model(ids, targets=targets, value_mask=value_mask)
        assert logits.shape[0] == 1
        assert loss is not None
        assert loss.item() > 0

    def test_forward_with_value_mask(self, config, tokenizer, buck_ids):
        model = TwoHeadARCSModel(config)
        ids = torch.tensor([buck_ids[:-1]])
        targets = torch.tensor([buck_ids[1:]])
        # Build real value mask
        vm = []
        for tid in buck_ids[1:]:
            if 0 <= tid < len(tokenizer.tokens):
                vm.append(tokenizer.tokens[tid].token_type == TokenType.VALUE)
            else:
                vm.append(False)
        value_mask = torch.tensor([vm], dtype=torch.bool)
        logits, loss = model(ids, targets=targets, value_mask=value_mask)
        assert loss is not None
        assert loss.item() > 0

    def test_generate(self, config, tokenizer):
        model = TwoHeadARCSModel(config)
        model.eval()
        prefix = torch.tensor([[tokenizer.start_id]])
        output = model.generate(
            prefix, max_new_tokens=10, temperature=1.0,
            tokenizer=tokenizer,
        )
        assert output.shape[1] > 1  # generated at least 1 token
        assert output.shape[1] <= 11

    def test_interface_compatible_with_baseline(self, config, buck_ids):
        """Both models should accept the same forward() signature."""
        baseline = ARCSModel(config)
        two_head = TwoHeadARCSModel(config)
        ids = torch.tensor([buck_ids[:-1]])
        targets = torch.tensor([buck_ids[1:]])

        b_logits, b_loss = baseline(ids, targets=targets)
        t_logits, t_loss = two_head(ids, targets=targets)

        assert b_logits.shape == t_logits.shape
        assert b_loss is not None and t_loss is not None


class TestGraphTransformerModel:
    """Test GraphTransformerARCSModel."""

    def test_instantiation(self, config):
        model = GraphTransformerARCSModel(config)
        n_params = model.count_parameters()
        assert n_params > 0

    def test_parameter_groups(self, config):
        model = GraphTransformerARCSModel(config)
        groups = model.count_parameters_by_group()
        assert groups["rwpe_proj"] > 0
        assert groups["graph_bias"] > 0
        assert groups["value_proj"] > 0

    def test_forward_no_graph(self, config, buck_ids):
        """Should work without graph features (falls back to no bias)."""
        model = GraphTransformerARCSModel(config)
        model.eval()
        ids = torch.tensor([buck_ids])
        logits, loss = model(ids)
        assert logits.shape == (1, len(buck_ids), config.vocab_size)

    def test_forward_with_tokenizer(self, config, tokenizer, buck_ids):
        """When tokenizer is provided, graph features are auto-computed."""
        model = GraphTransformerARCSModel(config)
        model.eval()
        ids = torch.tensor([buck_ids])
        logits, loss = model(ids, tokenizer=tokenizer)
        assert logits.shape == (1, len(buck_ids), config.vocab_size)

    def test_forward_with_targets(self, config, tokenizer, buck_ids):
        model = GraphTransformerARCSModel(config)
        ids = torch.tensor([buck_ids[:-1]])
        targets = torch.tensor([buck_ids[1:]])
        logits, loss = model(ids, targets=targets, tokenizer=tokenizer)
        assert loss is not None
        assert loss.item() > 0

    def test_compute_graph_features(self, config, tokenizer, buck_ids):
        """Graph features should reflect buck topology adjacency."""
        ids = torch.tensor([buck_ids])
        g_adj, e_types, rwpe_features = GraphTransformerARCSModel.compute_graph_features(
            ids, tokenizer
        )
        assert g_adj.shape == (1, len(buck_ids), len(buck_ids))
        assert e_types.shape == (1, len(buck_ids), len(buck_ids))
        # RWPE features: (B, T, K_WALK=8)
        from arcs.model_enhanced import K_WALK
        assert rwpe_features.shape == (1, len(buck_ids), K_WALK)

        # Buck has 4 components — should have adjacency entries
        assert g_adj.sum() > 0, "No adjacency detected for buck topology"

        # RWPE should have non-zero entries at component positions
        assert rwpe_features.abs().sum() > 0, "No RWPE features set"
        # Return probabilities at k=2 should distinguish degree-2 and degree-1 nodes
        # (buck: nodes 0,1 have degree 2; nodes 2,3 have degree 1)

    def test_adjacency_matches_buck_topology(self, config, tokenizer, buck_ids):
        """Verify adjacency matches TOPOLOGY_ADJACENCY['buck']."""
        ids = torch.tensor([buck_ids])
        g_adj, _, _ = GraphTransformerARCSModel.compute_graph_features(ids, tokenizer)

        # Find component positions (COMP_* tokens)
        comp_positions = []
        for i, tid in enumerate(buck_ids):
            if 0 <= tid < len(tokenizer.tokens):
                if tokenizer.tokens[tid].token_type == TokenType.COMPONENT:
                    comp_positions.append(i)

        assert len(comp_positions) == 4  # INDUCTOR, CAPACITOR, RESISTOR, MOSFET_N

        # Buck adjacency: (0,3), (0,1), (1,2)
        # Positions: IND=comp[0], CAP=comp[1], RES=comp[2], MOS=comp[3]
        p = comp_positions
        assert g_adj[0, p[0], p[3]] == 1.0, "INDUCTOR-MOSFET should be adjacent"
        assert g_adj[0, p[0], p[1]] == 1.0, "INDUCTOR-CAPACITOR should be adjacent"
        assert g_adj[0, p[1], p[2]] == 1.0, "CAPACITOR-RESISTOR should be adjacent"
        # Non-adjacent pairs should be 0
        assert g_adj[0, p[2], p[3]] == 0.0, "RESISTOR-MOSFET should NOT be adjacent"

    def test_generate(self, config, tokenizer):
        model = GraphTransformerARCSModel(config)
        model.eval()
        prefix = torch.tensor([[tokenizer.start_id]])
        output = model.generate(
            prefix, max_new_tokens=10, temperature=1.0,
            tokenizer=tokenizer,
        )
        assert output.shape[1] > 1

    def test_generate_without_tokenizer(self, config, tokenizer):
        """Generate should work without tokenizer (no graph bias, uses struct head)."""
        model = GraphTransformerARCSModel(config)
        model.eval()
        prefix = torch.tensor([[tokenizer.start_id]])
        output = model.generate(prefix, max_new_tokens=5, temperature=1.0)
        assert output.shape[1] > 1


class TestModelFactory:
    """Test create_model() and load_model() factory functions."""

    def test_create_baseline(self, config):
        model = create_model("baseline", config)
        assert isinstance(model, ARCSModel)

    def test_create_two_head(self, config):
        model = create_model("two_head", config)
        assert isinstance(model, TwoHeadARCSModel)

    def test_create_graph_transformer(self, config):
        model = create_model("graph_transformer", config)
        assert isinstance(model, GraphTransformerARCSModel)

    def test_create_invalid(self, config):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("nonexistent", config)

    def test_save_load_roundtrip_baseline(self, config):
        model = create_model("baseline", config)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "model_type": "baseline",
            }, f.name)
            loaded, loaded_cfg, mt = load_model(f.name)
            assert mt == "baseline"
            assert isinstance(loaded, ARCSModel)

    def test_save_load_roundtrip_two_head(self, config):
        model = create_model("two_head", config)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "model_type": "two_head",
            }, f.name)
            loaded, loaded_cfg, mt = load_model(f.name)
            assert mt == "two_head"
            assert isinstance(loaded, TwoHeadARCSModel)

    def test_save_load_roundtrip_graph_transformer(self, config):
        model = create_model("graph_transformer", config)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "model_type": "graph_transformer",
            }, f.name)
            loaded, loaded_cfg, mt = load_model(f.name)
            assert mt == "graph_transformer"
            assert isinstance(loaded, GraphTransformerARCSModel)

    def test_load_defaults_to_baseline(self, config):
        """Old checkpoints without model_type should load as baseline."""
        model = ARCSModel(config)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                # No model_type key
            }, f.name)
            loaded, _, mt = load_model(f.name)
            assert mt == "baseline"
            assert isinstance(loaded, ARCSModel)


class TestGraphFeaturesEdgeCases:
    """Edge cases for graph feature computation."""

    def test_empty_sequence(self, config, tokenizer):
        """Sequence with no components should produce zero graph features."""
        ids = torch.tensor([[tokenizer.start_id, tokenizer.end_id]])
        g_adj, e_types, rwpe_f = GraphTransformerARCSModel.compute_graph_features(
            ids, tokenizer
        )
        assert g_adj.sum() == 0
        assert e_types.sum() == 0
        assert rwpe_f.sum() == 0

    def test_unknown_topology(self, config, tokenizer):
        """Sequence without valid TOPO_X should still work (no adjacency bias)."""
        # Just START, SEP, some components, END -- no TOPO token
        ids_list = [
            tokenizer.start_id,
            tokenizer.sep_id,
            tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(1000),
            tokenizer.name_to_id["COMP_CAPACITOR"], tokenizer.encode_value(1e-6),
            tokenizer.end_id,
        ]
        ids = torch.tensor([ids_list])
        g_adj, e_types, rwpe_f = GraphTransformerARCSModel.compute_graph_features(
            ids, tokenizer
        )
        # No topology -> no adjacency from TOPOLOGY_ADJACENCY
        assert g_adj.sum() == 0
        # No topology -> no RWPE features (they come from precomputed tables)
        assert rwpe_f.abs().sum() == 0

    def test_batch_of_different_topologies(self, config, tokenizer):
        """Graph features should handle batches with different topologies."""
        buck_ids = [
            tokenizer.start_id, tokenizer.name_to_id["TOPO_BUCK"], tokenizer.sep_id,
            tokenizer.name_to_id["COMP_INDUCTOR"], tokenizer.encode_value(1e-4),
            tokenizer.name_to_id["COMP_CAPACITOR"], tokenizer.encode_value(1e-5),
            tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(0.01),
            tokenizer.name_to_id["COMP_MOSFET_N"], tokenizer.encode_value(0.05),
            tokenizer.end_id,
        ]
        inv_ids = [
            tokenizer.start_id, tokenizer.name_to_id["TOPO_INVERTING_AMP"],
            tokenizer.sep_id,
            tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(10000),
            tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(1000),
            tokenizer.end_id,
        ]
        # Pad shorter sequence
        max_len = max(len(buck_ids), len(inv_ids))
        buck_ids += [tokenizer.pad_id] * (max_len - len(buck_ids))
        inv_ids += [tokenizer.pad_id] * (max_len - len(inv_ids))

        ids = torch.tensor([buck_ids, inv_ids])
        g_adj, e_types, rwpe_f = GraphTransformerARCSModel.compute_graph_features(
            ids, tokenizer
        )
        assert g_adj.shape == (2, max_len, max_len)
        # Both should have some adjacency
        assert g_adj[0].sum() > 0, "Buck should have adjacency"
        assert g_adj[1].sum() > 0, "Inverting amp should have adjacency"
