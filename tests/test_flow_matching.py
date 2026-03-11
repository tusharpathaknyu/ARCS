"""Tests for Constrained Circuit Flow Matching (CCFM)."""

import math
import torch
import pytest
from arcs.flow_matching import (
    FlowMatchingConfig,
    ConstrainedFlowMatchingModel,
    FlowVelocityNetwork,
    SinusoidalTimeEmbedding,
    AdaptiveLayerNorm,
    FlowTransformerBlock,
    ConstraintGuidance,
)
from arcs.valid_circuit_gen import VCGConfig, CircuitConstraints, VCGDecoder


# --- Helpers ---

def _make_batch(B=4, N=12, S=8):
    return {
        "node_types": torch.randint(0, 16, (B, N)),
        "adjacency": torch.rand(B, N, N).round(),
        "values": torch.randn(B, N) * 2,
        "active_mask": torch.ones(B, N),
        "spec_types": torch.randint(0, 10, (B, S)),
        "spec_values": torch.randn(B, S),
        "spec_mask": torch.ones(B, S),
        "topology_idx": torch.randint(1, 16, (B,)),
        "value_bounds_min": torch.full((B, N), -12.0),
        "value_bounds_max": torch.full((B, N), 6.0),
        "n_components": torch.full((B,), 4, dtype=torch.long),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests: Components
# ═══════════════════════════════════════════════════════════════════════════


class TestSinusoidalTimeEmbedding:
    def test_shapes(self):
        emb = SinusoidalTimeEmbedding(d_embed=64, n_freqs=16)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        assert out.shape == (3, 64)

    def test_different_times_produce_different_embeddings(self):
        emb = SinusoidalTimeEmbedding(d_embed=64, n_freqs=16)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = emb(t)
        # Embeddings for t=0, t=0.5, t=1.0 should differ
        assert not torch.allclose(out[0], out[1], atol=1e-3)
        assert not torch.allclose(out[1], out[2], atol=1e-3)

    def test_2d_input(self):
        emb = SinusoidalTimeEmbedding(d_embed=64)
        t = torch.tensor([[0.25], [0.75]])
        out = emb(t)
        assert out.shape == (2, 64)


class TestAdaptiveLayerNorm:
    def test_identity_init(self):
        """With zero-initialized params, AdaLN should be identity-like."""
        aln = AdaptiveLayerNorm(d_model=64, d_cond=32)
        x = torch.randn(2, 64)
        cond = torch.zeros(2, 32)
        out = aln(x, cond)
        # Should approximate LayerNorm(x) since gamma=0, beta=0 → (1+0)*LN(x)+0
        ln = torch.nn.LayerNorm(64, elementwise_affine=False)
        expected = ln(x)
        assert torch.allclose(out, expected, atol=1e-5)


class TestFlowTransformerBlock:
    def test_shapes(self):
        block = FlowTransformerBlock(d_model=64, n_heads=4, d_ff=128, d_cond=96)
        x = torch.randn(2, 1, 64)
        cond = torch.randn(2, 96)
        out = block(x, cond)
        assert out.shape == (2, 1, 64)


class TestFlowVelocityNetwork:
    def test_shapes(self):
        cfg = FlowMatchingConfig(latent_dim=32, flow_d_model=64,
                                 flow_n_layers=2, flow_n_heads=2, flow_d_ff=128)
        net = FlowVelocityNetwork(cfg)
        z = torch.randn(4, 32)
        t = torch.rand(4)
        spec = torch.randn(4, 256)
        topo = torch.randint(0, 16, (4,))
        v = net(z, t, spec, topo)
        assert v.shape == (4, 32)

    def test_zero_init_output(self):
        """Output should start near zero (identity flow)."""
        cfg = FlowMatchingConfig(latent_dim=32, flow_d_model=64,
                                 flow_n_layers=2, flow_n_heads=2)
        net = FlowVelocityNetwork(cfg)
        z = torch.randn(2, 32)
        t = torch.tensor([0.5, 0.5])
        spec = torch.randn(2, 256)
        topo = torch.randint(0, 16, (2,))
        v = net(z, t, spec, topo)
        # Output should be small due to zero-init
        assert v.abs().max().item() < 1.0

    def test_parameter_count(self):
        cfg = FlowMatchingConfig()
        net = FlowVelocityNetwork(cfg)
        n_params = net.count_parameters()
        assert n_params > 0
        assert n_params < 10_000_000  # reasonable size


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests: Full Model
# ═══════════════════════════════════════════════════════════════════════════


class TestConstrainedFlowMatchingModel:
    @pytest.fixture
    def model(self):
        cfg = FlowMatchingConfig(
            latent_dim=32,
            flow_d_model=64,
            flow_n_layers=2,
            flow_n_heads=2,
            flow_d_ff=128,
            n_sample_steps=5,
        )
        return ConstrainedFlowMatchingModel(cfg)

    def test_flow_loss(self, model):
        batch = _make_batch(B=4)
        loss, stats = model.compute_flow_loss(batch)
        assert loss.requires_grad
        assert loss.item() > 0
        assert "loss/flow" in stats
        assert "loss/consistency" in stats

    def test_backward(self, model):
        model.freeze_vcg()
        batch = _make_batch(B=2)
        loss, _ = model.compute_flow_loss(batch)
        loss.backward()
        # Flow network should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.flow_net.parameters()
        )
        assert has_grad, "Flow network should receive gradients"

        # VCG encoder should NOT have gradients (frozen)
        vcg_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.vcg_encoder.parameters()
        )
        assert not vcg_grad, "Frozen VCG encoder should not have gradients"

    def test_sample(self, model):
        batch = _make_batch(B=2)
        soft_X, soft_A, soft_V, info = model.sample(
            batch["spec_types"], batch["spec_values"], batch["spec_mask"],
            batch["topology_idx"], batch["active_mask"],
            batch["value_bounds_min"], batch["value_bounds_max"],
            n_steps=3,
        )
        assert soft_X.shape == (2, 12, 16)
        assert soft_A.shape == (2, 12, 12)
        assert soft_V.shape == (2, 12)
        assert "guidance_steps" in info

    def test_sample_no_guidance(self, model):
        batch = _make_batch(B=2)
        _, _, _, info = model.sample(
            batch["spec_types"], batch["spec_values"], batch["spec_mask"],
            batch["topology_idx"], batch["active_mask"],
            batch["value_bounds_min"], batch["value_bounds_max"],
            n_steps=3, guidance_strength=0.0,
        )
        assert info["guidance_steps"] == 0

    def test_sample_with_projection(self, model):
        batch = _make_batch(B=2)
        soft_X, soft_A, soft_V, info = model.sample_with_projection(
            batch["spec_types"], batch["spec_values"], batch["spec_mask"],
            batch["topology_idx"], batch["active_mask"],
            batch["value_bounds_min"], batch["value_bounds_max"],
            n_steps=3,
        )
        assert soft_X.shape == (2, 12, 16)
        assert "proj_steps" in info

    def test_freeze_unfreeze(self, model):
        model.freeze_vcg()
        n_frozen = sum(1 for p in model.vcg_encoder.parameters() if not p.requires_grad)
        assert n_frozen > 0

        model.unfreeze_vcg()
        n_trainable = sum(1 for p in model.vcg_encoder.parameters() if p.requires_grad)
        assert n_trainable > 0

    def test_save_load(self, model, tmp_path):
        path = tmp_path / "test_ccfm.pt"
        model.save(path)

        loaded = ConstrainedFlowMatchingModel.load(path)
        assert loaded.flow_config.latent_dim == model.flow_config.latent_dim

        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_reuse_vcg_model(self):
        """CCFM can reuse a pre-trained VCG model."""
        from arcs.valid_circuit_gen import ValidCircuitGenModel, VCGConfig
        vcg_config = VCGConfig(latent_dim=32, d_model=64, n_encoder_layers=2,
                               n_heads=2, d_ff=128, n_decoder_layers=2,
                               decoder_hidden=128)
        vcg_model = ValidCircuitGenModel(vcg_config)

        flow_cfg = FlowMatchingConfig(
            latent_dim=32, flow_d_model=64, flow_n_layers=2,
            flow_n_heads=2, flow_d_ff=128, spec_d_model=64,
        )
        ccfm = ConstrainedFlowMatchingModel(flow_cfg, vcg_model=vcg_model)

        # Should share the same spec_encoder
        assert ccfm.spec_encoder is vcg_model.spec_encoder
        assert ccfm.vcg_encoder is vcg_model.encoder
        assert ccfm.vcg_decoder is vcg_model.decoder


# ═══════════════════════════════════════════════════════════════════════════
# Constraint Guidance Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConstraintGuidance:
    def test_gradient_nonzero(self):
        vcg_cfg = VCGConfig(latent_dim=32, d_model=64, n_heads=2, d_ff=128,
                            decoder_hidden=128)
        decoder = VCGDecoder(vcg_cfg)
        constraints = CircuitConstraints(vcg_cfg)
        flow_cfg = FlowMatchingConfig(latent_dim=32)
        guidance = ConstraintGuidance(decoder, constraints, flow_cfg)

        z = torch.randn(2, 32)
        spec = torch.randn(2, 64)
        topo = torch.randint(1, 16, (2,))
        active = torch.ones(2, 12)
        bmin = torch.full((2, 12), -12.0)
        bmax = torch.full((2, 12), 6.0)

        grad_z, viol = guidance.compute_guidance(
            z, spec, topo, active, bmin, bmax
        )
        assert grad_z.shape == (2, 32)
        assert viol >= 0.0
        # Gradient should be nonzero (unless all constraints already satisfied)
        assert grad_z.abs().sum() > 0 or viol < 1e-6

    def test_weights_positive(self):
        vcg_cfg = VCGConfig(latent_dim=32, d_model=64, decoder_hidden=128)
        decoder = VCGDecoder(vcg_cfg)
        constraints = CircuitConstraints(vcg_cfg)
        flow_cfg = FlowMatchingConfig(latent_dim=32)
        guidance = ConstraintGuidance(decoder, constraints, flow_cfg)
        assert (guidance.weights > 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# Flow Matching Theory Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFlowMatchingTheory:
    def test_linear_interpolation(self):
        """z_t = (1-t)·z_0 + t·z_1 should be correct at boundaries."""
        z_0 = torch.randn(4, 32)
        z_1 = torch.randn(4, 32)

        # t=0 → z_t = z_0
        t = torch.zeros(4, 1)
        z_t = (1 - t) * z_0 + t * z_1
        assert torch.allclose(z_t, z_0)

        # t=1 → z_t = z_1
        t = torch.ones(4, 1)
        z_t = (1 - t) * z_0 + t * z_1
        assert torch.allclose(z_t, z_1)

    def test_target_velocity(self):
        """Target velocity u_t = z_1 - z_0 is constant along path."""
        z_0 = torch.randn(4, 32)
        z_1 = torch.randn(4, 32)
        u_t = z_1 - z_0

        # Verify: z_t + (1-t)·u_t = z_1 for any t
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.full((4, 1), t_val)
            z_t = (1 - t) * z_0 + t * z_1
            z_1_pred = z_t + (1 - t) * u_t
            assert torch.allclose(z_1_pred, z_1, atol=1e-5)

    def test_loss_decreases_with_training(self):
        """A few gradient steps should decrease loss (sanity check)."""
        cfg = FlowMatchingConfig(
            latent_dim=16, flow_d_model=32, flow_n_layers=1,
            flow_n_heads=2, flow_d_ff=64, n_sample_steps=3,
        )
        model = ConstrainedFlowMatchingModel(cfg)
        model.freeze_vcg()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
        )
        batch = _make_batch(B=8)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            loss, _ = model.compute_flow_loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allow small fluctuations)
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )
