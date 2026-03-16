"""Tests for latent-space reward predictor and refinement."""

from __future__ import annotations

import torch
import pytest

from arcs.latent_reward import (
    LatentRewardConfig,
    LatentRewardPredictor,
    LatentRefinement,
    LatentRewardTrainer,
)


@pytest.fixture
def config():
    return LatentRewardConfig(
        latent_dim=16,
        spec_dim=32,
        hidden_dim=64,
        n_layers=2,
        n_refine_steps=5,
        refine_lr=0.05,
    )


@pytest.fixture
def predictor(config):
    return LatentRewardPredictor(config)


class TestLatentRewardPredictor:
    def test_forward_shape(self, predictor, config):
        z = torch.randn(4, config.latent_dim)
        spec = torch.randn(4, config.spec_dim)
        out = predictor(z, spec)
        assert out.shape == (4,)

    def test_batch_independence(self, predictor, config):
        """Each sample should be processed independently."""
        predictor.eval()  # disable dropout for deterministic comparison
        z = torch.randn(2, config.latent_dim)
        spec = torch.randn(2, config.spec_dim)
        out = predictor(z, spec)
        out1 = predictor(z[:1], spec[:1])
        assert torch.allclose(out[0], out1[0], atol=1e-5)

    def test_gradient_flows(self, predictor, config):
        z = torch.randn(4, config.latent_dim, requires_grad=True)
        spec = torch.randn(4, config.spec_dim)
        out = predictor(z, spec)
        out.sum().backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape


class TestLatentRefinement:
    def test_refine_returns_correct_shapes(self, predictor, config):
        refinement = LatentRefinement(predictor, config)
        z = torch.randn(4, config.latent_dim)
        spec = torch.randn(4, config.spec_dim)
        refined_z, stats = refinement.refine(z, spec)
        assert refined_z.shape == z.shape
        assert "initial_reward" in stats
        assert "final_reward" in stats
        assert "reward_improvement" in stats

    def test_refine_changes_z(self, predictor, config):
        refinement = LatentRefinement(predictor, config)
        z = torch.randn(4, config.latent_dim)
        spec = torch.randn(4, config.spec_dim)
        refined_z, _ = refinement.refine(z, spec)
        # z should change after refinement
        assert not torch.allclose(z, refined_z, atol=1e-6)

    def test_refine_with_constraint_fn(self, predictor, config):
        refinement = LatentRefinement(predictor, config)
        z = torch.randn(4, config.latent_dim)
        spec = torch.randn(4, config.spec_dim)

        def dummy_constraints(z_opt):
            return z_opt.norm(dim=-1).mean()

        refined_z, stats = refinement.refine(z, spec, constraints_fn=dummy_constraints)
        assert refined_z.shape == z.shape

    def test_drift_bounded(self, predictor, config):
        config.max_z_drift = 1.0
        config.n_refine_steps = 20
        config.refine_lr = 0.1
        refinement = LatentRefinement(predictor, config)
        z = torch.randn(4, config.latent_dim)
        spec = torch.randn(4, config.spec_dim)
        refined_z, stats = refinement.refine(z, spec)
        # Drift should be bounded (not strictly, but penalized)
        assert stats["final_drift"] < 10.0  # generous bound


class TestLatentRewardTrainer:
    def test_train_step(self, predictor, config):
        trainer = LatentRewardTrainer(predictor, lr=1e-3)
        z = torch.randn(8, config.latent_dim)
        spec = torch.randn(8, config.spec_dim)
        reward = torch.rand(8) * 8.0
        stats = trainer.train_step(z, spec, reward)
        assert "loss" in stats
        assert "correlation" in stats
        assert stats["step"] == 1

    def test_loss_decreases(self, predictor, config):
        trainer = LatentRewardTrainer(predictor, lr=1e-2)
        # Fixed data
        z = torch.randn(16, config.latent_dim)
        spec = torch.randn(16, config.spec_dim)
        reward = torch.rand(16) * 8.0

        losses = []
        for _ in range(50):
            stats = trainer.train_step(z, spec, reward)
            losses.append(stats["loss"])

        # Loss should decrease over training
        assert losses[-1] < losses[0]

    def test_state_dict_roundtrip(self, predictor, config):
        trainer = LatentRewardTrainer(predictor, lr=1e-3)
        z = torch.randn(4, config.latent_dim)
        spec = torch.randn(4, config.spec_dim)
        reward = torch.rand(4) * 8.0
        trainer.train_step(z, spec, reward)

        state = trainer.state_dict()
        assert "predictor" in state
        assert "optimizer" in state
        assert state["step_count"] == 1
