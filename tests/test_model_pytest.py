"""Tests for ARCS model architecture and inference."""

import pytest
import torch

from arcs.model import ARCSModel, ARCSConfig


class TestARCSConfig:
    """Config preset tests."""

    def test_small_config(self):
        cfg = ARCSConfig.small()
        assert cfg.d_model == 256
        assert cfg.n_layers == 6
        assert cfg.n_heads == 4

    def test_base_config(self):
        cfg = ARCSConfig.base()
        assert cfg.d_model > 256

    def test_large_config(self):
        cfg = ARCSConfig.large()
        assert cfg.d_model > ARCSConfig.base().d_model

    def test_roundtrip_dict(self):
        cfg = ARCSConfig.small()
        d = cfg.to_dict()
        cfg2 = ARCSConfig.from_dict(d)
        assert cfg2.d_model == cfg.d_model
        assert cfg2.n_layers == cfg.n_layers


class TestModelForward:
    """Forward pass and shape tests."""

    def test_forward_shape(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 20))
        t = torch.zeros_like(x)
        logits, loss = model(x, token_types=t)
        assert logits.shape == (2, 20, small_config.vocab_size)
        assert loss is None

    def test_forward_with_targets(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (2, 20))
        t = torch.zeros_like(x)
        logits, loss = model(x, token_types=t, targets=x)
        assert loss is not None
        assert loss.item() > 0

    def test_single_token(self, model, small_config):
        x = torch.randint(0, small_config.vocab_size, (1, 1))
        t = torch.zeros_like(x)
        logits, _ = model(x, token_types=t)
        assert logits.shape == (1, 1, small_config.vocab_size)

    def test_batch_size_independence(self, model, small_config):
        """Same input should produce same logits regardless of batch."""
        x = torch.randint(0, small_config.vocab_size, (1, 10))
        t = torch.zeros_like(x)
        logits1, _ = model(x, token_types=t)

        x2 = x.repeat(3, 1)
        t2 = t.repeat(3, 1)
        logits2, _ = model(x2, token_types=t2)
        torch.testing.assert_close(logits1[0], logits2[0], atol=1e-5, rtol=1e-5)


class TestModelGeneration:
    """Autoregressive generation tests."""

    def test_generate_produces_tokens(self, model, tokenizer, device):
        prefix = torch.tensor([[tokenizer.start_id]], device=device)
        gen = model.generate(prefix, max_new_tokens=15, temperature=0.8, top_k=50)
        assert gen.shape[0] == 1
        assert gen.shape[1] > 1  # At least prefix + 1 generated

    def test_generate_respects_max_tokens(self, model, tokenizer, device):
        prefix = torch.tensor([[tokenizer.start_id]], device=device)
        gen = model.generate(prefix, max_new_tokens=5, temperature=1.0, top_k=50)
        assert gen.shape[1] <= 1 + 5  # prefix + max_new_tokens

    def test_generate_temperature_zero(self, model, tokenizer, device):
        """Temperature near 0 → deterministic (greedy)."""
        prefix = torch.tensor([[tokenizer.start_id]], device=device)
        gen1 = model.generate(prefix, max_new_tokens=10, temperature=0.01, top_k=1)
        gen2 = model.generate(prefix, max_new_tokens=10, temperature=0.01, top_k=1)
        torch.testing.assert_close(gen1, gen2)


class TestModelParameters:
    """Parameter counting and structure tests."""

    def test_count_parameters(self, model):
        n = model.count_parameters()
        assert n > 0
        assert n < 100_000_000  # Reasonable upper bound

    def test_count_by_group(self, model):
        groups = model.count_parameters_by_group()
        assert isinstance(groups, dict)
        total = sum(groups.values())
        assert total == model.count_parameters()
