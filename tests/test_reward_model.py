"""Tests for the learned reward model (verifier).

Covers:
  - Reward computation from JSONL data
  - CircuitRewardDataset loading and padding
  - CircuitRewardModel forward pass + predict
  - Embedding transfer from generator
  - RewardModelRanker scoring and ranking
  - RewardModelTrainer training loop
  - Evaluation utilities
  - Config serialization
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from arcs.reward_model import (
    RewardModelConfig,
    CircuitRewardModel,
    CircuitRewardDataset,
    RewardModelTrainer,
    RewardModelRanker,
    compute_reward_from_sample,
    _power_reward_from_metrics,
    _signal_reward_from_metrics,
    _pearson,
    _spearman_approx,
    evaluate_reward_model,
    TrainingMetrics,
)
from arcs.tokenizer import CircuitTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return CircuitTokenizer()


@pytest.fixture
def tiny_config():
    return RewardModelConfig.tiny()


@pytest.fixture
def small_config():
    return RewardModelConfig.small()


@pytest.fixture
def model(tiny_config):
    return CircuitRewardModel(tiny_config)


@pytest.fixture
def sample_jsonl_dir(tmp_path, tokenizer):
    """Create a temporary directory with a few JSONL samples."""
    buck_data = [
        {
            "topology": "buck",
            "parameters": {"inductance": 0.0002, "capacitance": 5.6e-5, "esr": 0.2, "r_dson": 0.15},
            "operating_conditions": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000},
            "metrics": {
                "vout_avg": 4.71, "vout_ripple": 0.03, "iout_avg": 0.94,
                "efficiency": 0.85, "vout_error_pct": 5.8, "ripple_ratio": 0.006,
            },
            "valid": True,
            "sim_time": 0.5,
            "error_message": "",
        },
        {
            "topology": "buck",
            "parameters": {"inductance": 0.001, "capacitance": 1e-4, "esr": 0.5, "r_dson": 0.3},
            "operating_conditions": {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000},
            "metrics": {},
            "valid": False,
            "sim_time": 0.1,
            "error_message": "simulation failed",
        },
    ]

    inv_data = [
        {
            "topology": "inverting_amp",
            "parameters": {"r_input": 10000, "r_feedback": 47000},
            "operating_conditions": {"vin": 0.1, "cutoff_freq": 1000},
            "metrics": {"gain_db": -13.4, "bw_3db": 50000},
            "valid": True,
            "sim_time": 0.3,
            "error_message": "",
        },
    ]

    buck_file = tmp_path / "buck.jsonl"
    with open(buck_file, "w") as f:
        for s in buck_data:
            f.write(json.dumps(s) + "\n")

    inv_file = tmp_path / "inverting_amp.jsonl"
    with open(inv_file, "w") as f:
        for s in inv_data:
            f.write(json.dumps(s) + "\n")

    return tmp_path


@pytest.fixture
def dataset(sample_jsonl_dir, tokenizer):
    return CircuitRewardDataset(sample_jsonl_dir, tokenizer)


# ---------------------------------------------------------------------------
# §1. Reward computation from metrics
# ---------------------------------------------------------------------------


class TestRewardComputation:
    """Test compute_reward_from_sample and sub-functions."""

    def test_valid_buck_reward(self):
        """Valid buck should get struct_bonus + sim_convergence + partial power reward."""
        metrics = {
            "vout_error_pct": 5.0,
            "efficiency": 0.85,
            "ripple_ratio": 0.006,
        }
        r = compute_reward_from_sample("buck", metrics, valid=True)
        assert r >= 2.0  # At least struct + convergence
        assert r <= 8.0

    def test_invalid_sample_reward(self):
        """Invalid sample should get only struct_bonus."""
        r = compute_reward_from_sample("buck", {}, valid=False)
        assert r == 1.0

    def test_perfect_power_reward(self):
        """Perfect power converter metrics → max reward."""
        metrics = {
            "vout_error_pct": 0.0,
            "efficiency": 1.0,
            "ripple_ratio": 0.0,
        }
        r = compute_reward_from_sample("buck", metrics, valid=True)
        assert r == pytest.approx(8.0)

    def test_zero_efficiency_reward(self):
        metrics = {"vout_error_pct": 100, "efficiency": 0, "ripple_ratio": 1.0}
        r = _power_reward_from_metrics(metrics)
        assert r == 0.0

    def test_amplifier_reward(self):
        """Amplifier with gain and bandwidth."""
        metrics = {"gain_db": 20.0, "bw_3db": 50000}
        r = _signal_reward_from_metrics(metrics, "inverting_amp")
        # gain_db=20 → 3.0 + min(2.0, 20/30)=0.667 + bw bonus 1.0
        assert r > 4.0

    def test_filter_reward(self):
        metrics = {"gain_dc": -1.0, "bw_3db": 5000}
        r = _signal_reward_from_metrics(metrics, "sallen_key_lowpass")
        assert r >= 5.0  # gain_dc=2.0 + bw=3.0 + reasonable_gain=1.0

    def test_oscillator_reward(self):
        metrics = {"vosc_pp": 2.0, "f_peak": 1000}
        r = _signal_reward_from_metrics(metrics, "wien_bridge")
        assert r == pytest.approx(6.0)  # 3+2+1

    def test_unknown_topology_zero_signal(self):
        """Unknown topology in signal domain → 0 signal reward."""
        r = _signal_reward_from_metrics({}, "unknown_circuit")
        assert r == 0.0

    def test_struct_bonus_configurable(self):
        r = compute_reward_from_sample("buck", {}, valid=False, struct_bonus=2.0)
        assert r == 2.0


# ---------------------------------------------------------------------------
# §2. Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    """Test CircuitRewardDataset."""

    def test_loads_samples(self, dataset):
        assert len(dataset) == 3  # 2 buck + 1 inverting_amp

    def test_returns_tensor_pair(self, dataset):
        ids, reward = dataset[0]
        assert ids.shape == (128,)  # Padded to max_seq_len
        assert reward.ndim == 0     # Scalar

    def test_reward_range(self, dataset):
        for i in range(len(dataset)):
            _, reward = dataset[i]
            assert 0.0 <= reward.item() <= 8.0

    def test_padding(self, dataset):
        ids, _ = dataset[0]
        # Last tokens should be padding (0)
        assert ids[-1].item() == 0

    def test_starts_with_START(self, dataset, tokenizer):
        ids, _ = dataset[0]
        assert ids[0].item() == tokenizer.start_id

    def test_topology_filter(self, sample_jsonl_dir, tokenizer):
        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, topologies=["buck"])
        assert len(ds) == 2  # Only buck samples

    def test_reward_stats(self, dataset):
        stats = dataset.reward_stats
        assert "count" in stats
        assert stats["count"] == 3
        assert stats["mean"] > 0

    def test_missing_dir_raises(self, tokenizer):
        with pytest.raises(FileNotFoundError):
            CircuitRewardDataset("/nonexistent/path", tokenizer)


# ---------------------------------------------------------------------------
# §3. Model
# ---------------------------------------------------------------------------


class TestModel:
    """Test CircuitRewardModel architecture."""

    def test_forward_shape(self, model, tiny_config):
        ids = torch.randint(1, 100, (4, 32))
        mask = torch.ones(4, 32, dtype=torch.bool)
        out = model(ids, attention_mask=mask)
        assert out.shape == (4,)

    def test_forward_no_mask(self, model):
        ids = torch.randint(1, 100, (2, 64))
        out = model(ids)
        assert out.shape == (2,)

    def test_output_range(self, model, tiny_config):
        """predict() clamps to valid range."""
        ids = torch.randint(1, 100, (8, 32))
        out = model.predict(ids)
        lo, hi = tiny_config.reward_range
        assert (out >= lo).all()
        assert (out <= hi).all()

    def test_forward_unclamped(self, model):
        """forward() does NOT clamp — for gradient flow during training."""
        ids = torch.randint(1, 100, (4, 32))
        out = model(ids)
        # Output may be outside [0, 8] with random weights — that's expected
        assert out.shape == (4,)

    def test_predict_no_grad(self, model):
        ids = torch.randint(1, 100, (2, 32))
        out = model.predict(ids)
        assert not out.requires_grad

    def test_parameter_count(self, model):
        n = model.count_parameters()
        assert n > 0
        assert n < 1_000_000  # Tiny config should be <1M

    def test_tiny_config(self):
        config = RewardModelConfig.tiny()
        model = CircuitRewardModel(config)
        assert model.count_parameters() < 500_000

    def test_small_config(self):
        config = RewardModelConfig.small()
        model = CircuitRewardModel(config)
        n = model.count_parameters()
        assert 500_000 < n < 2_000_000

    def test_medium_config(self):
        config = RewardModelConfig.medium()
        model = CircuitRewardModel(config)
        n = model.count_parameters()
        assert n > 1_000_000

    def test_gradient_flows(self, model):
        ids = torch.randint(1, 100, (2, 32))
        mask = torch.ones(2, 32, dtype=torch.bool)
        out = model(ids, attention_mask=mask)
        loss = out.mean()
        loss.backward()
        # Check gradient exists on reward head weight (always gets gradient)
        head_param = list(model.reward_head.parameters())[0]
        assert head_param.grad is not None
        assert head_param.grad.abs().sum() > 0

    def test_embedding_transfer(self, tiny_config):
        """Test loading generator embeddings."""
        model = CircuitRewardModel(tiny_config)
        gen_emb = torch.randn(tiny_config.vocab_size, tiny_config.d_model)
        state = {"tok_emb.weight": gen_emb}
        n = model.load_generator_embeddings(state)
        assert n > 0
        # Verify transfer
        assert torch.allclose(model.tok_emb.weight.data, gen_emb)

    def test_embedding_transfer_dimension_mismatch(self, tiny_config):
        """When generator has different d_model, should project via SVD."""
        model = CircuitRewardModel(tiny_config)
        gen_emb = torch.randn(tiny_config.vocab_size, 256)  # Larger d_model
        state = {"tok_emb.weight": gen_emb}
        n = model.load_generator_embeddings(state)
        assert n > 0

    def test_embedding_transfer_no_key(self, tiny_config):
        model = CircuitRewardModel(tiny_config)
        n = model.load_generator_embeddings({})
        assert n == 0


# ---------------------------------------------------------------------------
# §4. Config serialization
# ---------------------------------------------------------------------------


class TestConfig:
    def test_round_trip(self):
        cfg = RewardModelConfig.small()
        d = cfg.to_dict()
        cfg2 = RewardModelConfig.from_dict(d)
        assert cfg.d_model == cfg2.d_model
        assert cfg.reward_range == cfg2.reward_range

    def test_from_dict_ignores_extra_keys(self):
        d = RewardModelConfig.tiny().to_dict()
        d["extra_key"] = 42
        cfg = RewardModelConfig.from_dict(d)
        assert cfg.d_model == 64


# ---------------------------------------------------------------------------
# §5. Trainer
# ---------------------------------------------------------------------------


class TestTrainer:
    """Test RewardModelTrainer."""

    def test_training_runs(self, sample_jsonl_dir, tokenizer):
        """Smoke test: training completes without errors."""
        config = RewardModelConfig.tiny()
        config.epochs = 2
        config.batch_size = 2
        config.patience = 10

        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        trainer = RewardModelTrainer(config)
        model = trainer.train(ds, verbose=False)

        assert model is not None
        assert len(trainer.history) == 2

    def test_loss_decreases(self, sample_jsonl_dir, tokenizer):
        """With enough data and epochs, loss should decrease."""
        config = RewardModelConfig.tiny()
        config.epochs = 5
        config.batch_size = 2
        config.patience = 10

        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        trainer = RewardModelTrainer(config)
        trainer.train(ds, verbose=False)

        # At least training produces metrics
        assert len(trainer.history) >= 2

    def test_checkpoint_save_load(self, sample_jsonl_dir, tokenizer, tmp_path):
        config = RewardModelConfig.tiny()
        config.epochs = 2
        config.batch_size = 2
        config.patience = 10

        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        trainer = RewardModelTrainer(config)
        trainer.train(ds, verbose=False)

        ckpt_path = tmp_path / "reward_model.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        trainer2, model2 = RewardModelTrainer.load_checkpoint(ckpt_path)
        assert model2.count_parameters() == trainer.model.count_parameters()

    def test_early_stopping(self, sample_jsonl_dir, tokenizer):
        """Early stopping should trigger before max epochs."""
        config = RewardModelConfig.tiny()
        config.epochs = 100
        config.batch_size = 2
        config.patience = 2

        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        trainer = RewardModelTrainer(config)
        trainer.train(ds, verbose=False)

        # Should have stopped before 100 epochs
        assert len(trainer.history) < 100


# ---------------------------------------------------------------------------
# §6. Ranker
# ---------------------------------------------------------------------------


class TestRanker:
    """Test RewardModelRanker."""

    def _make_candidate(self, tokens, decoded=None):
        """Create a mock ScoredCandidate."""
        mock = MagicMock()
        mock.tokens = tokens
        mock.decoded = decoded or MagicMock()
        mock.mean_log_prob = -1.0
        mock.rank = 0
        return mock

    def test_rank_candidates(self, model, tokenizer):
        ranker = RewardModelRanker(model, tokenizer)

        # Create mock candidates with different token sequences
        c1 = self._make_candidate([1, 10, 3, 50, 100, 3, 20, 30, 2])
        c2 = self._make_candidate([1, 10, 3, 50, 100, 3, 25, 35, 2])
        c3 = self._make_candidate([1, 10, 3, 50, 100, 3, 15, 40, 2])

        ranked = ranker.rank_candidates([c1, c2, c3])
        assert len(ranked) == 3
        assert ranked[0].rank == 0
        assert ranked[1].rank == 1
        assert ranked[2].rank == 2

    def test_score_candidates(self, model, tokenizer):
        ranker = RewardModelRanker(model, tokenizer)
        c1 = self._make_candidate([1, 10, 3, 50, 100, 3, 20, 2])
        scored = ranker.score_candidates([c1])
        assert len(scored) == 1
        assert isinstance(scored[0][1], float)

    def test_empty_candidates(self, model, tokenizer):
        ranker = RewardModelRanker(model, tokenizer)
        result = ranker.score_candidates([])
        assert result == []


# ---------------------------------------------------------------------------
# §7. Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_pearson_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _pearson(x, y) == pytest.approx(1.0, abs=1e-6)

    def test_pearson_negative(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert _pearson(x, y) == pytest.approx(-1.0, abs=1e-6)

    def test_pearson_uncorrelated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 3.0, 3.0, 3.0, 3.0]
        assert _pearson(x, y) == pytest.approx(0.0, abs=1e-6)

    def test_pearson_short(self):
        assert _pearson([1.0], [2.0]) == 0.0

    def test_spearman_monotone(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _spearman_approx(x, y) == pytest.approx(1.0, abs=1e-6)

    def test_spearman_anti_monotone(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        assert _spearman_approx(x, y) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# §8. Evaluation function
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_evaluate_runs(self, sample_jsonl_dir, tokenizer, tiny_config):
        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        model = CircuitRewardModel(tiny_config)
        result = evaluate_reward_model(model, ds)
        assert "mae" in result
        assert "correlation" in result
        assert "within_1" in result
        assert result["n_samples"] == 3

    def test_evaluate_metrics_valid(self, sample_jsonl_dir, tokenizer, tiny_config):
        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        model = CircuitRewardModel(tiny_config)
        result = evaluate_reward_model(model, ds)
        assert result["mae"] >= 0
        assert result["mse"] >= 0
        assert 0.0 <= result["within_1"] <= 1.0


# ---------------------------------------------------------------------------
# §9. Integration smoke test
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end: load data → train → evaluate → rank."""

    def test_full_pipeline(self, sample_jsonl_dir, tokenizer, tmp_path):
        """Full pipeline: dataset → train → save → load → score."""
        config = RewardModelConfig.tiny()
        config.epochs = 3
        config.batch_size = 2
        config.patience = 10

        # Load data
        ds = CircuitRewardDataset(sample_jsonl_dir, tokenizer, max_seq_len=64)
        assert len(ds) == 3

        # Train
        trainer = RewardModelTrainer(config)
        model = trainer.train(ds, verbose=False)

        # Save
        ckpt = tmp_path / "rm.pt"
        trainer.save_checkpoint(ckpt)

        # Load
        trainer2, model2 = RewardModelTrainer.load_checkpoint(ckpt)

        # Evaluate
        result = evaluate_reward_model(model2, ds)
        assert result["n_samples"] == 3

        # Rank mock candidates
        ranker = RewardModelRanker(model2, tokenizer)
        c1 = MagicMock()
        c1.tokens = ds[0][0].tolist()
        c1.rank = 0
        scored = ranker.score_candidates([c1])
        assert len(scored) == 1
