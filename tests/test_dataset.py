"""Tests for CircuitDataset (synthetic data, no real files needed)."""

import json
import pytest
import tempfile
from pathlib import Path

from arcs.tokenizer import CircuitTokenizer
from arcs.dataset import CircuitDataset
from arcs.datagen import CircuitSample


@pytest.fixture
def sample_data_file(tmp_path):
    """Create a temporary JSONL file with synthetic circuit samples."""
    samples = [
        CircuitSample(
            topology="buck",
            parameters={"inductance": 22e-6, "capacitance": 470e-6,
                        "r_dson": 0.05, "esr": 0.01, "r_load": 5.0},
            operating_conditions={"vin": 12.0, "vout": 5.0,
                                  "iout": 1.0, "fsw": 100000.0},
            metrics={"efficiency": 92.5, "vout_avg": 4.95,
                     "vout_ripple": 0.03, "vout_error_pct": 1.0,
                     "ripple_ratio": 0.006},
            valid=True, sim_time=1.5,
        ),
        CircuitSample(
            topology="boost",
            parameters={"inductance": 47e-6, "capacitance": 220e-6,
                        "r_dson": 0.05, "esr": 0.02, "r_load": 12.0},
            operating_conditions={"vin": 5.0, "vout": 12.0,
                                  "iout": 0.5, "fsw": 100000.0},
            metrics={"efficiency": 88.0, "vout_avg": 11.8,
                     "vout_ripple": 0.1, "vout_error_pct": 1.7,
                     "ripple_ratio": 0.008},
            valid=True, sim_time=2.0,
        ),
        CircuitSample(
            topology="buck_boost",
            parameters={"inductance": 33e-6, "capacitance": 330e-6,
                        "r_dson": 0.05, "esr": 0.015, "r_load": 9.0},
            operating_conditions={"vin": 12.0, "vout": -9.0,
                                  "iout": 1.0, "fsw": 100000.0},
            metrics={}, valid=False, sim_time=0.0,
            error_message="Sim failed",
        ),
    ]
    fpath = tmp_path / "test.jsonl"
    with open(fpath, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")
    return fpath


class TestCircuitDataset:
    """Dataset loading and batching."""

    def test_load(self, tokenizer, sample_data_file):
        ds = CircuitDataset(data_path=sample_data_file,
                            tokenizer=tokenizer, max_seq_len=64)
        # Should load only valid samples (2 out of 3)
        assert len(ds) >= 2

    def test_item_keys(self, tokenizer, sample_data_file):
        ds = CircuitDataset(data_path=sample_data_file,
                            tokenizer=tokenizer, max_seq_len=64)
        item = ds[0]
        assert "input_ids" in item
        assert "targets" in item
        assert "token_types" in item

    def test_item_shapes(self, tokenizer, sample_data_file):
        ds = CircuitDataset(data_path=sample_data_file,
                            tokenizer=tokenizer, max_seq_len=64)
        item = ds[0]
        assert item["input_ids"].ndim == 1
        assert item["targets"].shape == item["input_ids"].shape

    def test_max_seq_len_respected(self, tokenizer, sample_data_file):
        ds = CircuitDataset(data_path=sample_data_file,
                            tokenizer=tokenizer, max_seq_len=32)
        for i in range(len(ds)):
            assert ds[i]["input_ids"].shape[0] <= 32
