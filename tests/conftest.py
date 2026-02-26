"""Shared pytest fixtures for ARCS test suite."""

import pytest
import torch

from arcs.tokenizer import CircuitTokenizer
from arcs.model import ARCSModel, ARCSConfig


@pytest.fixture(scope="session")
def tokenizer():
    """Shared tokenizer instance."""
    return CircuitTokenizer()


@pytest.fixture(scope="session")
def device():
    """CPU device for deterministic testing."""
    return torch.device("cpu")


@pytest.fixture
def small_config(tokenizer):
    """Small model config for fast tests."""
    cfg = ARCSConfig.small()
    cfg.vocab_size = tokenizer.vocab_size
    return cfg


@pytest.fixture
def model(small_config, device):
    """Fresh small model on CPU."""
    m = ARCSModel(small_config).to(device)
    m.eval()
    return m


@pytest.fixture
def buck_token_ids(tokenizer):
    """Token sequence for a known-good buck converter."""
    return [
        tokenizer.name_to_id["START"],
        tokenizer.name_to_id["TOPO_BUCK"],
        tokenizer.sep_id,
        tokenizer.name_to_id["SPEC_VIN"], tokenizer.encode_value(12.0),
        tokenizer.name_to_id["SPEC_VOUT"], tokenizer.encode_value(5.0),
        tokenizer.name_to_id["SPEC_IOUT"], tokenizer.encode_value(1.0),
        tokenizer.name_to_id["SPEC_FSW"], tokenizer.encode_value(100000),
        tokenizer.sep_id,
        tokenizer.name_to_id["COMP_INDUCTOR"], tokenizer.encode_value(100e-6),
        tokenizer.name_to_id["COMP_CAPACITOR"], tokenizer.encode_value(47e-6),
        tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(0.01),
        tokenizer.name_to_id["COMP_MOSFET_N"], tokenizer.encode_value(0.05),
        tokenizer.name_to_id["END"],
    ]


@pytest.fixture
def inverting_amp_token_ids(tokenizer):
    """Token sequence for a known-good inverting amplifier."""
    return [
        tokenizer.name_to_id["START"],
        tokenizer.name_to_id["TOPO_INVERTING_AMP"],
        tokenizer.sep_id,
        tokenizer.name_to_id["SPEC_GAIN"], tokenizer.encode_value(10.0),
        tokenizer.name_to_id["SPEC_BANDWIDTH"], tokenizer.encode_value(1000.0),
        tokenizer.sep_id,
        tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(10000),
        tokenizer.name_to_id["COMP_RESISTOR"], tokenizer.encode_value(1000),
        tokenizer.name_to_id["END"],
    ]
