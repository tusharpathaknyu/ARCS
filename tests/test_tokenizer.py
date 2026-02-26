"""Tests for CircuitTokenizer."""

import math
import pytest
from arcs.tokenizer import CircuitTokenizer, TokenType


class TestTokenizerVocabulary:
    """Vocabulary structure and consistency tests."""

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 686

    def test_special_tokens_exist(self, tokenizer):
        assert tokenizer.pad_id >= 0
        assert tokenizer.start_id >= 0
        assert tokenizer.end_id >= 0
        assert tokenizer.sep_id >= 0

    def test_special_tokens_unique(self, tokenizer):
        ids = {tokenizer.pad_id, tokenizer.start_id,
               tokenizer.end_id, tokenizer.sep_id}
        assert len(ids) == 4

    def test_topology_tokens(self, tokenizer):
        """All 16 topologies should have tokens."""
        expected = [
            "TOPO_BUCK", "TOPO_BOOST", "TOPO_BUCK_BOOST",
            "TOPO_CUK", "TOPO_SEPIC", "TOPO_FLYBACK", "TOPO_FORWARD",
            "TOPO_INVERTING_AMP", "TOPO_NONINVERTING_AMP",
            "TOPO_INSTRUMENTATION_AMP", "TOPO_DIFFERENTIAL_AMP",
            "TOPO_SALLEN_KEY_LP", "TOPO_SALLEN_KEY_HP",
            "TOPO_SALLEN_KEY_BP",
            "TOPO_WIEN_BRIDGE", "TOPO_COLPITTS",
        ]
        for name in expected:
            assert name in tokenizer.name_to_id, f"Missing topology token: {name}"

    def test_component_tokens(self, tokenizer):
        for name in ["COMP_RESISTOR", "COMP_CAPACITOR", "COMP_INDUCTOR",
                      "COMP_MOSFET_N", "COMP_DIODE", "COMP_TRANSFORMER"]:
            assert name in tokenizer.name_to_id

    def test_spec_tokens(self, tokenizer):
        for name in ["SPEC_VIN", "SPEC_VOUT", "SPEC_IOUT", "SPEC_FSW",
                      "SPEC_GAIN", "SPEC_CUTOFF_FREQ"]:
            assert name in tokenizer.name_to_id

    def test_value_bins_count(self, tokenizer):
        """Should have 500 value bins."""
        value_count = sum(1 for tok in tokenizer.tokens
                          if tok.token_type == TokenType.VALUE)
        assert value_count == 500


class TestValueEncoding:
    """Value encoding / decoding round-trip tests."""

    @pytest.mark.parametrize("value", [1.0, 10.0, 100.0, 1e-6, 1e6, 0.001, 47e-6])
    def test_round_trip_approximate(self, tokenizer, value):
        """Encoded then decoded value should be within 1 bin width."""
        token_id = tokenizer.encode_value(value)
        decoded = tokenizer.decode_value(token_id)
        # Log-scale binning: allow ~2% relative error
        ratio = decoded / value
        assert 0.9 < ratio < 1.1, f"Value {value} -> token {token_id} -> {decoded}, ratio={ratio:.4f}"

    def test_encode_value_clamps_extremes(self, tokenizer):
        """Very small / large values should map to boundary bins."""
        tiny = tokenizer.encode_value(1e-20)
        huge = tokenizer.encode_value(1e20)
        assert tiny != huge  # Different bins

    def test_encode_value_deterministic(self, tokenizer):
        """Same value always maps to same token."""
        for _ in range(10):
            assert tokenizer.encode_value(100e-6) == tokenizer.encode_value(100e-6)


class TestSequenceEncoding:
    """Test full sequence encode/decode."""

    def test_encode_component(self, tokenizer):
        cid = tokenizer.encode_component("RESISTOR")
        assert cid == tokenizer.name_to_id["COMP_RESISTOR"]

    def test_encode_spec(self, tokenizer):
        tokens = tokenizer.encode_spec("VIN", 12.0)
        assert len(tokens) == 2
        assert tokens[0] == tokenizer.name_to_id["SPEC_VIN"]

    def test_sequence_to_string(self, tokenizer, buck_token_ids):
        s = tokenizer.sequence_to_string(buck_token_ids)
        assert "START" in s
        assert "BUCK" in s
        assert "END" in s

    def test_decode_tokens(self, tokenizer, buck_token_ids):
        tokens = tokenizer.decode_tokens(buck_token_ids)
        assert len(tokens) == len(buck_token_ids)
        assert tokens[0].name == "START"
        assert tokens[-1].name == "END"


class TestSaveLoad:
    """Tokenizer serialization."""

    def test_save_and_load(self, tokenizer, tmp_path):
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        loaded = CircuitTokenizer.load(str(path))

        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.start_id == tokenizer.start_id
        assert loaded.sep_id == tokenizer.sep_id
        # Value round-trip preserved
        val = 47e-6
        assert loaded.encode_value(val) == tokenizer.encode_value(val)
