"""Tests for sequence decoding (evaluate module)."""

import pytest
from arcs.evaluate import decode_generated_sequence, DecodedCircuit


class TestDecodeGeneratedSequence:
    """Test token sequence → DecodedCircuit parsing."""

    def test_buck_valid_structure(self, tokenizer, buck_token_ids):
        dec = decode_generated_sequence(buck_token_ids, tokenizer)
        assert isinstance(dec, DecodedCircuit)
        assert dec.topology == "buck"
        assert dec.valid_structure is True
        assert len(dec.components) >= 4

    def test_buck_specs(self, tokenizer, buck_token_ids):
        dec = decode_generated_sequence(buck_token_ids, tokenizer)
        assert "vin" in dec.specs or "VIN" in str(dec.specs).upper()

    def test_inverting_amp_valid(self, tokenizer, inverting_amp_token_ids):
        dec = decode_generated_sequence(inverting_amp_token_ids, tokenizer)
        assert dec.topology == "inverting_amp"
        assert dec.valid_structure is True
        assert len(dec.components) >= 2

    def test_empty_sequence(self, tokenizer):
        dec = decode_generated_sequence([], tokenizer)
        assert dec.valid_structure is False

    def test_only_start(self, tokenizer):
        dec = decode_generated_sequence([tokenizer.start_id], tokenizer)
        assert dec.valid_structure is False

    def test_missing_end(self, tokenizer):
        """Sequence without END token → still decodable but may be invalid."""
        ids = [
            tokenizer.name_to_id["START"],
            tokenizer.name_to_id["TOPO_BUCK"],
            tokenizer.sep_id,
        ]
        dec = decode_generated_sequence(ids, tokenizer)
        # Should decode but structure incomplete (no components)
        assert dec.topology == "buck"

    def test_garbage_tokens(self, tokenizer):
        """Random token ids should not crash."""
        import random
        random.seed(42)
        ids = [random.randint(0, tokenizer.vocab_size - 1) for _ in range(20)]
        dec = decode_generated_sequence(ids, tokenizer)
        assert isinstance(dec, DecodedCircuit)
