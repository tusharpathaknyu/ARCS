"""Smoke tests for arcs.demo — format helpers and CLI argument parsing."""
from __future__ import annotations

import pytest
from arcs.demo import _format_value, format_circuit
from arcs.evaluate import DecodedCircuit


class TestFormatValue:
    def test_micro(self):
        result = _format_value(1e-6)
        assert "u" in result.lower() or "μ" in result

    def test_kilo(self):
        result = _format_value(1e3)
        assert "k" in result.lower() or "1000" in result

    def test_milli(self):
        result = _format_value(0.001)
        assert "m" in result.lower()

    def test_plain(self):
        result = _format_value(1.0)
        assert isinstance(result, str)


class TestFormatCircuit:
    def test_format_valid_circuit(self):
        decoded = DecodedCircuit(
            topology="buck",
            specs={"vin": 12.0, "vout": 5.0},
            components=[
                ("capacitor", 38e-6),
                ("resistor", 0.13),
                ("mosfet_n", 0.021),
                ("inductor", 300e-6),
            ],
            raw_tokens=[],
            valid_structure=True,
        )
        output = format_circuit(decoded)
        assert "buck" in output.lower()
        assert "VALID" in output

    def test_format_invalid_circuit(self):
        decoded = DecodedCircuit(
            topology="unknown",
            specs={},
            components=[],
            raw_tokens=[],
            valid_structure=False,
        )
        output = format_circuit(decoded)
        assert "INVALID" in output
