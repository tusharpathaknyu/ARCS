"""ARCS Tokenizer: Native circuit component vocabulary.

Unlike text-based approaches that tokenize SPICE netlists as character sequences,
ARCS uses a structured vocabulary where each token represents a meaningful circuit
element, connection, or specification.

Token categories:
  - Component tokens (MOSFET, RESISTOR, CAPACITOR, etc.)
  - Value tokens (discretized on log scale or E-series)
  - Pin tokens (drain, gate, source, positive, negative, etc.)
  - Connection tokens (net assignments)
  - Spec tokens (target performance values)
  - Special tokens (START, END, PAD, SEP)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np


class TokenType(Enum):
    """Categories of tokens in the circuit vocabulary."""

    SPECIAL = auto()       # START, END, PAD, SEP, INVALID
    COMPONENT = auto()     # MOSFET_N, RESISTOR, CAPACITOR, etc.
    VALUE = auto()         # Discretized component values
    PIN = auto()           # _D, _G, _S, _P, _N, etc.
    CONNECTION = auto()    # NET_0, NET_1, ... CONNECT
    SPEC = auto()          # SPEC_VIN, SPEC_VOUT, SPEC_IOUT, etc.
    TOPOLOGY = auto()      # TOPO_BUCK, TOPO_BOOST, etc.


@dataclass
class Token:
    """A single token in the circuit vocabulary."""

    id: int
    name: str
    token_type: TokenType
    value: Optional[float] = None  # For VALUE tokens, the numeric value
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        if self.value is not None:
            return f"Token({self.name}={self.value:.4g})"
        return f"Token({self.name})"


class CircuitTokenizer:
    """Tokenizer with native circuit vocabulary.

    Vocabulary structure:
        [0-4]       Special tokens: PAD, START, END, SEP, INVALID
        [5-24]      Component type tokens
        [25-34]     Topology tokens
        [35-54]     Spec tokens
        [55-74]     Pin tokens
        [75-174]    Net/connection tokens (100 nets)
        [175-674]   Value tokens (500 bins, log-discretized)
    """

    # Value discretization: log-scale bins covering 1e-12 to 1e6
    VALUE_MIN = 1e-12
    VALUE_MAX = 1e6
    N_VALUE_BINS = 500

    def __init__(self):
        self.tokens: list[Token] = []
        self.name_to_id: dict[str, int] = {}
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Construct the complete token vocabulary."""
        idx = 0

        # --- Special tokens ---
        for name in ["PAD", "START", "END", "SEP", "INVALID"]:
            self._add_token(idx, name, TokenType.SPECIAL)
            idx += 1

        # --- Component type tokens ---
        components = [
            "RESISTOR", "CAPACITOR", "INDUCTOR",
            "DIODE", "DIODE_SCHOTTKY", "DIODE_ZENER",
            "MOSFET_N", "MOSFET_P",
            "BJT_NPN", "BJT_PNP",
            "OPAMP",
            "TRANSFORMER",
            "VOLTAGE_SOURCE", "CURRENT_SOURCE",
            "SWITCH_IDEAL",
            "IC_GENERIC",
        ]
        for name in components:
            self._add_token(idx, f"COMP_{name}", TokenType.COMPONENT)
            idx += 1

        # Pad to component block
        while idx < 25:
            self._add_token(idx, f"COMP_RESERVED_{idx}", TokenType.COMPONENT)
            idx += 1

        # --- Topology tokens ---
        topologies = [
            # Tier 1 — power converters
            "BUCK", "BOOST", "BUCK_BOOST", "CUK", "SEPIC", "FLYBACK", "FORWARD",
            "LLC", "PFC", "INVERTER",
            # Tier 2 — signal processing
            "INVERTING_AMP", "NONINVERTING_AMP", "INSTRUMENTATION_AMP",
            "DIFFERENTIAL_AMP",
            "SALLEN_KEY_LP", "SALLEN_KEY_HP", "SALLEN_KEY_BP",
            "WIEN_BRIDGE", "COLPITTS",
        ]
        for name in topologies:
            self._add_token(idx, f"TOPO_{name}", TokenType.TOPOLOGY)
            idx += 1

        while idx < 45:
            self._add_token(idx, f"TOPO_RESERVED_{idx}", TokenType.TOPOLOGY)
            idx += 1

        # --- Spec tokens ---
        specs = [
            "SPEC_VIN", "SPEC_VOUT", "SPEC_IOUT",
            "SPEC_EFFICIENCY", "SPEC_RIPPLE", "SPEC_RIPPLE_RATIO",
            "SPEC_VOUT_ERROR",
            "SPEC_GAIN", "SPEC_BANDWIDTH", "SPEC_PHASE_MARGIN",
            "SPEC_CMRR", "SPEC_THD", "SPEC_SNR",
            "SPEC_NOISE_FIGURE", "SPEC_POWER",
            "SPEC_FSW",  # Switching frequency
            "SPEC_CUTOFF_FREQ",  # Filter -3dB frequency
            "SPEC_CENTER_FREQ",  # Band-pass center frequency
            "SPEC_Q_FACTOR",     # Filter quality factor
            "SPEC_OSC_FREQ",     # Oscillator frequency
        ]
        for name in specs:
            self._add_token(idx, name, TokenType.SPEC)
            idx += 1

        while idx < 65:
            self._add_token(idx, f"SPEC_RESERVED_{idx}", TokenType.SPEC)
            idx += 1

        # --- Pin tokens ---
        pins = [
            # MOSFET pins
            "PIN_DRAIN", "PIN_GATE", "PIN_SOURCE",
            # Passive pins
            "PIN_POS", "PIN_NEG",
            # BJT pins
            "PIN_COLLECTOR", "PIN_BASE", "PIN_EMITTER",
            # Op-amp pins
            "PIN_INP", "PIN_INN", "PIN_OUT",
            # Power pins
            "PIN_VCC", "PIN_VDD", "PIN_GND", "PIN_VSS",
            # Transformer pins
            "PIN_PRI_DOT", "PIN_PRI_END", "PIN_SEC_DOT", "PIN_SEC_END",
            # Generic
            "PIN_A", "PIN_B",
        ]
        for name in pins:
            self._add_token(idx, name, TokenType.PIN)
            idx += 1

        while idx < 85:
            self._add_token(idx, f"PIN_RESERVED_{idx}", TokenType.PIN)
            idx += 1

        # --- Net/Connection tokens ---
        for net_id in range(100):
            self._add_token(idx, f"NET_{net_id}", TokenType.CONNECTION)
            idx += 1

        # --- Value tokens (log-discretized) ---
        log_min = math.log10(self.VALUE_MIN)
        log_max = math.log10(self.VALUE_MAX)
        bin_edges = np.linspace(log_min, log_max, self.N_VALUE_BINS + 1)

        for i in range(self.N_VALUE_BINS):
            center = 10 ** ((bin_edges[i] + bin_edges[i + 1]) / 2)
            self._add_token(
                idx,
                f"VAL_{i}",
                TokenType.VALUE,
                value=center,
            )
            idx += 1

        self.vocab_size = idx

    def _add_token(
        self, idx: int, name: str, token_type: TokenType, value: float | None = None
    ) -> None:
        token = Token(id=idx, name=name, token_type=token_type, value=value)
        self.tokens.append(token)
        self.name_to_id[name] = idx

    @property
    def pad_id(self) -> int:
        return self.name_to_id["PAD"]

    @property
    def start_id(self) -> int:
        return self.name_to_id["START"]

    @property
    def end_id(self) -> int:
        return self.name_to_id["END"]

    @property
    def sep_id(self) -> int:
        return self.name_to_id["SEP"]

    def encode_value(self, value: float) -> int:
        """Map a continuous value to the nearest value bin token ID."""
        if value <= 0:
            return self.name_to_id["VAL_0"]

        log_val = math.log10(value)
        log_min = math.log10(self.VALUE_MIN)
        log_max = math.log10(self.VALUE_MAX)

        # Clamp to range
        log_val = max(log_min, min(log_max, log_val))

        # Find bin index
        bin_idx = int((log_val - log_min) / (log_max - log_min) * self.N_VALUE_BINS)
        bin_idx = min(bin_idx, self.N_VALUE_BINS - 1)

        return self.name_to_id[f"VAL_{bin_idx}"]

    def decode_value(self, token_id: int) -> float:
        """Get the continuous value represented by a value token."""
        token = self.tokens[token_id]
        if token.token_type != TokenType.VALUE or token.value is None:
            raise ValueError(f"Token {token.name} is not a value token")
        return token.value

    def encode_component(self, component_type: str) -> int:
        """Get token ID for a component type."""
        key = f"COMP_{component_type.upper()}"
        if key not in self.name_to_id:
            raise ValueError(f"Unknown component: {component_type}. Available: "
                           f"{[t.name for t in self.tokens if t.token_type == TokenType.COMPONENT]}")
        return self.name_to_id[key]

    def encode_spec(self, spec_name: str, spec_value: float) -> list[int]:
        """Encode a spec as [SPEC_NAME, VALUE_BIN] token pair."""
        spec_key = f"SPEC_{spec_name.upper()}"
        if spec_key not in self.name_to_id:
            raise ValueError(f"Unknown spec: {spec_name}")
        return [self.name_to_id[spec_key], self.encode_value(spec_value)]

    def encode_circuit_sample(self, sample) -> list[int]:
        """Encode a CircuitSample into a token sequence.

        Format:
            START, TOPO_X, SEP,
            SPEC_<key>, val, ..., SEP,
            COMP_X, VAL, COMP_Y, VAL, ...,
            END

        Supports both Tier 1 (power converter) and Tier 2 (signal) topologies.
        """
        from arcs.datagen import CircuitSample  # Avoid circular import

        tokens = [self.start_id]

        # Topology token
        topo_key = f"TOPO_{sample.topology.upper()}"
        if topo_key in self.name_to_id:
            tokens.append(self.name_to_id[topo_key])
        tokens.append(self.sep_id)

        # Spec tokens — build map from operating conditions + key metrics
        oc = sample.operating_conditions

        # Common OC → spec-token mapping
        _OC_SPEC = {
            "vin": "SPEC_VIN", "vout": "SPEC_VOUT", "iout": "SPEC_IOUT",
            "fsw": "SPEC_FSW",
            "vin_amp": "SPEC_VIN", "freq_test": "SPEC_CUTOFF_FREQ",
            "vcc": "SPEC_VIN",
        }
        spec_map: dict[str, float | None] = {}
        for oc_key, oc_val in oc.items():
            spec_tok = _OC_SPEC.get(oc_key)
            if spec_tok and spec_tok not in spec_map:
                spec_map[spec_tok] = oc_val

        # Key metrics (add if valid)
        if sample.valid:
            m = sample.metrics
            _METRIC_SPEC = {
                "efficiency": "SPEC_EFFICIENCY",
                "vout_ripple": "SPEC_RIPPLE",
                "gain_db": "SPEC_GAIN",
                "bw_3db": "SPEC_BANDWIDTH",
                "fc_3db": "SPEC_CUTOFF_FREQ",
                "phase_rad": "SPEC_PHASE_MARGIN",  # VP() returns radians
                "f_peak": "SPEC_CENTER_FREQ",
                "vosc_pp": "SPEC_VIN",   # oscillator amplitude → reuse VIN slot
            }
            for mk, sk in _METRIC_SPEC.items():
                val = m.get(mk)
                if val is not None and sk not in spec_map:
                    spec_map[sk] = val

        for spec_key, spec_val in spec_map.items():
            if spec_val is not None and spec_key in self.name_to_id:
                tokens.append(self.name_to_id[spec_key])
                tokens.append(self.encode_value(abs(spec_val)))
        tokens.append(self.sep_id)

        # Component tokens: each component as (TYPE, VALUE) pair
        component_mapping = self._params_to_components(sample.topology, sample.parameters)
        for comp_type, comp_value in component_mapping:
            comp_key = f"COMP_{comp_type.upper()}"
            if comp_key in self.name_to_id:
                tokens.append(self.name_to_id[comp_key])
                tokens.append(self.encode_value(comp_value))

        tokens.append(self.end_id)
        return tokens

    def _params_to_components(
        self, topology: str, params: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Map topology parameters to (component_type, value) pairs."""
        components = []

        param_to_comp = {
            # Tier 1 — power converter components
            "inductance": ("INDUCTOR", None),
            "inductance_1": ("INDUCTOR", None),
            "inductance_2": ("INDUCTOR", None),
            "inductance_primary": ("INDUCTOR", None),
            "inductance_output": ("INDUCTOR", None),
            "capacitance": ("CAPACITOR", None),
            "cap_coupling": ("CAPACITOR", None),
            "esr": ("RESISTOR", None),
            "r_dson": ("MOSFET_N", None),
            "r_load": ("RESISTOR", None),
            "turns_ratio": ("TRANSFORMER", None),
            # Tier 2 — resistors
            "r_input": ("RESISTOR", None),
            "r_feedback": ("RESISTOR", None),
            "r_ground": ("RESISTOR", None),
            "r_gain": ("RESISTOR", None),
            "r1": ("RESISTOR", None),
            "r2": ("RESISTOR", None),
            "r3": ("RESISTOR", None),
            "r_freq": ("RESISTOR", None),
            "r_bias_1": ("RESISTOR", None),
            "r_bias_2": ("RESISTOR", None),
            "r_emitter": ("RESISTOR", None),
            "r_collector": ("RESISTOR", None),
            # Tier 2 — capacitors
            "c1": ("CAPACITOR", None),
            "c2": ("CAPACITOR", None),
            "c_freq": ("CAPACITOR", None),
        }

        for param_name, param_value in params.items():
            if param_name in param_to_comp:
                comp_type = param_to_comp[param_name][0]
                components.append((comp_type, param_value))

        return components

    def decode_tokens(self, token_ids: list[int]) -> list[Token]:
        """Convert token IDs back to Token objects."""
        return [self.tokens[tid] for tid in token_ids if 0 <= tid < len(self.tokens)]

    def sequence_to_string(self, token_ids: list[int]) -> str:
        """Human-readable string representation of a token sequence."""
        parts = []
        for tid in token_ids:
            token = self.tokens[tid]
            if token.token_type == TokenType.VALUE and token.value is not None:
                parts.append(f"{token.value:.3g}")
            else:
                parts.append(token.name)
        return " → ".join(parts)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "tokens": [
                {
                    "id": t.id,
                    "name": t.name,
                    "type": t.token_type.name,
                    "value": t.value,
                }
                for t in self.tokens
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CircuitTokenizer":
        """Load vocabulary from JSON."""
        tokenizer = cls.__new__(cls)
        tokenizer.tokens = []
        tokenizer.name_to_id = {}

        with open(path) as f:
            data = json.load(f)

        tokenizer.vocab_size = data["vocab_size"]
        for td in data["tokens"]:
            token = Token(
                id=td["id"],
                name=td["name"],
                token_type=TokenType[td["type"]],
                value=td.get("value"),
            )
            tokenizer.tokens.append(token)
            tokenizer.name_to_id[token.name] = token.id

        return tokenizer
