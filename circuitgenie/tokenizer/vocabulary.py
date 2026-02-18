"""
Token vocabulary and value binning for CircuitGenie.

Vocabulary: 157 tokens total
  - 4 special tokens (PAD, BOS, EOS, SEP)
  - 7 topology tokens
  - 13 parameter name tokens
  - 5 spec name tokens
  - 128 value bin tokens (shared, context from preceding name token)
"""

import math
from enum import Enum
from typing import Tuple

import numpy as np

# ============================================================
# Special tokens
# ============================================================
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3

# ============================================================
# Topology tokens (IDs 4-10)
# ============================================================
TOPOLOGY_TOKENS = [
    "TOPO_BUCK",
    "TOPO_BOOST",
    "TOPO_BUCK_BOOST",
    "TOPO_SEPIC",
    "TOPO_CUK",
    "TOPO_FLYBACK",
    "TOPO_QR_FLYBACK",
]
TOPO_OFFSET = 4

# ============================================================
# Parameter name tokens (IDs 11-23)
# ============================================================
PARAM_TOKENS = [
    "PARAM_L",
    "PARAM_C",
    "PARAM_R_LOAD",
    "PARAM_V_IN",
    "PARAM_F_SW",
    "PARAM_DUTY",
    "PARAM_L1",
    "PARAM_L2",
    "PARAM_L_PRI",
    "PARAM_C_COUPLE",
    "PARAM_C_OUT",
    "PARAM_N_RATIO",
    "PARAM_V_OUT",
]
PARAM_OFFSET = 11

# ============================================================
# Spec name tokens (IDs 24-28)
# ============================================================
SPEC_TOKENS = [
    "SPEC_V_IN",
    "SPEC_V_OUT",
    "SPEC_I_OUT",
    "SPEC_RIPPLE_PCT",
    "SPEC_EFF",
]
SPEC_OFFSET = 24

# ============================================================
# Value bin tokens (IDs 29-156)
# ============================================================
NUM_VALUE_BINS = 128
VALUE_BIN_OFFSET = 29
VALUE_TOKENS = [f"VAL_{i:03d}" for i in range(NUM_VALUE_BINS)]

# ============================================================
# Full vocabulary
# ============================================================
ALL_TOKENS = (
    [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
    + TOPOLOGY_TOKENS
    + PARAM_TOKENS
    + SPEC_TOKENS
    + VALUE_TOKENS
)

TOKEN_TO_ID = {tok: i for i, tok in enumerate(ALL_TOKENS)}
ID_TO_TOKEN = {i: tok for i, tok in enumerate(ALL_TOKENS)}
VOCAB_SIZE = len(ALL_TOKENS)  # 157

assert VOCAB_SIZE == 4 + 7 + 13 + 5 + 128  # = 157


# ============================================================
# Quantity types and their ranges for value binning
# ============================================================
class QuantityType(Enum):
    INDUCTANCE = "inductance"      # Henries
    CAPACITANCE = "capacitance"    # Farads
    RESISTANCE = "resistance"      # Ohms
    VOLTAGE = "voltage"            # Volts
    FREQUENCY = "frequency"        # Hz
    DUTY = "duty"                  # 0-1 ratio
    TURNS_RATIO = "turns_ratio"    # dimensionless
    RIPPLE_PCT = "ripple_pct"      # percentage
    EFFICIENCY = "efficiency"      # 0-1 ratio
    CURRENT = "current"            # Amps


# (min, max, is_log_scale)
QUANTITY_RANGES: dict[QuantityType, Tuple[float, float, bool]] = {
    QuantityType.INDUCTANCE:  (5e-6, 2e-3, True),
    QuantityType.CAPACITANCE: (5e-7, 2e-3, True),
    QuantityType.RESISTANCE:  (1.0, 200.0, True),
    QuantityType.VOLTAGE:     (0.1, 1500.0, True),
    QuantityType.FREQUENCY:   (20e3, 1e6, True),
    QuantityType.DUTY:        (0.05, 0.95, False),
    QuantityType.TURNS_RATIO: (0.05, 5.0, True),
    QuantityType.RIPPLE_PCT:  (0.001, 100.0, True),
    QuantityType.EFFICIENCY:  (0.50, 1.0, False),
    QuantityType.CURRENT:     (0.001, 200.0, True),
}


# Maps circuit parameter names -> quantity type
PARAM_NAME_TO_QUANTITY = {
    'L':        QuantityType.INDUCTANCE,
    'C':        QuantityType.CAPACITANCE,
    'R_load':   QuantityType.RESISTANCE,
    'V_in':     QuantityType.VOLTAGE,
    'f_sw':     QuantityType.FREQUENCY,
    'duty':     QuantityType.DUTY,
    'L1':       QuantityType.INDUCTANCE,
    'L2':       QuantityType.INDUCTANCE,
    'L_pri':    QuantityType.INDUCTANCE,
    'C_couple': QuantityType.CAPACITANCE,
    'C_out':    QuantityType.CAPACITANCE,
    'n_ratio':  QuantityType.TURNS_RATIO,
    'V_out':    QuantityType.VOLTAGE,
}

# Maps circuit parameter names -> PARAM token name
PARAM_NAME_TO_TOKEN = {
    'L':        'PARAM_L',
    'C':        'PARAM_C',
    'R_load':   'PARAM_R_LOAD',
    'V_in':     'PARAM_V_IN',
    'f_sw':     'PARAM_F_SW',
    'duty':     'PARAM_DUTY',
    'L1':       'PARAM_L1',
    'L2':       'PARAM_L2',
    'L_pri':    'PARAM_L_PRI',
    'C_couple': 'PARAM_C_COUPLE',
    'C_out':    'PARAM_C_OUT',
    'n_ratio':  'PARAM_N_RATIO',
    'V_out':    'PARAM_V_OUT',
}

# Maps spec keys -> (SPEC token name, quantity type)
SPEC_KEY_TO_INFO = {
    'v_in':        ('SPEC_V_IN',        QuantityType.VOLTAGE),
    'v_out':       ('SPEC_V_OUT',       QuantityType.VOLTAGE),
    'i_out':       ('SPEC_I_OUT',       QuantityType.CURRENT),
    'ripple_pct':  ('SPEC_RIPPLE_PCT',  QuantityType.RIPPLE_PCT),
    'efficiency':  ('SPEC_EFF',         QuantityType.EFFICIENCY),
}


# ============================================================
# Value binning functions
# ============================================================
def value_to_bin(value: float, qtype: QuantityType) -> int:
    """Map a continuous value to a bin index [0, 127]."""
    vmin, vmax, is_log = QUANTITY_RANGES[qtype]

    # Clamp
    value = max(vmin, min(vmax, value))

    if is_log:
        if value <= 0:
            return 0
        normalized = (math.log(value) - math.log(vmin)) / (math.log(vmax) - math.log(vmin))
    else:
        normalized = (value - vmin) / (vmax - vmin)

    bin_idx = int(normalized * (NUM_VALUE_BINS - 1))
    return max(0, min(NUM_VALUE_BINS - 1, bin_idx))


def bin_to_value(bin_idx: int, qtype: QuantityType) -> float:
    """Map a bin index [0, 127] back to a continuous value (bin center)."""
    vmin, vmax, is_log = QUANTITY_RANGES[qtype]

    bin_idx = max(0, min(NUM_VALUE_BINS - 1, bin_idx))
    normalized = (bin_idx + 0.5) / NUM_VALUE_BINS  # bin center

    if is_log:
        value = math.exp(math.log(vmin) + normalized * (math.log(vmax) - math.log(vmin)))
    else:
        value = vmin + normalized * (vmax - vmin)

    return value


def value_to_token_id(value: float, qtype: QuantityType) -> int:
    """Map a continuous value to its token ID."""
    return VALUE_BIN_OFFSET + value_to_bin(value, qtype)


def token_id_to_value(token_id: int, qtype: QuantityType) -> float:
    """Map a value token ID back to a continuous value."""
    bin_idx = token_id - VALUE_BIN_OFFSET
    return bin_to_value(bin_idx, qtype)


def is_value_token(token_id: int) -> bool:
    """Check if a token ID is a value bin token."""
    return VALUE_BIN_OFFSET <= token_id < VALUE_BIN_OFFSET + NUM_VALUE_BINS
