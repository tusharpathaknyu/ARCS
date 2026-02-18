"""
Token vocabulary v2 for CircuitGenie with Eulerian walk representation.

Vocabulary layout:
  - 5 special tokens (PAD, BOS, EOS, SEP, WALK_END)
  - 8 circuit node tokens (GND, INP, SW, OUT, N1, N2, PRI, SEC)
  - 15 component tokens (S1, D1, L1, C1, RLOAD, VIN, ...)
  - 5 spec name tokens
  - 128 value bin tokens

Total: 161 tokens

Sequence format:
  <BOS> [specs] <SEP> [Eulerian walk nodes+components] <WALK_END> [component values] <EOS>

Reuses value binning functions from vocabulary.py (v1).
"""

from typing import Dict, Tuple

# Import value binning from v1 (unchanged)
from .vocabulary import (
    QuantityType, QUANTITY_RANGES,
    value_to_bin, bin_to_value, value_to_token_id as _v1_val_to_tok,
    NUM_VALUE_BINS,
    PARAM_NAME_TO_QUANTITY,
)


# ============================================================
# Special tokens
# ============================================================
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
WALK_END_TOKEN = "<WALK_END>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
WALK_END_ID = 4

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, WALK_END_TOKEN]

# ============================================================
# Circuit node tokens (IDs 5-12)
# ============================================================
NODE_TOKENS = [
    "N_GND",   # Ground / VSS
    "N_INP",   # Input voltage node
    "N_SW",    # Switch node
    "N_OUT",   # Output node
    "N_N1",    # Internal node 1 (SEPIC, Cuk)
    "N_N2",    # Internal node 2 (SEPIC, Cuk)
    "N_PRI",   # Primary node (QR Flyback)
    "N_SEC",   # Secondary node (Flyback, QR Flyback)
]
NODE_OFFSET = len(SPECIAL_TOKENS)  # 5

# Map graph node names → token names
NODE_NAME_TO_TOKEN = {
    'GND': 'N_GND', 'INP': 'N_INP', 'SW': 'N_SW', 'OUT': 'N_OUT',
    'N1': 'N_N1', 'N2': 'N_N2', 'PRI': 'N_PRI', 'SEC': 'N_SEC',
}

# ============================================================
# Component tokens (IDs 13-27)
# ============================================================
COMPONENT_TOKENS = [
    "C_S1",     # Switch (MOSFET)
    "C_D1",     # Diode (main)
    "C_L1",     # Inductor (main)
    "C_C1",     # Capacitor (main output)
    "C_RLOAD",  # Load resistor
    "C_VIN",    # Input voltage source
    "C_L1IN",   # Input inductor (SEPIC renamed from L1)
    "C_L2",     # Output inductor (SEPIC, Cuk)
    "C_CC",     # Coupling capacitor (SEPIC, Cuk)
    "C_CO",     # Output capacitor (SEPIC, Cuk)
    "C_LPRI",   # Primary winding (Flyback)
    "C_LSEC",   # Secondary winding (Flyback)
    "C_LR",     # Resonant inductor (QR Flyback)
    "C_CR",     # Resonant capacitor (QR Flyback)
    "C_DBODY",  # Body diode (QR Flyback)
]
COMPONENT_OFFSET = NODE_OFFSET + len(NODE_TOKENS)  # 13

# Map graph component names → token names
COMP_NAME_TO_TOKEN = {
    'S1': 'C_S1', 'D1': 'C_D1', 'L1': 'C_L1', 'C1': 'C_C1',
    'RLOAD': 'C_RLOAD', 'VIN': 'C_VIN',
    'L2': 'C_L2', 'CC': 'C_CC', 'CO': 'C_CO',
    'LPRI': 'C_LPRI', 'LSEC': 'C_LSEC',
    'LR': 'C_LR', 'CR': 'C_CR', 'DBODY': 'C_DBODY',
}

# ============================================================
# Spec name tokens (IDs 28-32)
# ============================================================
SPEC_TOKENS = [
    "SPEC_V_IN",
    "SPEC_V_OUT",
    "SPEC_I_OUT",
    "SPEC_RIPPLE_PCT",
    "SPEC_EFF",
]
SPEC_OFFSET = COMPONENT_OFFSET + len(COMPONENT_TOKENS)  # 28

# ============================================================
# Value bin tokens (IDs 33-160)
# ============================================================
VALUE_BIN_OFFSET = SPEC_OFFSET + len(SPEC_TOKENS)  # 33
VALUE_TOKENS = [f"VAL_{i:03d}" for i in range(NUM_VALUE_BINS)]

# ============================================================
# Full vocabulary
# ============================================================
ALL_TOKENS_V2 = (
    SPECIAL_TOKENS
    + NODE_TOKENS
    + COMPONENT_TOKENS
    + SPEC_TOKENS
    + VALUE_TOKENS
)

TOKEN_TO_ID_V2 = {tok: i for i, tok in enumerate(ALL_TOKENS_V2)}
ID_TO_TOKEN_V2 = {i: tok for i, tok in enumerate(ALL_TOKENS_V2)}
VOCAB_SIZE_V2 = len(ALL_TOKENS_V2)

assert VOCAB_SIZE_V2 == 5 + 8 + 15 + 5 + 128  # = 161


# ============================================================
# Spec key mappings (same quantity types as v1)
# ============================================================
SPEC_KEY_TO_INFO_V2 = {
    'v_in':       ('SPEC_V_IN',       QuantityType.VOLTAGE),
    'v_out':      ('SPEC_V_OUT',      QuantityType.VOLTAGE),
    'i_out':      ('SPEC_I_OUT',      QuantityType.CURRENT),
    'ripple_pct': ('SPEC_RIPPLE_PCT', QuantityType.RIPPLE_PCT),
    'efficiency': ('SPEC_EFF',        QuantityType.EFFICIENCY),
}


# ============================================================
# Value binning functions (delegate to v1 math, adjust offsets)
# ============================================================
def value_to_token_id_v2(value: float, qtype: QuantityType) -> int:
    """Map a continuous value to its v2 token ID."""
    bin_idx = value_to_bin(value, qtype)
    return VALUE_BIN_OFFSET + bin_idx


def token_id_to_value_v2(token_id: int, qtype: QuantityType) -> float:
    """Map a v2 value token ID back to a continuous value."""
    bin_idx = token_id - VALUE_BIN_OFFSET
    return bin_to_value(bin_idx, qtype)


def is_value_token_v2(token_id: int) -> bool:
    """Check if a token ID is a value bin token in v2 vocabulary."""
    return VALUE_BIN_OFFSET <= token_id < VALUE_BIN_OFFSET + NUM_VALUE_BINS


def is_node_token_v2(token_id: int) -> bool:
    """Check if a token ID is a circuit node token."""
    return NODE_OFFSET <= token_id < NODE_OFFSET + len(NODE_TOKENS)


def is_component_token_v2(token_id: int) -> bool:
    """Check if a token ID is a component token."""
    return COMPONENT_OFFSET <= token_id < COMPONENT_OFFSET + len(COMPONENT_TOKENS)
