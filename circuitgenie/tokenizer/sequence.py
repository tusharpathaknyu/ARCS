"""
Convert between CircuitSample and token sequences.

Sequence format:
  <BOS> SPEC_V_IN VAL_xx SPEC_V_OUT VAL_xx SPEC_I_OUT VAL_xx
  SPEC_RIPPLE_PCT VAL_xx SPEC_EFF VAL_xx <SEP>
  TOPO_xxx <SEP>
  PARAM_x VAL_xx PARAM_x VAL_xx ... <EOS>

Lengths:
  Buck/Boost/Buck-Boost: 27 tokens (6 params)
  Flyback/QR Flyback:    29 tokens (7 params, includes n_ratio)
  SEPIC/Cuk:             31 tokens (8 params)
  Max sequence length:   32 (padded)
"""

from typing import Dict, List, Optional

from ..data.spice_templates import Topology, PARAM_NAMES, SPICE_TEMPLATES
from .vocabulary import (
    BOS_ID, EOS_ID, SEP_ID, PAD_ID,
    TOPO_OFFSET, PARAM_OFFSET, SPEC_OFFSET, VALUE_BIN_OFFSET,
    TOKEN_TO_ID, ID_TO_TOKEN,
    TOPOLOGY_TOKENS, PARAM_TOKENS, SPEC_TOKENS,
    PARAM_NAME_TO_TOKEN, PARAM_NAME_TO_QUANTITY,
    SPEC_KEY_TO_INFO,
    value_to_token_id, token_id_to_value, is_value_token,
    NUM_VALUE_BINS, VOCAB_SIZE,
    QuantityType,
)

MAX_SEQ_LEN = 32

# Spec keys in fixed order
SPEC_ORDER = ['v_in', 'v_out', 'i_out', 'ripple_pct', 'efficiency']

# Topology enum value -> token string
_TOPO_TO_TOKEN = {
    Topology.BUCK: "TOPO_BUCK",
    Topology.BOOST: "TOPO_BOOST",
    Topology.BUCK_BOOST: "TOPO_BUCK_BOOST",
    Topology.SEPIC: "TOPO_SEPIC",
    Topology.CUK: "TOPO_CUK",
    Topology.FLYBACK: "TOPO_FLYBACK",
    Topology.QR_FLYBACK: "TOPO_QR_FLYBACK",
}

_TOKEN_TO_TOPO = {v: k for k, v in _TOPO_TO_TOKEN.items()}


def circuit_to_tokens(
    topology: Topology,
    params: Dict[str, float],
    specs: Dict[str, float],
) -> List[int]:
    """
    Encode a circuit sample as a token sequence.

    Args:
        topology: Topology enum
        params: Component parameter dict (topology-specific keys)
        specs: Performance spec dict with keys: v_in, v_out, i_out, ripple_pct, efficiency

    Returns:
        List of token IDs
    """
    tokens = [BOS_ID]

    # 1) Spec block: 5 spec pairs
    for spec_key in SPEC_ORDER:
        spec_token_name, qtype = SPEC_KEY_TO_INFO[spec_key]
        spec_token_id = TOKEN_TO_ID[spec_token_name]
        val_token_id = value_to_token_id(specs[spec_key], qtype)
        tokens.extend([spec_token_id, val_token_id])

    tokens.append(SEP_ID)

    # 2) Topology token
    topo_token_name = _TOPO_TO_TOKEN[topology]
    tokens.append(TOKEN_TO_ID[topo_token_name])

    tokens.append(SEP_ID)

    # 3) Component parameters in canonical order for this topology
    param_names = PARAM_NAMES[topology]
    for pname in param_names:
        pname_str = str(pname)  # handle np.str_
        param_token_name = PARAM_NAME_TO_TOKEN[pname_str]
        qtype = PARAM_NAME_TO_QUANTITY[pname_str]
        param_token_id = TOKEN_TO_ID[param_token_name]
        val_token_id = value_to_token_id(params[pname], qtype)
        tokens.extend([param_token_id, val_token_id])

    tokens.append(EOS_ID)
    return tokens


def tokens_to_circuit(
    token_ids: List[int],
) -> Optional[Dict]:
    """
    Decode a token sequence back to topology, params, and specs.

    Returns:
        Dict with keys: topology, params, specs — or None if invalid
    """
    # Strip padding
    ids = [t for t in token_ids if t != PAD_ID]

    # Validate BOS
    if not ids or ids[0] != BOS_ID:
        return None

    # Find SEP positions
    sep_positions = [i for i, t in enumerate(ids) if t == SEP_ID]
    if len(sep_positions) < 2:
        return None

    # Parse specs (between BOS and first SEP)
    spec_region = ids[1:sep_positions[0]]
    specs = {}
    for i in range(0, len(spec_region) - 1, 2):
        token_name = ID_TO_TOKEN.get(spec_region[i])
        val_id = spec_region[i + 1]
        if token_name is None or not is_value_token(val_id):
            continue
        # Find which spec key this is
        for spec_key, (stoken, qtype) in SPEC_KEY_TO_INFO.items():
            if stoken == token_name:
                specs[spec_key] = token_id_to_value(val_id, qtype)
                break

    # Parse topology (between first and second SEP)
    topo_region = ids[sep_positions[0] + 1:sep_positions[1]]
    if len(topo_region) != 1:
        return None
    topo_token_name = ID_TO_TOKEN.get(topo_region[0])
    topology = _TOKEN_TO_TOPO.get(topo_token_name)
    if topology is None:
        return None

    # Parse component params (between second SEP and EOS)
    eos_pos = len(ids)
    if ids[-1] == EOS_ID:
        eos_pos = len(ids) - 1
    param_region = ids[sep_positions[1] + 1:eos_pos]

    params = {}
    # Track which param name token we're looking at to determine quantity type
    for i in range(0, len(param_region) - 1, 2):
        token_name = ID_TO_TOKEN.get(param_region[i])
        val_id = param_region[i + 1]
        if token_name is None or not is_value_token(val_id):
            continue
        # Find the param name and quantity type
        for pname, ptok in PARAM_NAME_TO_TOKEN.items():
            if ptok == token_name:
                qtype = PARAM_NAME_TO_QUANTITY[pname]
                params[pname] = token_id_to_value(val_id, qtype)
                break

    return {
        'topology': topology,
        'params': params,
        'specs': specs,
    }


def tokens_to_netlist(token_ids: List[int]) -> Optional[str]:
    """
    Decode a token sequence to a SPICE netlist string.

    Returns:
        SPICE netlist string or None if decoding fails
    """
    decoded = tokens_to_circuit(token_ids)
    if decoded is None:
        return None

    topology = decoded['topology']
    params = decoded['params']
    template = SPICE_TEMPLATES[topology]

    try:
        # Format template with params (output_file placeholder)
        netlist = template.format(output_file="output.txt", **params)
        return netlist
    except (KeyError, ValueError):
        return None


def tokens_to_readable(token_ids: List[int]) -> str:
    """Convert token IDs to human-readable string."""
    return " ".join(ID_TO_TOKEN.get(t, f"?{t}") for t in token_ids)
