"""
Sequence encoding v2: Eulerian walk representation with embedded values.

Sequence format:
  <BOS>
    SPEC_V_IN VAL_xx SPEC_V_OUT VAL_xx SPEC_I_OUT VAL_xx
    SPEC_RIPPLE_PCT VAL_xx SPEC_EFF VAL_xx
  <SEP>
    [Eulerian walk: N_GND C_S1 N_SW C_L1 N_OUT C_C1 N_GND ...]
  <WALK_END>
    [Value block: C_L1 VAL_xx C_C1 VAL_xx C_RLOAD VAL_xx ...]
  <EOS>

The walk section captures topology (wiring) — which components connect
which nodes. The value section captures sizing — what values each
component has. The model generates both autoregressively.

Max sequence lengths:
  Buck/Boost/Buck-Boost: 11 + 13 + 12 + 2 = 38 tokens
  SEPIC/Cuk:             11 + 17 + 16 + 2 = 46 tokens
  Flyback:               11 + 15 + 12 + 2 = 40 tokens
  QR Flyback:            11 + 21 + 12 + 2 = 46 tokens
  → max_seq_len = 48 (padded to 48 or 64)
"""

from typing import Dict, List, Optional, Tuple

from ..data.spice_templates import Topology, SPICE_TEMPLATES, PARAM_NAMES
from ..data.circuit_graph import (
    get_circuit_graph, hierholzer_walk, walk_to_tokens,
    COMPONENT_TO_PARAMS, COMPONENT_VALUE_ORDER,
)
from .vocabulary_v2 import (
    BOS_ID, EOS_ID, SEP_ID, PAD_ID, WALK_END_ID,
    TOKEN_TO_ID_V2, ID_TO_TOKEN_V2,
    NODE_NAME_TO_TOKEN, COMP_NAME_TO_TOKEN,
    SPEC_KEY_TO_INFO_V2,
    PARAM_NAME_TO_QUANTITY,
    value_to_token_id_v2, token_id_to_value_v2,
    is_value_token_v2, is_node_token_v2, is_component_token_v2,
    NUM_VALUE_BINS, VALUE_BIN_OFFSET,
    QuantityType,
)

MAX_SEQ_LEN_V2 = 64

# Spec keys in fixed order
SPEC_ORDER = ['v_in', 'v_out', 'i_out', 'ripple_pct', 'efficiency']

# Param name → QuantityType (for value block encoding)
# Extends the base mapping with component-level param names
PARAM_QUANTITY_MAP = {
    'L': QuantityType.INDUCTANCE,
    'C': QuantityType.CAPACITANCE,
    'R_load': QuantityType.RESISTANCE,
    'V_in': QuantityType.VOLTAGE,
    'f_sw': QuantityType.FREQUENCY,
    'duty': QuantityType.DUTY,
    'L1': QuantityType.INDUCTANCE,
    'L2': QuantityType.INDUCTANCE,
    'C_couple': QuantityType.CAPACITANCE,
    'C_out': QuantityType.CAPACITANCE,
    'L_pri': QuantityType.INDUCTANCE,
    'n_ratio': QuantityType.TURNS_RATIO,
}


def circuit_to_tokens_v2(
    topology: Topology,
    params: Dict[str, float],
    specs: Dict[str, float],
    walk_seed: int = 42,
) -> List[int]:
    """
    Encode a circuit sample as a v2 token sequence with Eulerian walk.

    Args:
        topology: Topology enum
        params: Component parameter dict
        specs: Performance spec dict
        walk_seed: Seed for Eulerian walk randomization (augmentation)

    Returns:
        List of token IDs
    """
    tokens = [BOS_ID]

    # 1) Spec block
    for spec_key in SPEC_ORDER:
        spec_token_name, qtype = SPEC_KEY_TO_INFO_V2[spec_key]
        spec_token_id = TOKEN_TO_ID_V2[spec_token_name]
        val_token_id = value_to_token_id_v2(specs[spec_key], qtype)
        tokens.extend([spec_token_id, val_token_id])

    tokens.append(SEP_ID)

    # 2) Eulerian walk section
    graph = get_circuit_graph(topology)
    walk = hierholzer_walk(graph, seed=walk_seed)
    walk_str_tokens = walk_to_tokens(walk)

    for tok_str in walk_str_tokens:
        if tok_str in NODE_NAME_TO_TOKEN:
            token_name = NODE_NAME_TO_TOKEN[tok_str]
        elif tok_str in COMP_NAME_TO_TOKEN:
            token_name = COMP_NAME_TO_TOKEN[tok_str]
        else:
            raise ValueError(f"Unknown walk token: {tok_str}")
        tokens.append(TOKEN_TO_ID_V2[token_name])

    tokens.append(WALK_END_ID)

    # 3) Value block: component → param values
    comp_order = COMPONENT_VALUE_ORDER[topology]
    comp_params = COMPONENT_TO_PARAMS[topology]

    for comp_name in comp_order:
        comp_token_name = COMP_NAME_TO_TOKEN[comp_name]
        comp_token_id = TOKEN_TO_ID_V2[comp_token_name]
        param_names = comp_params[comp_name]

        for pname in param_names:
            qtype = PARAM_QUANTITY_MAP[pname]
            pname_str = str(pname)
            value = params.get(pname_str, params.get(pname, 0.0))
            val_token_id = value_to_token_id_v2(value, qtype)
            tokens.extend([comp_token_id, val_token_id])

    tokens.append(EOS_ID)
    return tokens


def tokens_to_circuit_v2(token_ids: List[int]) -> Optional[Dict]:
    """
    Decode a v2 token sequence back to topology, params, and specs.

    Returns:
        Dict with keys: topology, params, specs, walk_tokens — or None if invalid
    """
    # Strip padding
    ids = [t for t in token_ids if t != PAD_ID]

    if not ids or ids[0] != BOS_ID:
        return None

    # Find structural markers
    try:
        sep_pos = ids.index(SEP_ID)
    except ValueError:
        return None

    try:
        walk_end_pos = ids.index(WALK_END_ID)
    except ValueError:
        return None

    eos_pos = len(ids)
    if ids[-1] == EOS_ID:
        eos_pos = len(ids) - 1

    # 1) Parse specs
    spec_region = ids[1:sep_pos]
    specs = {}
    for i in range(0, len(spec_region) - 1, 2):
        token_name = ID_TO_TOKEN_V2.get(spec_region[i])
        val_id = spec_region[i + 1]
        if token_name is None or not is_value_token_v2(val_id):
            continue
        for spec_key, (stoken, qtype) in SPEC_KEY_TO_INFO_V2.items():
            if stoken == token_name:
                specs[spec_key] = token_id_to_value_v2(val_id, qtype)
                break

    # 2) Parse Eulerian walk → identify topology
    walk_region = ids[sep_pos + 1:walk_end_pos]
    walk_tokens = []
    for tid in walk_region:
        tname = ID_TO_TOKEN_V2.get(tid)
        if tname is not None:
            walk_tokens.append(tname)

    # Identify topology from components present in walk
    walk_components = set()
    for tname in walk_tokens:
        if tname.startswith('C_'):
            walk_components.add(tname)

    topology = _identify_topology(walk_components)
    if topology is None:
        return None

    # 3) Parse value block
    value_region = ids[walk_end_pos + 1:eos_pos]
    params = {}
    comp_params_map = COMPONENT_TO_PARAMS.get(topology, {})

    # Build reverse map: component token → list of param names
    i = 0
    while i < len(value_region) - 1:
        comp_tid = value_region[i]
        val_tid = value_region[i + 1]
        comp_name = ID_TO_TOKEN_V2.get(comp_tid)
        if comp_name is None or not is_value_token_v2(val_tid):
            i += 2
            continue

        # Find which graph component this token represents
        graph_comp = None
        for gc, ct in COMP_NAME_TO_TOKEN.items():
            if ct == comp_name:
                graph_comp = gc
                break

        if graph_comp and graph_comp in comp_params_map:
            param_names = comp_params_map[graph_comp]
            # Track which param to assign this value to
            # Count how many values we've already assigned to this component
            assigned_count = sum(1 for p in param_names if p in params)
            if assigned_count < len(param_names):
                pname = param_names[assigned_count]
                qtype = PARAM_QUANTITY_MAP.get(pname)
                if qtype:
                    params[pname] = token_id_to_value_v2(val_tid, qtype)

        i += 2

    return {
        'topology': topology,
        'params': params,
        'specs': specs,
        'walk_tokens': walk_tokens,
    }


def _identify_topology(walk_components: set) -> Optional[Topology]:
    """
    Identify which topology a set of component tokens belongs to.
    Uses distinctive components as fingerprints.
    """
    has = lambda t: t in walk_components

    # QR Flyback: has LR, CR, DBODY
    if has('C_LR') or has('C_CR') or has('C_DBODY'):
        return Topology.QR_FLYBACK

    # Flyback: has LPRI, LSEC but not LR
    if has('C_LPRI') or has('C_LSEC'):
        return Topology.FLYBACK

    # SEPIC: has CC (coupling cap) + L2 + CO + D1 going to output
    # Cuk: also has CC + L2 + CO
    # Distinguish: check if walk structure differs (both have same components)
    if has('C_CC') or has('C_CO'):
        # Both SEPIC and Cuk have CC, CO, L1, L2
        # For now, check if L2 is present (both have it)
        # We'll need topology-specific structure later; default to SEPIC
        # Actually: in our graph, SEPIC and Cuk have identical component sets
        # We need to use walk structure to distinguish (node connections)
        # For the simple case, return SEPIC (Cuk has D1 from GND→N2 vs SEPIC N2→OUT)
        # TODO: Use walk structure for disambiguation
        if has('C_L2'):
            return Topology.SEPIC  # Will be refined with walk structure

    # Buck/Boost/Buck-Boost: all have L1, C1, S1, D1, RLOAD, VIN
    # Distinguish by walk structure (node connections)
    if has('C_L1') and has('C_C1'):
        # Check: L1 connects SW→OUT = Buck, INP→SW = Boost, SW→GND = Buck-Boost
        # Without walk structure, default to Buck
        return Topology.BUCK  # Will be refined

    return None


def _identify_topology_from_walk(walk_tokens: List[str]) -> Optional[Topology]:
    """
    More precise topology identification using the full walk structure.
    Analyzes which nodes each component connects.
    """
    # Build component → (node_before, node_after) from walk
    comp_connections = {}
    for i, tok in enumerate(walk_tokens):
        if tok.startswith('C_'):
            node_before = walk_tokens[i - 1] if i > 0 else None
            node_after = walk_tokens[i + 1] if i < len(walk_tokens) - 1 else None
            if tok not in comp_connections:
                comp_connections[tok] = []
            comp_connections[tok].append((node_before, node_after))

    components = set(comp_connections.keys())

    # QR Flyback
    if 'C_LR' in components:
        return Topology.QR_FLYBACK

    # Flyback
    if 'C_LPRI' in components:
        return Topology.FLYBACK

    # SEPIC vs Cuk: both have CC, CO, L2
    if 'C_CC' in components:
        # Cuk: D1 connects GND↔N2; SEPIC: D1 connects N2↔OUT
        d1_conns = comp_connections.get('C_D1', [])
        for nb, na in d1_conns:
            if nb == 'N_GND' or na == 'N_GND':
                return Topology.CUK
            if nb == 'N_OUT' or na == 'N_OUT':
                return Topology.SEPIC
        return Topology.SEPIC  # default

    # Buck vs Boost vs Buck-Boost: all have L1, C1, S1, D1
    if 'C_L1' in components:
        l1_conns = comp_connections.get('C_L1', [])
        for nb, na in l1_conns:
            # Buck: L1 connects SW↔OUT
            if {nb, na} == {'N_SW', 'N_OUT'}:
                return Topology.BUCK
            # Boost: L1 connects INP↔SW
            if {nb, na} == {'N_INP', 'N_SW'}:
                return Topology.BOOST
            # Buck-Boost: L1 connects SW↔GND
            if {nb, na} == {'N_SW', 'N_GND'}:
                return Topology.BUCK_BOOST
        return Topology.BUCK  # default

    return None


def tokens_to_netlist_v2(token_ids: List[int]) -> Optional[str]:
    """Decode a v2 token sequence to a SPICE netlist string."""
    decoded = tokens_to_circuit_v2(token_ids)
    if decoded is None:
        return None

    topology = decoded['topology']
    params = decoded['params']

    # Refine topology using walk structure
    walk_tokens = decoded.get('walk_tokens', [])
    if walk_tokens:
        refined = _identify_topology_from_walk(walk_tokens)
        if refined is not None:
            topology = refined
            decoded['topology'] = topology

    template = SPICE_TEMPLATES.get(topology)
    if template is None:
        return None

    try:
        netlist = template.format(output_file="output.txt", **params)
        return netlist
    except (KeyError, ValueError):
        return None


def tokens_to_readable_v2(token_ids: List[int]) -> str:
    """Convert v2 token IDs to human-readable string."""
    return " ".join(ID_TO_TOKEN_V2.get(t, f"?{t}") for t in token_ids)
