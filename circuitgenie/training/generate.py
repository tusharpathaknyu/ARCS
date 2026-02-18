"""
Circuit generation from specs using a trained model.
"""

from typing import Dict, List, Optional

import torch

from ..data.spice_templates import Topology, PARAM_NAMES, PARAM_RANGES
from ..data.generator import CircuitSample
from ..model.transformer import CircuitGenieModel
from ..tokenizer.tokenizer import CircuitTokenizer
from ..tokenizer.vocabulary import (
    BOS_ID, SEP_ID, TOKEN_TO_ID,
    SPEC_KEY_TO_INFO,
    value_to_token_id,
)
from ..tokenizer.sequence import SPEC_ORDER, tokens_to_circuit, tokens_to_netlist


def build_spec_prefix(specs: Dict[str, float]) -> List[int]:
    """
    Build the spec prefix token sequence from a spec dict.

    Returns: [BOS, SPEC_V_IN, VAL_xx, SPEC_V_OUT, VAL_xx, ..., SEP]
    """
    tokens = [BOS_ID]
    for spec_key in SPEC_ORDER:
        spec_token_name, qtype = SPEC_KEY_TO_INFO[spec_key]
        spec_token_id = TOKEN_TO_ID[spec_token_name]
        val_token_id = value_to_token_id(specs[spec_key], qtype)
        tokens.extend([spec_token_id, val_token_id])
    tokens.append(SEP_ID)
    return tokens


def generate_circuit(
    model: CircuitGenieModel,
    tokenizer: CircuitTokenizer,
    specs: Dict[str, float],
    topology: Optional[Topology] = None,
    temperature: float = 0.8,
    top_k: int = 20,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Generate a circuit from performance specs.

    Args:
        model: Trained CircuitGenieModel
        tokenizer: CircuitTokenizer
        specs: Dict with keys: v_in, v_out, i_out, ripple_pct, efficiency
        topology: If provided, force this topology; otherwise let model predict
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: torch device

    Returns:
        Dict with keys: topology, params, specs, token_ids — or None if invalid
    """
    prefix = build_spec_prefix(specs)

    # Optionally force topology
    if topology is not None:
        from ..tokenizer.sequence import _TOPO_TO_TOKEN
        topo_token = TOKEN_TO_ID[_TOPO_TO_TOKEN[topology]]
        prefix.extend([topo_token, SEP_ID])

    prefix_tensor = torch.tensor([prefix], dtype=torch.long, device=device)

    # Generate
    max_new = 32 - len(prefix)
    generated = model.generate(
        prefix_tensor,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
    )

    token_ids = generated[0].tolist()

    # Decode
    decoded = tokens_to_circuit(token_ids)
    if decoded is not None:
        decoded['token_ids'] = token_ids
    return decoded


def generate_netlist(
    model: CircuitGenieModel,
    tokenizer: CircuitTokenizer,
    specs: Dict[str, float],
    topology: Optional[Topology] = None,
    temperature: float = 0.8,
    top_k: int = 20,
    device: str = "cpu",
) -> Optional[str]:
    """Generate a SPICE netlist string from specs."""
    result = generate_circuit(
        model, tokenizer, specs, topology, temperature, top_k, device
    )
    if result is None:
        return None
    return tokens_to_netlist(result['token_ids'])


def validate_circuit(decoded: Dict) -> Dict[str, bool]:
    """
    Validate a decoded circuit for structural correctness.

    Returns dict of check_name -> passed.
    """
    checks = {}

    topo = decoded.get('topology')
    params = decoded.get('params', {})
    specs = decoded.get('specs', {})

    # Check topology exists
    checks['has_topology'] = topo is not None

    if topo is not None:
        # Check all required params present
        expected_params = PARAM_NAMES[topo]
        checks['all_params_present'] = all(
            str(p) in params for p in expected_params
        )

        # Check params in valid ranges
        ranges = PARAM_RANGES[topo]
        all_in_range = True
        for pname, (lo, hi) in ranges.items():
            if pname in params:
                val = params[pname]
                # Allow 20% margin for binning quantization
                if val < lo * 0.8 or val > hi * 1.2:
                    all_in_range = False
        checks['params_in_range'] = all_in_range

    # Check specs present
    checks['has_specs'] = len(specs) >= 3

    # Check V_in consistency (spec V_in should match circuit V_in)
    if 'v_in' in specs and 'V_in' in params:
        ratio = specs['v_in'] / params['V_in'] if params['V_in'] > 0 else 0
        checks['v_in_consistent'] = 0.7 < ratio < 1.4  # Within binning tolerance

    return checks


def batch_generate(
    model: CircuitGenieModel,
    tokenizer: CircuitTokenizer,
    specs: Dict[str, float],
    n_samples: int = 10,
    topology: Optional[Topology] = None,
    temperature: float = 0.8,
    top_k: int = 20,
    device: str = "cpu",
) -> List[Dict]:
    """Generate multiple circuit candidates for the same specs."""
    results = []
    for _ in range(n_samples):
        result = generate_circuit(
            model, tokenizer, specs, topology, temperature, top_k, device
        )
        if result is not None:
            result['validation'] = validate_circuit(result)
            results.append(result)
    return results
