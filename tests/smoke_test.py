"""Quick smoke test: generate one Buck converter netlist, simulate it, tokenize."""

import sys
sys.path.insert(0, "src")

from arcs.templates import get_topology
from arcs.spice import NGSpiceRunner
from arcs.tokenizer import CircuitTokenizer
from arcs.datagen import compute_derived_metrics, is_valid_result
import numpy as np


def main():
    print("=" * 60)
    print("ARCS Smoke Test")
    print("=" * 60)

    # 1. Generate a Buck converter netlist with reasonable values
    topo = get_topology("buck")
    params = {
        "inductance": 22e-6,    # 22 µH
        "capacitance": 470e-6,  # 470 µF
        "esr": 0.02,            # 20 mΩ ESR
        "r_dson": 0.05,         # 50 mΩ Rds(on)
    }

    netlist = topo.generate_netlist(params)
    print("\n--- Generated Netlist ---")
    print(netlist[:500], "..." if len(netlist) > 500 else "")

    # 2. Simulate
    runner = NGSpiceRunner(timeout=30)
    print("\n--- Running ngspice simulation ---")
    result = runner.run(netlist)

    print(f"Success: {result.success}")
    print(f"Sim time: {result.sim_time_seconds:.2f}s")
    if result.error_message:
        print(f"Error: {result.error_message}")

    if result.metrics:
        print(f"Raw metrics: {result.metrics}")
        derived = compute_derived_metrics(result.metrics, topo.operating_conditions, "buck")
        print(f"Efficiency: {derived.get('efficiency', 0)*100:.1f}%")
        print(f"Vout avg: {derived.get('vout_avg', 0):.3f}V (target: {topo.operating_conditions['vout']}V)")
        print(f"Vout error: {derived.get('vout_error_pct', 0):.1f}%")
        print(f"Ripple: {derived.get('vout_ripple', 0)*1000:.1f}mV")
        print(f"Valid: {is_valid_result(derived, topo.operating_conditions)}")

    # 3. Tokenize
    print("\n--- Tokenizer Test ---")
    tokenizer = CircuitTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Encode a component
    r_id = tokenizer.encode_component("RESISTOR")
    print(f"RESISTOR token ID: {r_id} -> {tokenizer.tokens[r_id]}")

    # Encode a value
    v_id = tokenizer.encode_value(22e-6)
    print(f"22µH value token: {tokenizer.tokens[v_id]}")

    # Encode spec pair
    spec_tokens = tokenizer.encode_spec("vout", 5.0)
    print(f"SPEC_VOUT=5V tokens: {[tokenizer.tokens[t] for t in spec_tokens]}")

    # 4. Random parameter sampling
    print("\n--- Random Sampling Test ---")
    rng = np.random.default_rng(42)
    for i in range(3):
        params = topo.sample_parameters(rng)
        print(f"  Sample {i+1}: {params}")

    print("\n✅ All smoke tests passed!")


if __name__ == "__main__":
    main()
