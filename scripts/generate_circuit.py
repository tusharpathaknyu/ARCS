#!/usr/bin/env python3
"""Generate circuits from specs using a trained CircuitGenie model."""

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from circuitgenie.data.spice_templates import Topology
from circuitgenie.model.config import CircuitGenieConfig
from circuitgenie.model.transformer import CircuitGenieModel
from circuitgenie.tokenizer.tokenizer import CircuitTokenizer
from circuitgenie.training.generate import (
    generate_circuit, generate_netlist, validate_circuit, batch_generate,
)


def main():
    print("=" * 60)
    print("CircuitGenie: Circuit Generation")
    print("=" * 60)

    # Load model
    checkpoint_path = project_root / "checkpoints" / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Run scripts/train.py first.")
        return

    device = "cpu"  # Use CPU for inference
    config = CircuitGenieConfig()
    model = CircuitGenieModel(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    tokenizer = CircuitTokenizer()

    # Test specs
    test_cases = [
        {
            'name': 'Buck 12V->5V, 2A',
            'specs': {'v_in': 12.0, 'v_out': 5.0, 'i_out': 2.0, 'ripple_pct': 2.0, 'efficiency': 0.92},
            'topology': Topology.BUCK,
        },
        {
            'name': 'Boost 5V->12V, 0.5A',
            'specs': {'v_in': 5.0, 'v_out': 12.0, 'i_out': 0.5, 'ripple_pct': 3.0, 'efficiency': 0.90},
            'topology': Topology.BOOST,
        },
        {
            'name': 'Flyback 48V->5V, 1A (isolated)',
            'specs': {'v_in': 48.0, 'v_out': 5.0, 'i_out': 1.0, 'ripple_pct': 5.0, 'efficiency': 0.85},
            'topology': Topology.FLYBACK,
        },
        {
            'name': 'Auto-select topology: 24V->12V, 3A',
            'specs': {'v_in': 24.0, 'v_out': 12.0, 'i_out': 3.0, 'ripple_pct': 1.0, 'efficiency': 0.93},
            'topology': None,  # Let model choose
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['name']}")
        print(f"Specs: {tc['specs']}")
        print(f"Forced topology: {tc['topology']}")
        print("-" * 60)

        result = generate_circuit(
            model, tokenizer,
            specs=tc['specs'],
            topology=tc['topology'],
            temperature=0.7,
            top_k=15,
            device=device,
        )

        if result is None:
            print("  FAILED: Could not generate valid circuit")
            continue

        print(f"  Topology: {result['topology'].name}")
        print(f"  Params: { {k: f'{v:.4g}' for k, v in result['params'].items()} }")
        print(f"  Decoded specs: { {k: f'{v:.4g}' for k, v in result['specs'].items()} }")

        # Validate
        checks = validate_circuit(result)
        print(f"  Validation: {checks}")

        # Token sequence
        readable = tokenizer.to_readable(result['token_ids'])
        print(f"  Tokens: {readable}")

    # Batch generation test
    print(f"\n{'='*60}")
    print("Batch generation: 20 circuits for Buck 12V->5V")
    print("-" * 60)
    specs = {'v_in': 12.0, 'v_out': 5.0, 'i_out': 2.0, 'ripple_pct': 2.0, 'efficiency': 0.92}
    results = batch_generate(
        model, tokenizer, specs,
        n_samples=20,
        topology=Topology.BUCK,
        temperature=0.7,
        top_k=15,
        device=device,
    )

    valid = sum(1 for r in results if all(r['validation'].values()))
    print(f"Generated: {len(results)}/20 decodeable")
    print(f"Fully valid: {valid}/{len(results)}")

    if results:
        # Show param diversity
        print("\nParameter diversity (L, C, duty):")
        for i, r in enumerate(results[:5]):
            p = r['params']
            L = p.get('L', 0)
            C = p.get('C', 0)
            d = p.get('duty', 0)
            print(f"  [{i}] L={L:.2e} C={C:.2e} duty={d:.3f}")

    # Generate SPICE netlist
    print(f"\n{'='*60}")
    print("SPICE netlist generation")
    print("-" * 60)
    netlist = generate_netlist(
        model, tokenizer,
        specs={'v_in': 12.0, 'v_out': 5.0, 'i_out': 2.0, 'ripple_pct': 2.0, 'efficiency': 0.92},
        topology=Topology.BUCK,
        temperature=0.5,
        top_k=10,
        device=device,
    )
    if netlist:
        print(netlist[:500])
    else:
        print("Failed to generate netlist")


if __name__ == "__main__":
    main()
