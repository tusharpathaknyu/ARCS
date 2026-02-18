#!/usr/bin/env python3
"""Load MLEntry simulation data and prepare CircuitGenie dataset."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from circuitgenie.data.generator import load_mlentry_data, save_dataset


def main():
    print("=" * 60)
    print("CircuitGenie Phase 1: Data Generation")
    print("=" * 60)

    # Load and subsample MLEntry data
    samples = load_mlentry_data(
        samples_per_topology=2000,
        seed=42,
    )

    # Print summary statistics
    from collections import Counter
    topo_counts = Counter(s.topology.name for s in samples)
    print("\nSamples per topology:")
    for name, count in sorted(topo_counts.items()):
        print(f"  {name}: {count}")

    # Print a few examples
    print("\n--- Example samples ---")
    for i in [0, len(samples) // 2, -1]:
        s = samples[i]
        print(f"\n[{i}] Topology: {s.topology.name}")
        print(f"    Params: { {k: f'{v:.4g}' for k, v in s.params.items()} }")
        print(f"    Specs:  { {k: f'{v:.4g}' for k, v in s.specs.items()} }")

    # Check spec ranges
    print("\n--- Spec ranges ---")
    for key in ['v_in', 'v_out', 'i_out', 'ripple_pct', 'efficiency']:
        vals = [s.specs[key] for s in samples]
        print(f"  {key}: min={min(vals):.4f}, max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}")

    # Save
    output_path = project_root / "data" / "phase1_dataset.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(samples, str(output_path))

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {len(samples)}")


if __name__ == "__main__":
    main()
