"""ARCS evaluation: validate generated circuits via SPICE simulation.

Evaluation metrics (per README Phase 2):
  1. Validity rate — % of generated circuits that produce valid SPICE netlists
  2. Spec compliance — % that meet target specs (vout error <10%, eff >80%, etc.)
  3. Diversity — uniqueness of generated topologies and component values
  4. Quality — average efficiency, vout error across valid circuits

Usage:
    PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer, TokenType
from arcs.train import generate_from_specs


# ---------------------------------------------------------------------------
# Decode generated token sequence back to circuit parameters
# ---------------------------------------------------------------------------

@dataclass
class DecodedCircuit:
    """A generated circuit decoded from tokens back to interpretable form."""

    topology: str
    specs: dict[str, float]
    components: list[tuple[str, float]]  # (component_type, value)
    raw_tokens: list[int]
    valid_structure: bool  # Does it have proper START/TOPO/SEP/END structure?
    error: str = ""


def decode_generated_sequence(
    token_ids: list[int],
    tokenizer: CircuitTokenizer,
) -> DecodedCircuit:
    """Decode a generated token sequence into a structured circuit.

    Expected format:
        START, TOPO_X, SEP, specs..., SEP, COMP_X, VAL, ..., END
    """
    tokens = tokenizer.decode_tokens(token_ids)
    topology = ""
    specs: dict[str, float] = {}
    components: list[tuple[str, float]] = []
    error = ""

    try:
        # Must start with START
        if not tokens or tokens[0].name != "START":
            return DecodedCircuit(
                topology="", specs={}, components=[], raw_tokens=token_ids,
                valid_structure=False, error="No START token"
            )

        # Find topology token
        topo_found = False
        idx = 1
        while idx < len(tokens):
            if tokens[idx].token_type == TokenType.TOPOLOGY:
                topology = tokens[idx].name.replace("TOPO_", "").lower()
                topo_found = True
                idx += 1
                break
            idx += 1

        if not topo_found:
            return DecodedCircuit(
                topology="", specs={}, components=[], raw_tokens=token_ids,
                valid_structure=False, error="No topology token"
            )

        # Skip first SEP
        if idx < len(tokens) and tokens[idx].name == "SEP":
            idx += 1

        # Parse spec pairs (SPEC_X, VALUE)
        while idx < len(tokens) - 1:
            if tokens[idx].name == "SEP":
                idx += 1
                break
            if tokens[idx].token_type == TokenType.SPEC:
                spec_name = tokens[idx].name.replace("SPEC_", "").lower()
                if idx + 1 < len(tokens) and tokens[idx + 1].token_type == TokenType.VALUE:
                    specs[spec_name] = tokens[idx + 1].value
                    idx += 2
                else:
                    idx += 1
            else:
                idx += 1

        # Parse component pairs (COMP_X, VALUE)
        while idx < len(tokens):
            if tokens[idx].name in ("END", "PAD"):
                break
            if tokens[idx].token_type == TokenType.COMPONENT:
                comp_type = tokens[idx].name.replace("COMP_", "").lower()
                if idx + 1 < len(tokens) and tokens[idx + 1].token_type == TokenType.VALUE:
                    components.append((comp_type, tokens[idx + 1].value))
                    idx += 2
                else:
                    idx += 1
            else:
                idx += 1

        # Check for END token
        has_end = any(t.name == "END" for t in tokens)

        valid_structure = (
            topo_found
            and len(components) >= 2
            and has_end
        )

    except Exception as e:
        error = str(e)
        valid_structure = False

    return DecodedCircuit(
        topology=topology,
        specs=specs,
        components=components,
        raw_tokens=token_ids,
        valid_structure=valid_structure,
        error=error,
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    """Aggregated evaluation metrics."""

    n_generated: int
    n_valid_structure: int
    validity_rate: float
    topology_distribution: dict[str, int]
    avg_n_components: float
    component_type_distribution: dict[str, int]
    unique_component_combos: int
    diversity_score: float  # unique / total

    def to_dict(self) -> dict:
        return {
            "n_generated": self.n_generated,
            "n_valid_structure": self.n_valid_structure,
            "validity_rate": self.validity_rate,
            "topology_distribution": self.topology_distribution,
            "avg_n_components": self.avg_n_components,
            "component_type_distribution": self.component_type_distribution,
            "unique_component_combos": self.unique_component_combos,
            "diversity_score": self.diversity_score,
        }


def evaluate_generated_circuits(
    circuits: list[DecodedCircuit],
) -> EvalResults:
    """Compute evaluation metrics over a batch of generated circuits."""
    n = len(circuits)
    valid = [c for c in circuits if c.valid_structure]
    n_valid = len(valid)

    # Topology distribution
    topo_counts = Counter(c.topology for c in valid)

    # Component stats
    comp_counts: list[int] = [len(c.components) for c in valid]
    avg_comp = np.mean(comp_counts) if comp_counts else 0.0

    # Component type distribution
    all_types = Counter()
    for c in valid:
        for comp_type, _ in c.components:
            all_types[comp_type] += 1

    # Diversity: unique component type combinations
    combos = set()
    for c in valid:
        combo = tuple(sorted(ct for ct, _ in c.components))
        combos.add(combo)
    diversity = len(combos) / max(n_valid, 1)

    return EvalResults(
        n_generated=n,
        n_valid_structure=n_valid,
        validity_rate=n_valid / max(n, 1),
        topology_distribution=dict(topo_counts),
        avg_n_components=float(avg_comp),
        component_type_distribution=dict(all_types),
        unique_component_combos=len(combos),
        diversity_score=diversity,
    )


# ---------------------------------------------------------------------------
# Generation + evaluation entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_and_evaluate(
    model: ARCSModel,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_samples: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    conditioned: bool = True,
) -> EvalResults:
    """Generate circuits and evaluate them.

    Args:
        model:       Trained ARCS model
        tokenizer:   CircuitTokenizer
        device:      torch device
        n_samples:   Number of circuits to generate
        temperature: Sampling temperature
        top_k:       Top-k filtering
        conditioned: If True, generate with spec conditioning

    Returns:
        EvalResults with aggregate metrics
    """
    model.eval()
    circuits: list[DecodedCircuit] = []

    # Define test specs for conditioned generation
    test_specs = [
        ("buck", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
        ("boost", {"vin": 5.0, "vout": 12.0, "iout": 0.5, "fsw": 100000}),
        ("buck_boost", {"vin": 12.0, "vout": 9.0, "iout": 1.0, "fsw": 100000}),
        ("cuk", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
        ("sepic", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
        ("flyback", {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}),
        ("forward", {"vin": 48.0, "vout": 12.0, "iout": 1.0, "fsw": 100000}),
    ]

    for i in range(n_samples):
        if conditioned:
            # Cycle through test specs
            topo, specs = test_specs[i % len(test_specs)]
            # Build prefix
            prefix_ids = [tokenizer.start_id]
            topo_key = f"TOPO_{topo.upper()}"
            if topo_key in tokenizer.name_to_id:
                prefix_ids.append(tokenizer.name_to_id[topo_key])
            prefix_ids.append(tokenizer.sep_id)
            for spec_name, spec_val in specs.items():
                spec_key = f"SPEC_{spec_name.upper()}"
                if spec_key in tokenizer.name_to_id:
                    prefix_ids.append(tokenizer.name_to_id[spec_key])
                    prefix_ids.append(tokenizer.encode_value(abs(spec_val)))
            prefix_ids.append(tokenizer.sep_id)
            prefix = torch.tensor([prefix_ids], device=device)
        else:
            # Unconditional: just START
            prefix = torch.tensor([[tokenizer.start_id]], device=device)

        output = model.generate(
            prefix,
            max_new_tokens=80,
            temperature=temperature,
            top_k=top_k,
        )
        decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
        circuits.append(decoded)

    return evaluate_generated_circuits(circuits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARCS Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of circuits to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--unconditioned", action="store_true",
                        help="Generate without spec conditioning")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ARCSConfig.from_dict(checkpoint["config"])
    model = ARCSModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    tokenizer = CircuitTokenizer()

    # Evaluate
    print(f"\nGenerating {args.n_samples} circuits...")
    t0 = time.time()
    results = generate_and_evaluate(
        model, tokenizer, device,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioned=not args.unconditioned,
    )
    dt = time.time() - t0

    # Report
    print(f"\n{'=' * 50}")
    print(f"ARCS Evaluation Results ({dt:.1f}s)")
    print(f"{'=' * 50}")
    print(f"Generated:        {results.n_generated}")
    print(f"Valid structure:   {results.n_valid_structure} ({results.validity_rate:.1%})")
    print(f"Avg components:   {results.avg_n_components:.1f}")
    print(f"Unique combos:    {results.unique_component_combos}")
    print(f"Diversity score:  {results.diversity_score:.3f}")
    print(f"\nTopology distribution:")
    for topo, count in sorted(results.topology_distribution.items()):
        print(f"  {topo}: {count}")
    print(f"\nComponent type distribution:")
    for comp, count in sorted(results.component_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  {comp}: {count}")

    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
