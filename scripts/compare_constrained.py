#!/usr/bin/env python3
"""Compare constrained vs unconstrained ARCS generation.

Generates circuits at each constraint level and reports:
  - Structural validity rate
  - Component type correctness
  - Average generation time
  - (Optional) SPICE simulation metrics

Usage:
    PYTHONPATH=src python scripts/compare_constrained.py
    PYTHONPATH=src python scripts/compare_constrained.py --checkpoint checkpoints/best_model.pt
    PYTHONPATH=src python scripts/compare_constrained.py --n-samples 200 --simulate
"""

from __future__ import annotations

import argparse
import time

import torch

from arcs.constrained import ConstrainedGenerator, ConstraintLevel
from arcs.evaluate import (
    decode_generated_sequence,
    evaluate_generated_circuits,
    DecodedCircuit,
)
from arcs.model import ARCSConfig, ARCSModel
from arcs.model_enhanced import load_model
from arcs.simulate import (
    COMPONENT_TO_PARAM,
    TIER1_TEST_SPECS,
    TIER2_TEST_SPECS,
    ALL_TEST_SPECS,
    simulate_decoded_circuit,
    compute_reward,
)
from arcs.spice import NGSpiceRunner
from arcs.tokenizer import CircuitTokenizer


def build_prefix(tokenizer: CircuitTokenizer, topo: str, specs: dict) -> torch.Tensor:
    """Build a conditioning prefix."""
    prefix_ids = [tokenizer.start_id]
    topo_key = f"TOPO_{topo.upper()}"
    _topo_to_token = {
        "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
        "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
        "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
    }
    topo_key = _topo_to_token.get(topo, topo_key)
    if topo_key in tokenizer.name_to_id:
        prefix_ids.append(tokenizer.name_to_id[topo_key])
    prefix_ids.append(tokenizer.sep_id)
    for spec_name, spec_val in specs.items():
        spec_key = f"SPEC_{spec_name.upper()}"
        if spec_key in tokenizer.name_to_id:
            prefix_ids.append(tokenizer.name_to_id[spec_key])
            prefix_ids.append(tokenizer.encode_value(abs(spec_val)))
    prefix_ids.append(tokenizer.sep_id)
    return torch.tensor([prefix_ids])


def run_comparison(
    model,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    n_samples: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    tier: int | None = None,
    simulate: bool = False,
):
    """Run generation at each constraint level and compare results."""

    # Select test specs
    if tier == 1:
        test_specs = TIER1_TEST_SPECS
    elif tier == 2:
        test_specs = TIER2_TEST_SPECS
    else:
        test_specs = ALL_TEST_SPECS

    levels = [
        ConstraintLevel.NONE,
        ConstraintLevel.GRAMMAR,
        ConstraintLevel.TOPOLOGY,
        ConstraintLevel.FULL,
    ]

    results = {}

    for level in levels:
        print(f"\n{'='*60}")
        print(f"  Constraint Level: {level.name} ({level.value})")
        print(f"{'='*60}")

        gen = ConstrainedGenerator(model, tokenizer, level=level)
        circuits: list[DecodedCircuit] = []
        gen_times = []

        for i in range(n_samples):
            topo, specs = test_specs[i % len(test_specs)]
            prefix = build_prefix(tokenizer, topo, specs).to(device)

            t0 = time.perf_counter()
            if level == ConstraintLevel.NONE:
                output = model.generate(
                    prefix, max_new_tokens=80,
                    temperature=temperature, top_k=top_k,
                    tokenizer=tokenizer,
                )
            else:
                output = gen.generate(
                    prefix, topology=topo,
                    max_new_tokens=80,
                    temperature=temperature, top_k=top_k,
                )
            dt = (time.perf_counter() - t0) * 1000
            gen_times.append(dt)

            decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
            circuits.append(decoded)

        # Compute metrics
        n_valid = sum(1 for c in circuits if c.valid_structure)
        validity_rate = n_valid / len(circuits)
        avg_time = sum(gen_times) / len(gen_times)

        # Component type correctness
        n_comp_correct = 0
        n_comp_checked = 0
        for c in circuits:
            if not c.valid_structure or not c.topology:
                continue
            expected = COMPONENT_TO_PARAM.get(c.topology, [])
            if not expected:
                continue
            n_comp_checked += 1
            expected_types = sorted([ct for ct, _ in expected])
            actual_types = sorted([ct.upper() for ct, _ in c.components])
            if expected_types == actual_types:
                n_comp_correct += 1

        comp_rate = n_comp_correct / max(n_comp_checked, 1)

        # Avg components
        comp_counts = [len(c.components) for c in circuits if c.valid_structure]
        avg_comps = sum(comp_counts) / max(len(comp_counts), 1) if comp_counts else 0

        print(f"  Samples:             {len(circuits)}")
        print(f"  Valid structure:     {n_valid}/{len(circuits)} ({validity_rate:.1%})")
        print(f"  Component correct:  {n_comp_correct}/{n_comp_checked} ({comp_rate:.1%})")
        print(f"  Avg components:     {avg_comps:.1f}")
        print(f"  Avg gen time:       {avg_time:.1f} ms")

        # SPICE simulation
        if simulate:
            runner = NGSpiceRunner()
            n_sim_ok = 0
            n_sim_valid = 0
            rewards = []
            for c in circuits:
                if c.valid_structure:
                    outcome = simulate_decoded_circuit(c, runner=runner)
                    if outcome.success:
                        n_sim_ok += 1
                    if outcome.valid:
                        n_sim_valid += 1
                    r = compute_reward(c, outcome, c.specs)
                    rewards.append(r)
            avg_r = sum(rewards) / max(len(rewards), 1)
            print(f"  Sim success:        {n_sim_ok}/{n_valid} ({n_sim_ok/max(n_valid,1):.1%})")
            print(f"  Sim valid:          {n_sim_valid}/{n_valid} ({n_sim_valid/max(n_valid,1):.1%})")
            print(f"  Avg reward:         {avg_r:.3f}/8.0")

        results[level.name] = {
            "validity_rate": validity_rate,
            "comp_correctness": comp_rate,
            "avg_components": avg_comps,
            "avg_time_ms": avg_time,
        }

    # Summary table
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Level':<12} {'Valid%':>8} {'CompOK%':>8} {'AvgComp':>8} {'Time(ms)':>8}")
    print("-" * 50)
    for level_name, metrics in results.items():
        print(
            f"{level_name:<12} "
            f"{metrics['validity_rate']:>7.1%} "
            f"{metrics['comp_correctness']:>7.1%} "
            f"{metrics['avg_components']:>7.1f} "
            f"{metrics['avg_time_ms']:>7.1f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare ARCS constrained vs unconstrained generation"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (uses random init if not given)")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
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

    tokenizer = CircuitTokenizer()

    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}...")
        model, _, model_type = load_model(args.checkpoint, device=device)
        print(f"  Loaded {model_type} model")
    else:
        print("No checkpoint — using random-init baseline model")
        config = ARCSConfig.small()
        config.vocab_size = tokenizer.vocab_size
        model = ARCSModel(config).to(device)

    run_comparison(
        model, tokenizer, device,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        tier=args.tier,
        simulate=args.simulate,
    )


if __name__ == "__main__":
    main()
