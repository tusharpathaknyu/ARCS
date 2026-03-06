#!/usr/bin/env python3
"""Run Best-of-N inference-time scaling experiments.

Generates circuits at N = {1, 3, 5, 10, 20, 50} candidates per spec,
ranks by model confidence, and reports quality scaling curves.

Usage:
    PYTHONPATH=src python scripts/run_bestofn.py
    PYTHONPATH=src python scripts/run_bestofn.py --checkpoint checkpoints/arcs_combined/best_model.pt
    PYTHONPATH=src python scripts/run_bestofn.py --simulate --n-specs 160
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from arcs.bestofn import (
    BestOfNGenerator,
    run_scaling_experiment,
    calibration_analysis,
)
from arcs.constrained import ConstraintLevel
from arcs.model import ARCSConfig, ARCSModel
from arcs.model_enhanced import load_model
from arcs.tokenizer import CircuitTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Best-of-N inference-time scaling experiment"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n-specs", type=int, default=160)
    parser.add_argument("--n-values", type=int, nargs="+",
                        default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--constraint", type=str, default="topology",
                        choices=["none", "grammar", "topology", "full"])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--calibration", action="store_true",
                        help="Run calibration analysis")
    parser.add_argument("--cal-samples", type=int, default=200)
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
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

    # Load model
    if args.checkpoint:
        print(f"Loading model from {args.checkpoint}...")
        model, _, model_type = load_model(args.checkpoint, device=device)
        print(f"  Loaded {model_type} model")
    else:
        print("No checkpoint — using random-init baseline model")
        config = ARCSConfig.small()
        config.vocab_size = tokenizer.vocab_size
        model = ARCSModel(config).to(device)
        model.eval()

    # Constraint level
    level_map = {
        "none": ConstraintLevel.NONE,
        "grammar": ConstraintLevel.GRAMMAR,
        "topology": ConstraintLevel.TOPOLOGY,
        "full": ConstraintLevel.FULL,
    }
    constraint_level = level_map[args.constraint]

    print(f"\nDevice: {device}")
    print(f"Constraint: {constraint_level.name}")
    print(f"N values: {args.n_values}")
    print(f"Specs: {args.n_specs}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    if args.simulate:
        print("SPICE simulation: ENABLED")

    # --- Scaling experiment ---
    print(f"\n{'=' * 60}")
    print("  INFERENCE-TIME SCALING EXPERIMENT")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    results = run_scaling_experiment(
        model, tokenizer, device,
        n_values=args.n_values,
        n_specs=args.n_specs,
        constraint_level=constraint_level,
        temperature=args.temperature,
        top_k=args.top_k,
        simulate=args.simulate,
        tier=args.tier,
        verbose=args.verbose,
    )
    total_time = time.perf_counter() - t0

    # Summary table
    print(f"\n{'=' * 60}")
    print("  SCALING SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'N':>4} {'Valid%':>8} {'Confidence':>12} {'Entropy':>10} {'Diversity':>10} {'Time/spec':>10}"
    if args.simulate:
        header += f" {'Reward':>8} {'SimOK%':>8}"
    print(header)
    print("-" * len(header))

    for n in sorted(results.keys()):
        m = results[n]
        line = (
            f"{n:>4} "
            f"{m['validity_rate']:>7.1%} "
            f"{m['avg_confidence']:>11.4f} "
            f"{m['avg_entropy']:>9.4f} "
            f"{m['avg_diversity']:>9.1%} "
            f"{m['avg_time_ms']:>8.1f}ms"
        )
        if args.simulate and "avg_reward" in m:
            line += f" {m['avg_reward']:>7.3f} {m['sim_valid_rate']:>7.1%}"
        print(line)

    print(f"\nTotal experiment time: {total_time:.1f}s")

    # Confidence improvement
    n_vals = sorted(results.keys())
    if len(n_vals) >= 2:
        base_conf = results[n_vals[0]]["avg_confidence"]
        best_conf = results[n_vals[-1]]["avg_confidence"]
        print(f"Confidence improvement (N=1→{n_vals[-1]}): "
              f"{base_conf:.4f} → {best_conf:.4f} "
              f"(Δ = {best_conf - base_conf:+.4f})")

    # --- Optional calibration ---
    if args.calibration:
        print(f"\n{'=' * 60}")
        print("  CALIBRATION ANALYSIS")
        print(f"{'=' * 60}")

        cal = calibration_analysis(
            model, tokenizer, device,
            n_samples=args.cal_samples,
            constraint_level=constraint_level,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        print(f"  Total samples:    {cal['n_total']}")
        print(f"  Valid (SPICE):    {cal['n_valid']}")
        print(f"  Invalid:          {cal['n_invalid']}")
        print(f"  Valid mean conf:  {cal['valid_mean_conf']:.4f}")
        print(f"  Invalid mean conf:{cal['invalid_mean_conf']:.4f}")
        print(f"  Confidence gap:   {cal['conf_gap']:.4f}")
        print(f"  Correlation:      {cal['correlation']:.4f}")

    # Save results
    if args.output:
        output_data = {
            "scaling": {str(k): v for k, v in results.items()},
            "config": {
                "checkpoint": args.checkpoint,
                "constraint": args.constraint,
                "n_specs": args.n_specs,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "device": str(device),
            },
        }
        if args.calibration:
            cal_save = {k: v for k, v in cal.items()
                        if k not in ("valid_confidences", "invalid_confidences")}
            output_data["calibration"] = cal_save

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
