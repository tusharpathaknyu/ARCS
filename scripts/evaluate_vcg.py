#!/usr/bin/env python3
"""Evaluate ValidCircuitGen and compare with autoregressive ARCS models.

Usage:
    # Evaluate a trained VCG checkpoint
    PYTHONPATH=src python scripts/evaluate_vcg.py \
        --vcg-checkpoint checkpoints/vcg/best_model.pt \
        --data data/combined \
        --n-samples 160

    # Compare VCG vs autoregressive models
    PYTHONPATH=src python scripts/evaluate_vcg.py \
        --vcg-checkpoint checkpoints/vcg/best_model.pt \
        --arcs-checkpoint checkpoints/arcs_graph_transformer/best_model.pt \
        --data data/combined \
        --n-samples 160 -v

Evaluates VCG on:
  - Structural validity (5 differentiable constraints)
  - Constraint satisfaction rates
  - Generation diversity (topology coverage, component diversity)
  - Latent space quality (interpolation smoothness, reconstruction)
  - Comparison table vs autoregressive ARCS (if checkpoint provided)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np

from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig,
    ValidCircuitGenModel,
    CircuitGraphDataset,
    CircuitGraph,
    check_circuit_validity,
    graph_to_token_sequence,
    IDX_TO_NODE_TYPE,
    TOPOLOGY_TO_IDX,
)


# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════


def load_vcg_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ValidCircuitGenModel, VCGConfig]:
    """Load a trained VCG model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = VCGConfig.from_dict(ckpt["config"])
    model = ValidCircuitGenModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_validity(
    model: ValidCircuitGenModel,
    dataset: CircuitGraphDataset,
    n_samples: int,
    device: torch.device,
    use_projection: bool = True,
) -> dict:
    """Evaluate structural validity of generated circuits.

    Returns dict with validity rates per constraint and overall.
    """
    model.eval()

    # Collect unique topologies (sample throughout dataset for diversity)
    topo_items: dict[int, dict] = {}
    step = max(1, len(dataset) // 1000)
    for i in range(0, len(dataset), step):
        item = dataset[i]
        topo_idx = item["topology_idx"].item()
        if topo_idx not in topo_items:
            topo_items[topo_idx] = item

    if not topo_items:
        return {"validity_rate": 0.0, "n_generated": 0}

    constraint_results: dict[str, list[bool]] = {
        "no_floating_nodes": [],
        "device_completeness": [],
        "no_short_circuits": [],
        "graph_connected": [],
        "values_in_bounds": [],
        "valid": [],
    }
    per_topology: dict[str, dict] = {}
    all_node_types: list[str] = []
    generation_times: list[float] = []

    per_topo_count = max(1, n_samples // len(topo_items))

    for topo_idx, template in topo_items.items():
        # Find topology name
        topo_name = "unknown"
        for name, idx in TOPOLOGY_TO_IDX.items():
            if idx == topo_idx:
                topo_name = name
                break

        t0 = time.time()
        graphs, stats = model.generate(
            spec_types=template["spec_types"].unsqueeze(0).to(device),
            spec_values=template["spec_values"].unsqueeze(0).to(device),
            spec_mask=template["spec_mask"].unsqueeze(0).to(device),
            topology_idx=template["topology_idx"].unsqueeze(0).to(device),
            active_mask=template["active_mask"].unsqueeze(0).to(device),
            bounds_min=template["value_bounds_min"].unsqueeze(0).to(device),
            bounds_max=template["value_bounds_max"].unsqueeze(0).to(device),
            n_samples=per_topo_count,
            use_projection=use_projection,
        )
        gen_time = time.time() - t0
        generation_times.append(gen_time / len(graphs))

        topo_valid = 0
        topo_total = 0

        for graph in graphs:
            validity = check_circuit_validity(graph)
            for k in constraint_results:
                constraint_results[k].append(validity.get(k, False))

            if validity["valid"]:
                topo_valid += 1
            topo_total += 1

            # Collect node types for diversity
            for ci in range(graph.n_components):
                nt = graph.node_types[ci].item()
                if nt in IDX_TO_NODE_TYPE:
                    all_node_types.append(IDX_TO_NODE_TYPE[nt])

        per_topology[topo_name] = {
            "n_generated": topo_total,
            "n_valid": topo_valid,
            "validity_rate": topo_valid / max(topo_total, 1),
            "avg_gen_time_ms": (gen_time / max(topo_total, 1)) * 1000,
        }

    # Aggregate
    n_total = len(constraint_results["valid"])
    rates = {
        k: sum(v) / max(len(v), 1) for k, v in constraint_results.items()
    }

    # Node type diversity
    type_counts = Counter(all_node_types)
    n_unique_types = len(type_counts)

    return {
        "n_generated": n_total,
        "validity_rate": rates["valid"],
        "constraint_rates": {
            "no_floating_nodes": rates["no_floating_nodes"],
            "device_completeness": rates["device_completeness"],
            "no_short_circuits": rates["no_short_circuits"],
            "graph_connected": rates["graph_connected"],
            "values_in_bounds": rates["values_in_bounds"],
        },
        "per_topology": per_topology,
        "avg_gen_time_ms": np.mean(generation_times) * 1000,
        "n_unique_node_types": n_unique_types,
        "node_type_distribution": dict(type_counts),
        "projection_enabled": use_projection,
    }


def evaluate_reconstruction(
    model: ValidCircuitGenModel,
    dataset: CircuitGraphDataset,
    n_samples: int,
    device: torch.device,
) -> dict:
    """Evaluate reconstruction quality (encode → decode → compare)."""
    model.eval()
    n_eval = min(n_samples, len(dataset))

    type_correct = 0
    type_total = 0
    adj_correct = 0
    adj_total = 0
    value_errors: list[float] = []
    kl_values: list[float] = []

    for i in range(n_eval):
        item = dataset[i]
        batch = {k: v.unsqueeze(0).to(device) for k, v in item.items()}

        with torch.no_grad():
            mu, logvar, spec_embed = model.encode(batch)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().item()
            kl_values.append(kl)

            # Decode from mean (no sampling)
            soft_X, soft_A, soft_V = model.decode(mu, spec_embed, batch["topology_idx"])

        active = batch["active_mask"][0]
        n_active = int(active.sum().item())

        # Type accuracy
        pred_types = soft_X[0].argmax(dim=-1)
        true_types = batch["node_types"][0]
        for j in range(n_active):
            if pred_types[j].item() == true_types[j].item():
                type_correct += 1
            type_total += 1

        # Adjacency accuracy
        pred_adj = (soft_A[0] > 0.5).float()
        true_adj = batch["adjacency"][0]
        for r in range(n_active):
            for c in range(r + 1, n_active):
                if pred_adj[r, c].item() == true_adj[r, c].item():
                    adj_correct += 1
                adj_total += 1

        # Value error (log10 scale)
        for j in range(n_active):
            err = abs(soft_V[0, j].item() - batch["values"][0, j].item())
            value_errors.append(err)

    return {
        "n_evaluated": n_eval,
        "type_accuracy": type_correct / max(type_total, 1),
        "adjacency_accuracy": adj_correct / max(adj_total, 1),
        "avg_value_error_log10": np.mean(value_errors) if value_errors else 0.0,
        "avg_kl": np.mean(kl_values) if kl_values else 0.0,
    }


def evaluate_latent_space(
    model: ValidCircuitGenModel,
    dataset: CircuitGraphDataset,
    n_pairs: int,
    device: torch.device,
) -> dict:
    """Evaluate latent space quality via interpolation smoothness.

    For pairs of circuits, interpolate in latent space and measure how
    smoothly the decoded output changes.
    """
    model.eval()
    n_pairs = min(n_pairs, len(dataset) // 2)

    smoothness_scores: list[float] = []

    for i in range(n_pairs):
        item_a = {k: v.unsqueeze(0).to(device) for k, v in dataset[2 * i].items()}
        item_b = {k: v.unsqueeze(0).to(device) for k, v in dataset[2 * i + 1].items()}

        with torch.no_grad():
            results = model.interpolate(item_a, item_b, n_steps=5)

        # Measure smoothness: average change between consecutive steps
        diffs = []
        for step in range(len(results) - 1):
            X_a, A_a, V_a = results[step]
            X_b, A_b, V_b = results[step + 1]
            diff = (
                (X_a - X_b).abs().mean().item()
                + (A_a - A_b).abs().mean().item()
                + (V_a - V_b).abs().mean().item()
            )
            diffs.append(diff)

        # Smooth = small & consistent diffs
        if diffs:
            smoothness = 1.0 / (1.0 + np.std(diffs))
            smoothness_scores.append(smoothness)

    return {
        "n_pairs": n_pairs,
        "avg_smoothness": np.mean(smoothness_scores) if smoothness_scores else 0.0,
        "std_smoothness": np.std(smoothness_scores) if smoothness_scores else 0.0,
    }


def evaluate_projection_impact(
    model: ValidCircuitGenModel,
    dataset: CircuitGraphDataset,
    n_samples: int,
    device: torch.device,
) -> dict:
    """Compare validity with and without constraint projection."""
    results_with = evaluate_validity(
        model, dataset, n_samples, device, use_projection=True,
    )
    results_without = evaluate_validity(
        model, dataset, n_samples, device, use_projection=False,
    )

    return {
        "with_projection": {
            "validity_rate": results_with["validity_rate"],
            "avg_gen_time_ms": results_with["avg_gen_time_ms"],
        },
        "without_projection": {
            "validity_rate": results_without["validity_rate"],
            "avg_gen_time_ms": results_without["avg_gen_time_ms"],
        },
        "projection_improvement": (
            results_with["validity_rate"] - results_without["validity_rate"]
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Comparison with Autoregressive ARCS
# ═══════════════════════════════════════════════════════════════════════════


def compare_with_arcs(
    vcg_results: dict,
    arcs_checkpoint: str | None,
    n_samples: int,
    device: torch.device,
) -> dict | None:
    """Compare VCG results with autoregressive ARCS model."""
    if arcs_checkpoint is None or not Path(arcs_checkpoint).exists():
        return None

    from arcs.model_enhanced import load_model
    from arcs.evaluate import generate_and_evaluate

    model, config, model_type = load_model(arcs_checkpoint, device=device)
    tokenizer = CircuitTokenizer()

    t0 = time.time()
    results = generate_and_evaluate(
        model, tokenizer, device,
        n_samples=n_samples,
        temperature=0.8,
        top_k=50,
        conditioned=True,
        simulate=False,
    )
    arcs_time = time.time() - t0

    return {
        "method": f"ARCS {model_type}",
        "n_params": model.count_parameters(),
        "struct_valid_rate": results.validity_rate,
        "n_samples": n_samples,
        "wall_time": arcs_time,
        "avg_time_per_sample_ms": (arcs_time / n_samples) * 1000,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════


def print_vcg_report(
    validity: dict,
    reconstruction: dict,
    latent: dict,
    projection: dict,
    arcs_comparison: dict | None,
    model: ValidCircuitGenModel,
) -> None:
    """Print comprehensive evaluation report."""
    print(f"\n{'='*70}")
    print("ValidCircuitGen Evaluation Report")
    print(f"{'='*70}")

    # Model info
    n_params = model.count_parameters()
    groups = model.count_parameters_by_group()
    print(f"\nModel: {n_params:,} parameters")
    for group, count in groups.items():
        print(f"  {group:<16s}: {count:>8,}")

    # Validity
    print(f"\n{'─'*70}")
    print("1. STRUCTURAL VALIDITY")
    print(f"{'─'*70}")
    print(f"  Overall validity:    {validity['validity_rate']:.1%}")
    print(f"  Samples generated:   {validity['n_generated']}")
    print(f"  Avg gen time:        {validity['avg_gen_time_ms']:.1f}ms")
    print(f"  Projection enabled:  {validity['projection_enabled']}")
    print(f"\n  Per-constraint satisfaction:")
    for c, rate in validity["constraint_rates"].items():
        print(f"    {c:<25s}: {rate:.1%}")

    if validity["per_topology"]:
        print(f"\n  Per-topology breakdown:")
        for topo, d in sorted(validity["per_topology"].items()):
            print(
                f"    {topo:<25s}: {d['n_valid']:>3d}/{d['n_generated']:>3d} "
                f"({d['validity_rate']:.0%}) [{d['avg_gen_time_ms']:.1f}ms]"
            )

    # Reconstruction
    print(f"\n{'─'*70}")
    print("2. RECONSTRUCTION QUALITY")
    print(f"{'─'*70}")
    print(f"  Type accuracy:       {reconstruction['type_accuracy']:.1%}")
    print(f"  Adjacency accuracy:  {reconstruction['adjacency_accuracy']:.1%}")
    print(f"  Value error (log10): {reconstruction['avg_value_error_log10']:.3f}")
    print(f"  Avg KL divergence:   {reconstruction['avg_kl']:.3f}")

    # Latent space
    print(f"\n{'─'*70}")
    print("3. LATENT SPACE QUALITY")
    print(f"{'─'*70}")
    print(f"  Interpolation smoothness: {latent['avg_smoothness']:.3f} ± {latent['std_smoothness']:.3f}")
    print(f"  Pairs evaluated:          {latent['n_pairs']}")

    # Projection impact
    print(f"\n{'─'*70}")
    print("4. CONSTRAINT PROJECTION IMPACT")
    print(f"{'─'*70}")
    print(f"  Validity w/ projection:   {projection['with_projection']['validity_rate']:.1%}")
    print(f"  Validity w/o projection:  {projection['without_projection']['validity_rate']:.1%}")
    print(f"  Improvement:              +{projection['projection_improvement']:.1%}")
    print(f"  Time w/ projection:       {projection['with_projection']['avg_gen_time_ms']:.1f}ms")
    print(f"  Time w/o projection:      {projection['without_projection']['avg_gen_time_ms']:.1f}ms")

    # ARCS comparison
    if arcs_comparison:
        print(f"\n{'─'*70}")
        print("5. COMPARISON WITH AUTOREGRESSIVE ARCS")
        print(f"{'─'*70}")

        header = f"| {'Method':<25s} | {'Params':>8s} | {'Struct Valid':>12s} | {'Time/Sample':>12s} |"
        sep = f"|{'-'*27}|{'-'*10}|{'-'*14}|{'-'*14}|"
        print(header)
        print(sep)

        print(
            f"| {'ValidCircuitGen':<25s} | {n_params/1e6:>6.2f}M | "
            f"{validity['validity_rate']:>11.1%} | "
            f"{validity['avg_gen_time_ms']:>10.1f}ms |"
        )
        print(
            f"| {arcs_comparison['method']:<25s} | "
            f"{arcs_comparison['n_params']/1e6:>6.2f}M | "
            f"{arcs_comparison['struct_valid_rate']:>11.1%} | "
            f"{arcs_comparison['avg_time_per_sample_ms']:>10.1f}ms |"
        )

    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Evaluate ValidCircuitGen")
    parser.add_argument(
        "--vcg-checkpoint", type=str, required=True,
        help="Path to VCG checkpoint (.pt)",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to JSONL data (directory or file)",
    )
    parser.add_argument(
        "--arcs-checkpoint", type=str, default=None,
        help="Optional autoregressive ARCS checkpoint for comparison",
    )
    parser.add_argument("--n-samples", type=int, default=160)
    parser.add_argument("--n-interp-pairs", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/vcg_evaluation.json")
    parser.add_argument("-v", "--verbose", action="store_true")
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
    print(f"Device: {device}")

    # Load model
    print(f"Loading VCG from {args.vcg_checkpoint}...")
    model, config = load_vcg_model(args.vcg_checkpoint, device)
    print(f"  {model.count_parameters():,} parameters, latent_dim={config.latent_dim}")

    # Load data
    tokenizer = CircuitTokenizer()
    print(f"Loading data from {args.data}...")
    dataset = CircuitGraphDataset(args.data, tokenizer, config, valid_only=True)

    # Run evaluations
    print("\n[1/4] Evaluating structural validity...")
    validity = evaluate_validity(model, dataset, args.n_samples, device)

    print("[2/4] Evaluating reconstruction quality...")
    reconstruction = evaluate_reconstruction(model, dataset, args.n_samples, device)

    print("[3/4] Evaluating latent space...")
    latent = evaluate_latent_space(model, dataset, args.n_interp_pairs, device)

    print("[4/4] Evaluating projection impact...")
    projection = evaluate_projection_impact(
        model, dataset, min(args.n_samples, 50), device,
    )

    # Optional ARCS comparison
    arcs_comparison = None
    if args.arcs_checkpoint:
        print("[+] Comparing with autoregressive ARCS...")
        arcs_comparison = compare_with_arcs(
            validity, args.arcs_checkpoint, args.n_samples, device,
        )

    # Report
    print_vcg_report(
        validity, reconstruction, latent, projection, arcs_comparison, model,
    )

    # Save
    all_results = {
        "vcg_params": model.count_parameters(),
        "vcg_config": config.to_dict(),
        "validity": validity,
        "reconstruction": reconstruction,
        "latent_space": latent,
        "projection_impact": projection,
    }
    if arcs_comparison:
        all_results["arcs_comparison"] = arcs_comparison

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
