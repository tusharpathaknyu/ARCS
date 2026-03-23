#!/usr/bin/env python3
"""Evaluate the trained CCFM model on generation quality.

Usage:
    # Structural validity only (fast)
    PYTHONPATH=src python scripts/evaluate_ccfm.py \
        --ccfm-checkpoint checkpoints/ccfm/best_ccfm.pt \
        --data data/combined \
        --n-samples 340

    # Unified SPICE simulation evaluation (apples-to-apples with ARCS)
    PYTHONPATH=src python scripts/evaluate_ccfm.py \
        --ccfm-checkpoint checkpoints/ccfm/best_ccfm.pt \
        --data data/combined \
        --n-samples 340 \
        --spice \
        --output results/ccfm_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np

from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig, CircuitGraphDataset, CircuitGraph,
    check_circuit_validity, TOPOLOGY_TO_IDX,
)
from arcs.flow_matching import ConstrainedFlowMatchingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Core Evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_structural_validity(
    model: ConstrainedFlowMatchingModel,
    dataset: CircuitGraphDataset,
    n_samples: int,
    device: torch.device,
) -> dict:
    """Evaluate structural validity of CCFM-generated circuits."""
    model.eval()

    # Collect one template per topology
    topo_items: dict[int, dict] = {}
    step = max(1, len(dataset) // 1000)
    for i in range(0, len(dataset), step):
        item = dataset[i]
        tidx = item["topology_idx"].item()
        if tidx not in topo_items:
            topo_items[tidx] = item

    if not topo_items:
        return {"validity_rate": 0.0, "n_generated": 0}

    n_per_topo = max(1, n_samples // len(topo_items))
    logger.info(f"Found {len(topo_items)} topologies → {n_per_topo} samples each")

    per_topology: dict[str, dict] = {}
    total_valid = 0
    total_gen = 0

    for topo_idx, template in topo_items.items():
        topo_name = "unknown"
        for name, idx in TOPOLOGY_TO_IDX.items():
            if idx == topo_idx:
                topo_name = name
                break

        spec_types = template["spec_types"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        spec_values = template["spec_values"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        spec_mask = template["spec_mask"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        topo_t = template["topology_idx"].unsqueeze(0).expand(n_per_topo).to(device)
        active = template["active_mask"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        bmin = template["value_bounds_min"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        bmax = template["value_bounds_max"].unsqueeze(0).expand(n_per_topo, -1).to(device)

        with torch.no_grad():
            soft_X, soft_A, soft_V, _ = model.sample_with_projection(
                spec_types, spec_values, spec_mask, topo_t, active, bmin, bmax,
            )

        pred_types = soft_X.argmax(dim=-1)
        pred_adj = (soft_A > 0.5).float()
        pred_values = torch.clamp(soft_V, min=bmin, max=bmax)

        topo_valid = 0
        for b in range(n_per_topo):
            n_active = int(active[b].sum().item())
            graph = CircuitGraph(
                topology=topo_name, n_components=n_active,
                node_types=pred_types[b].cpu(), adjacency=pred_adj[b].cpu(),
                values=pred_values[b].cpu(), active_mask=active[b].cpu(),
                spec_types=spec_types[b].cpu(), spec_values=spec_values[b].cpu(),
                spec_mask=spec_mask[b].cpu(),
                value_bounds_min=bmin[b].cpu(), value_bounds_max=bmax[b].cpu(),
            )
            v = check_circuit_validity(graph)
            if v["valid"]:
                topo_valid += 1

        per_topology[topo_name] = {
            "n_generated": n_per_topo,
            "n_valid": topo_valid,
            "validity_rate": topo_valid / n_per_topo,
        }
        total_valid += topo_valid
        total_gen += n_per_topo
        logger.info(
            f"  {topo_name:25s}: {topo_valid}/{n_per_topo} valid "
            f"({100*topo_valid/n_per_topo:.0f}%)"
        )

    overall = total_valid / max(total_gen, 1)
    return {
        "n_generated": total_gen,
        "n_valid": total_valid,
        "validity_rate": overall,
        "per_topology": per_topology,
    }


def evaluate_spice_validity(
    model: ConstrainedFlowMatchingModel,
    dataset: CircuitGraphDataset,
    n_samples: int,
    device: torch.device,
) -> dict:
    """Run SPICE simulation on structurally valid CCFM-generated circuits.

    Provides apples-to-apples comparison with ARCS autoregressive evaluation.
    """
    from arcs.hybrid_pipeline import vcg_graph_to_spice
    from arcs.spice import NGSpiceRunner

    model.eval()
    runner = NGSpiceRunner()
    tokenizer = CircuitTokenizer()

    topo_items: dict[int, dict] = {}
    step = max(1, len(dataset) // 1000)
    for i in range(0, len(dataset), step):
        item = dataset[i]
        tidx = item["topology_idx"].item()
        if tidx not in topo_items:
            topo_items[tidx] = item

    if not topo_items:
        return {"n_generated": 0, "sim_valid_rate": 0.0}

    n_per_topo = max(1, n_samples // len(topo_items))

    per_topology: dict[str, dict] = {}
    total_generated = 0
    total_struct_valid = 0
    total_sim_valid = 0
    total_reward = 0.0

    for topo_idx, template in topo_items.items():
        topo_name = "unknown"
        for name, idx in TOPOLOGY_TO_IDX.items():
            if idx == topo_idx:
                topo_name = name
                break

        spec_types = template["spec_types"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        spec_values = template["spec_values"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        spec_mask = template["spec_mask"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        topo_t = template["topology_idx"].unsqueeze(0).expand(n_per_topo).to(device)
        active = template["active_mask"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        bmin = template["value_bounds_min"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        bmax = template["value_bounds_max"].unsqueeze(0).expand(n_per_topo, -1).to(device)

        with torch.no_grad():
            soft_X, soft_A, soft_V, _ = model.sample_with_projection(
                spec_types, spec_values, spec_mask, topo_t, active, bmin, bmax,
            )

        pred_types = soft_X.argmax(dim=-1)
        pred_adj = (soft_A > 0.5).float()
        pred_values = torch.clamp(soft_V, min=bmin, max=bmax)

        t_struct = 0
        t_sim = 0
        t_reward = 0.0
        graphs_to_simulate = []

        for b in range(n_per_topo):
            n_active = int(active[b].sum().item())
            graph = CircuitGraph(
                topology=topo_name, n_components=n_active,
                node_types=pred_types[b].cpu(), adjacency=pred_adj[b].cpu(),
                values=pred_values[b].cpu(), active_mask=active[b].cpu(),
                spec_types=spec_types[b].cpu(), spec_values=spec_values[b].cpu(),
                spec_mask=spec_mask[b].cpu(),
                value_bounds_min=bmin[b].cpu(), value_bounds_max=bmax[b].cpu(),
            )
            v = check_circuit_validity(graph)
            if v["valid"]:
                t_struct += 1
                graphs_to_simulate.append(graph)

        for graph in graphs_to_simulate:
            try:
                _, outcome, reward = vcg_graph_to_spice(graph, runner, tokenizer)
                if outcome.valid:
                    t_sim += 1
                    t_reward += reward
            except Exception:
                pass

        per_topology[topo_name] = {
            "n_generated": n_per_topo,
            "n_struct_valid": t_struct,
            "n_sim_valid": t_sim,
            "struct_valid_rate": t_struct / n_per_topo,
            "sim_valid_rate": t_sim / n_per_topo,
            "avg_reward": t_reward / max(t_sim, 1),
        }
        total_generated += n_per_topo
        total_struct_valid += t_struct
        total_sim_valid += t_sim
        total_reward += t_reward
        logger.info(
            f"  {topo_name:25s}: {t_sim}/{n_per_topo} sim_ok "
            f"({100*t_sim/n_per_topo:.0f}%) "
            f"struct {100*t_struct/n_per_topo:.0f}%"
        )

    runner.cleanup()

    return {
        "n_generated": total_generated,
        "n_struct_valid": total_struct_valid,
        "n_sim_valid": total_sim_valid,
        "struct_valid_rate": total_struct_valid / max(total_generated, 1),
        "sim_valid_rate": total_sim_valid / max(total_generated, 1),
        "avg_reward": total_reward / max(total_sim_valid, 1),
        "per_topology": per_topology,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════


def print_ccfm_report(
    struct_results: dict,
    spice_results: dict | None,
    n_params: int,
) -> None:
    print(f"\n{'='*70}")
    print("CCFM Evaluation Report")
    print(f"{'='*70}")
    print(f"\nModel: {n_params:,} parameters")

    print(f"\n{'─'*70}")
    print("1. STRUCTURAL VALIDITY")
    print(f"{'─'*70}")
    print(f"  Overall validity:    {struct_results['validity_rate']:.1%}")
    print(f"  Samples generated:   {struct_results['n_generated']}")

    if struct_results.get("per_topology"):
        print(f"\n  Per-topology breakdown:")
        for topo, d in sorted(struct_results["per_topology"].items()):
            bar = "█" * int(d["validity_rate"] * 10) + "░" * (10 - int(d["validity_rate"] * 10))
            print(f"    {topo:<25s}: {d['n_valid']:>3d}/{d['n_generated']:>3d} "
                  f"{bar} {d['validity_rate']:.0%}")

    if spice_results:
        print(f"\n{'─'*70}")
        print("2. UNIFIED SPICE SIMULATION EVALUATION")
        print(f"{'─'*70}")
        print(f"  Struct valid rate:   {spice_results['struct_valid_rate']:.1%}")
        print(f"  Sim valid rate:      {spice_results['sim_valid_rate']:.1%}")
        print(f"  Avg reward (sim ok): {spice_results['avg_reward']:.3f}")
        print(f"  n_sim_valid:         {spice_results['n_sim_valid']}/{spice_results['n_generated']}")

        if spice_results.get("per_topology"):
            print(f"\n  Per-topology SPICE breakdown:")
            for topo, d in sorted(spice_results["per_topology"].items()):
                print(
                    f"    {topo:<25s}: {d['n_sim_valid']:>3d}/{d['n_generated']:>3d} "
                    f"sim_ok ({d['sim_valid_rate']:.0%}) "
                    f"struct ({d['struct_valid_rate']:.0%})"
                )

    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CCFM model")
    parser.add_argument(
        "--ccfm-checkpoint", type=str, default="checkpoints/ccfm/best_ccfm.pt",
        help="Path to CCFM checkpoint (.pt)",
    )
    parser.add_argument(
        "--data", type=str, default="data/combined",
        help="Path to JSONL data directory",
    )
    parser.add_argument("--n-samples", type=int, default=340,
                        help="Total samples (spread evenly across topologies)")
    parser.add_argument("--output", type=str, default="results/ccfm_evaluation.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--spice", action="store_true",
        help="Run SPICE simulation on generated circuits (unified evaluation mode)",
    )
    parser.add_argument(
        "--n-spice-samples", type=int, default=None,
        help="Number of circuits to simulate (defaults to --n-samples)",
    )
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
    logger.info(f"Device: {device}")

    # Load CCFM
    logger.info(f"Loading CCFM from {args.ccfm_checkpoint}...")
    model = ConstrainedFlowMatchingModel.load(args.ccfm_checkpoint, device=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CCFM loaded: {n_params:,} params")

    # Load data
    tokenizer = CircuitTokenizer()
    vcg_config = model.flow_config.vcg_config
    logger.info(f"Loading data from {args.data}...")
    dataset = CircuitGraphDataset(args.data, tokenizer, vcg_config, valid_only=True)

    # Structural validity
    logger.info("\n[1] Evaluating structural validity...")
    struct_results = evaluate_structural_validity(model, dataset, args.n_samples, device)

    # Optional SPICE evaluation
    spice_results = None
    if args.spice:
        n_spice = args.n_spice_samples or args.n_samples
        logger.info(f"\n[2] Running unified SPICE evaluation (n={n_spice})...")
        spice_results = evaluate_spice_validity(model, dataset, n_spice, device)

    # Report
    print_ccfm_report(struct_results, spice_results, n_params)

    # Save
    all_results = {
        "n_params": n_params,
        "structural_validity": struct_results,
    }
    if spice_results:
        all_results["spice_evaluation"] = spice_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
