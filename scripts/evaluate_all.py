#!/usr/bin/env python3
"""Phase 13 Comprehensive Evaluation: Compare all generators.

Evaluates and compares:
  1. ARCS-SL (supervised GraphTransformer)
  2. ARCS-RL (REINFORCE, 5000 steps)
  3. ARCS-GRPO (Group Relative Policy Optimization, 500 steps)
  4. VCG (structural validity — graph-based VAE)
  5. CCFM (structural validity — flow matching)

For autoregressive models (1-3): generates circuits, decodes, simulates in SPICE.
For graph models (4-5): generates circuit graphs, checks structural validity.

Usage:
    PYTHONPATH=src python scripts/evaluate_all.py [--n-samples 80] [-v]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

from arcs.model_enhanced import load_model
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import generate_and_evaluate
from arcs.valid_circuit_gen import (
    VCGConfig, ValidCircuitGenModel, CircuitGraphDataset,
    CircuitGraph, check_circuit_validity, TOPOLOGY_TO_IDX,
)
from arcs.flow_matching import ConstrainedFlowMatchingModel
from arcs import DEFAULT_TEMPERATURE, DEFAULT_TOP_K


def evaluate_arcs_model(
    name: str,
    checkpoint: str,
    n_samples: int,
    device: torch.device,
    tokenizer: CircuitTokenizer,
    verbose: bool = False,
) -> dict | None:
    """Evaluate an autoregressive ARCS model with SPICE simulation."""
    if not Path(checkpoint).exists():
        print(f"  SKIP {name}: {checkpoint} not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*60}")

    try:
        model, config, mt = load_model(checkpoint, device=device)
    except RuntimeError as e:
        # Handle old checkpoints with different architecture
        print(f"  Warning: {e}")
        print(f"  Attempting lenient load...")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        
        # Check if this is an RL checkpoint (different format)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            print(f"  FAIL: Cannot find model state_dict in checkpoint")
            return None

        from arcs.model_enhanced import ARCSConfig, create_model
        config = ARCSConfig.from_dict(ckpt["config"])
        mt = ckpt.get("model_type", "graph_transformer")
        model = create_model(mt, config)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    t0 = time.time()
    results = generate_and_evaluate(
        model, tokenizer, device,
        n_samples=n_samples,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        conditioned=True,
        simulate=True,
    )
    dt = time.time() - t0

    out = {
        "name": name,
        "type": "autoregressive",
        "checkpoint": checkpoint,
        "n_params": n_params,
        "n_samples": n_samples,
        "eval_time_s": round(dt, 1),
        "valid_structure_rate": results.validity_rate,
        "sim_success_rate": results.sim_success_rate,
        "sim_valid_rate": results.sim_valid_rate,
        "mean_reward": results.avg_reward,
        "mean_efficiency": results.avg_efficiency,
        "mean_vout_error": results.avg_vout_error,
    }

    print(f"  Results:")
    print(f"    Structure valid: {out['valid_structure_rate']:.1%}")
    print(f"    Sim success:     {out['sim_success_rate']:.1%}")
    print(f"    Sim valid:       {out['sim_valid_rate']:.1%}")
    print(f"    Mean reward:     {out['mean_reward']:.3f}")
    print(f"    Time:            {dt:.1f}s")

    return out


def evaluate_graph_model(
    name: str,
    model_type: str,  # "vcg" or "ccfm"
    checkpoint: str,
    data_path: str,
    n_per_topo: int,
    device: torch.device,
    tokenizer: CircuitTokenizer,
) -> dict | None:
    """Evaluate a graph-based model (VCG or CCFM) on structural validity."""
    if not Path(checkpoint).exists():
        print(f"  SKIP {name}: {checkpoint} not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Evaluating: {name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*60}")

    t0 = time.time()

    if model_type == "vcg":
        vcg_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        vcg_config = VCGConfig.from_dict(vcg_ckpt["config"])
        model = ValidCircuitGenModel(vcg_config).to(device)
        model.load_state_dict(vcg_ckpt["model"])
        model.eval()
        n_params = model.count_parameters()

        dataset = CircuitGraphDataset(data_path, tokenizer, vcg_config, valid_only=True)
    elif model_type == "ccfm":
        model = ConstrainedFlowMatchingModel.load(checkpoint, device=device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        vcg_config = model.flow_config.vcg_config

        dataset = CircuitGraphDataset(data_path, tokenizer, vcg_config, valid_only=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"  Parameters: {n_params:,}")

    # Collect topology templates
    topo_items = {}
    for i in range(len(dataset)):
        item = dataset[i]
        tidx = item["topology_idx"].item()
        if tidx not in topo_items:
            topo_items[tidx] = item
        if len(topo_items) >= len(TOPOLOGY_TO_IDX):
            break

    total_valid = 0
    total_gen = 0
    per_topo = {}

    for topo_idx, template in topo_items.items():
        topo_name = "unknown"
        for name_k, idx in TOPOLOGY_TO_IDX.items():
            if idx == topo_idx:
                topo_name = name_k
                break

        spec_types = template["spec_types"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        spec_values = template["spec_values"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        spec_mask = template["spec_mask"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        topo_t = template["topology_idx"].unsqueeze(0).expand(n_per_topo).to(device)
        active = template["active_mask"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        bmin = template["value_bounds_min"].unsqueeze(0).expand(n_per_topo, -1).to(device)
        bmax = template["value_bounds_max"].unsqueeze(0).expand(n_per_topo, -1).to(device)

        with torch.no_grad():
            if model_type == "vcg":
                graphs, _ = model.generate(
                    spec_types=template["spec_types"].unsqueeze(0).to(device),
                    spec_values=template["spec_values"].unsqueeze(0).to(device),
                    spec_mask=template["spec_mask"].unsqueeze(0).to(device),
                    topology_idx=template["topology_idx"].unsqueeze(0).to(device),
                    active_mask=template["active_mask"].unsqueeze(0).to(device),
                    bounds_min=template["value_bounds_min"].unsqueeze(0).to(device),
                    bounds_max=template["value_bounds_max"].unsqueeze(0).to(device),
                    n_samples=n_per_topo,
                    use_projection=True,
                )
                topo_valid = sum(1 for g in graphs if check_circuit_validity(g)["valid"])
            else:
                # CCFM
                soft_X, soft_A, soft_V, info = model.sample_with_projection(
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
                    if check_circuit_validity(graph)["valid"]:
                        topo_valid += 1

        total_valid += topo_valid
        total_gen += n_per_topo
        per_topo[topo_name] = topo_valid / n_per_topo

    dt = time.time() - t0
    validity_rate = total_valid / max(total_gen, 1)

    n_full = sum(1 for v in per_topo.values() if v == 1.0)

    out = {
        "name": name,
        "type": "graph_" + model_type,
        "checkpoint": checkpoint,
        "n_params": n_params,
        "n_topologies": len(topo_items),
        "n_per_topology": n_per_topo,
        "total_generated": total_gen,
        "total_valid": total_valid,
        "validity_rate": validity_rate,
        "topologies_at_100pct": n_full,
        "per_topology": per_topo,
        "eval_time_s": round(dt, 1),
    }

    print(f"  Results:")
    print(f"    Structural validity: {validity_rate:.1%} ({total_valid}/{total_gen})")
    print(f"    Topologies at 100%:  {n_full}/{len(topo_items)}")
    print(f"    Time:                {dt:.1f}s")

    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 13: Compare all generators")
    parser.add_argument("--n-samples", type=int, default=80,
                        help="Samples per autoregressive model (cycles through specs)")
    parser.add_argument("--n-per-topo", type=int, default=10,
                        help="Samples per topology for graph models")
    parser.add_argument("--data", type=str, default="data/combined")
    parser.add_argument("--output", type=str, default="results/phase13_comparison.json")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    tokenizer = CircuitTokenizer()

    # ── Autoregressive models ──
    arcs_models = [
        ("ARCS-SL v3 (GraphTransformer)", "checkpoints/arcs_gt_v3/best_model.pt"),
        ("ARCS-GRPO v2 (3000 steps)", "checkpoints/arcs_grpo_v2/best_rl_model.pt"),
    ]

    # ── Graph models ──
    graph_models = [
        ("VCG v4 (VAE)", "vcg", "checkpoints/vcg_v4/best_model.pt"),
        ("CCFM v4 (Flow Matching)", "ccfm", "checkpoints/ccfm_v4/best_ccfm.pt"),
    ]

    all_results = []

    for name, ckpt in arcs_models:
        r = evaluate_arcs_model(name, ckpt, args.n_samples, device, tokenizer, args.verbose)
        if r:
            all_results.append(r)

    for name, model_type, ckpt in graph_models:
        r = evaluate_graph_model(name, model_type, ckpt, args.data, args.n_per_topo, device, tokenizer)
        if r:
            all_results.append(r)

    # ── Summary table ──
    print(f"\n\n{'='*80}")
    print(f"  PHASE 13 COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*80}\n")

    # Autoregressive models
    arcs_results = [r for r in all_results if r["type"] == "autoregressive"]
    if arcs_results:
        print(f"  AUTOREGRESSIVE MODELS (n={args.n_samples} circuits, SPICE simulation)")
        print(f"  {'Model':<30s} {'Params':>10s} {'Struct%':>8s} {'SimOK%':>8s} {'Valid%':>8s} {'Reward':>8s} {'Eff%':>8s} {'Verr%':>8s}")
        print(f"  {'-'*90}")
        for r in arcs_results:
            print(
                f"  {r['name']:<30s} "
                f"{r['n_params']/1e6:>8.1f}M "
                f"{r['valid_structure_rate']*100:>7.1f} "
                f"{r['sim_success_rate']*100:>7.1f} "
                f"{r['sim_valid_rate']*100:>7.1f} "
                f"{r['mean_reward']:>7.3f} "
                f"{r.get('mean_efficiency', 0)*100:>7.1f} "
                f"{r.get('mean_vout_error', 0):>7.1f} "
            )

    # Graph models
    graph_results = [r for r in all_results if r["type"].startswith("graph_")]
    if graph_results:
        print(f"\n  GRAPH MODELS (n={args.n_per_topo} per topology, structural validity)")
        print(f"  {'Model':<30s} {'Params':>10s} {'Valid%':>8s} {'100%Topos':>10s} {'Total':>12s}")
        print(f"  {'-'*76}")
        for r in graph_results:
            print(
                f"  {r['name']:<30s} "
                f"{r['n_params']/1e6:>8.1f}M "
                f"{r['validity_rate']*100:>7.1f} "
                f"{r['topologies_at_100pct']:>5d}/{r['n_topologies']:<4d} "
                f"{r['total_valid']:>5d}/{r['total_generated']:<5d} "
            )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
