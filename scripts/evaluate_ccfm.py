#!/usr/bin/env python3
"""Evaluate the trained CCFM model on generation quality."""
import torch
import logging
from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig, CircuitGraphDataset, CircuitGraph,
    check_circuit_validity, TOPOLOGY_TO_IDX,
)
from arcs.flow_matching import ConstrainedFlowMatchingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load CCFM
    model = ConstrainedFlowMatchingModel.load("checkpoints/ccfm/best_ccfm.pt", device=device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"CCFM loaded: {n_params:,} params")

    # Load dataset for topology templates
    tokenizer = CircuitTokenizer()
    vcg_config = model.flow_config.vcg_config
    dataset = CircuitGraphDataset("data/combined", tokenizer, vcg_config, valid_only=True)

    # Collect unique topologies
    topo_items = {}
    for i in range(len(dataset)):
        item = dataset[i]
        tidx = item["topology_idx"].item()
        if tidx not in topo_items:
            topo_items[tidx] = item
        if len(topo_items) >= 17:  # all topologies found
            break

    logger.info(f"Found {len(topo_items)} topologies for evaluation")
    n_per_topo = 10

    total_valid = 0
    total_gen = 0
    per_topo_results = {}

    for topo_idx, template in topo_items.items():
        # Look up name
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
            v = check_circuit_validity(graph)
            if v["valid"]:
                topo_valid += 1
                total_valid += 1
            total_gen += 1

        rate = topo_valid / n_per_topo * 100
        per_topo_results[topo_name] = rate
        logger.info(f"  {topo_name:25s}: {topo_valid}/{n_per_topo} valid ({rate:.0f}%)")

    overall = total_valid / total_gen * 100
    logger.info(f"\nOverall CCFM validity: {total_valid}/{total_gen} = {overall:.1f}%")
    
    # Summary table
    logger.info(f"\n{'='*50}")
    logger.info(f"CCFM Generation Results (n={n_per_topo} per topology)")
    logger.info(f"{'='*50}")
    for name, rate in sorted(per_topo_results.items()):
        bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        logger.info(f"  {name:25s} {bar} {rate:5.1f}%")
    logger.info(f"  {'OVERALL':25s} {'='*10} {overall:5.1f}%")

if __name__ == "__main__":
    main()
