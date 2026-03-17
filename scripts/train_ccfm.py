#!/usr/bin/env python3
"""Train Constrained Circuit Flow Matching (CCFM).

Usage:
    python scripts/train_ccfm.py --data data/combined --vcg-checkpoint checkpoints/vcg/best_model.pt

This trains the CCFM model: a Conditional Flow Matching network operating in
the VCG latent space with constraint-guided ODE sampling. The VCG encoder/decoder
are loaded from a pre-trained checkpoint and frozen by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import random_split

from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    VCGConfig,
    ValidCircuitGenModel,
    CircuitGraphDataset,
    CircuitGraph,
    check_circuit_validity,
    LagrangianVAETrainer,
    TOPOLOGY_TO_IDX,
)
from arcs.flow_matching import (
    FlowMatchingConfig,
    ConstrainedFlowMatchingModel,
    train_flow_matching,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CCFM (Phase 13)")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data directory")
    parser.add_argument("--vcg-checkpoint", type=str, required=True,
                        help="Path to pre-trained VCG checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/ccfm", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent space dim")
    parser.add_argument("--flow-d-model", type=int, default=256, help="Flow network hidden dim")
    parser.add_argument("--flow-n-layers", type=int, default=4, help="Flow transformer layers")
    parser.add_argument("--n-sample-steps", type=int, default=50, help="ODE steps for sampling")
    parser.add_argument("--guidance-strength", type=float, default=1.0, help="Constraint guidance λ")
    parser.add_argument("--unfreeze-vcg", action="store_true", help="Fine-tune VCG encoder/decoder too")
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval-interval", type=int, default=20, help="Evaluate every N epochs")
    parser.add_argument("--eval-samples", type=int, default=50, help="Samples for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-valid-only", dest="valid_only", action="store_false",
                        help="Include invalid circuits (default: valid only)")
    parser.set_defaults(valid_only=True)
    return parser.parse_args()


def evaluate_ccfm_generation(
    model: ConstrainedFlowMatchingModel,
    dataset: CircuitGraphDataset,
    tokenizer: CircuitTokenizer,
    n_samples: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Evaluate CCFM generation: sample from flow, decode, check validity."""
    model.eval()

    # Collect unique topologies
    topo_items: dict[int, dict] = {}
    for i in range(min(len(dataset), 500)):
        item = dataset[i]
        topo_idx = item["topology_idx"].item()
        if topo_idx not in topo_items:
            topo_items[topo_idx] = item

    if not topo_items:
        return {"validity_rate": 0.0, "total_generated": 0, "total_valid": 0}

    total_valid = 0
    total_generated = 0

    for topo_idx, template_item in topo_items.items():
        per_topo = max(1, n_samples // len(topo_items))

        # Build spec conditioning
        spec_types = template_item["spec_types"].unsqueeze(0).to(device)
        spec_values = template_item["spec_values"].unsqueeze(0).to(device)
        spec_mask = template_item["spec_mask"].unsqueeze(0).to(device)
        topology_idx_t = template_item["topology_idx"].unsqueeze(0).to(device)
        active_mask = template_item["active_mask"].unsqueeze(0).to(device)
        bounds_min = template_item["value_bounds_min"].unsqueeze(0).to(device)
        bounds_max = template_item["value_bounds_max"].unsqueeze(0).to(device)

        # Repeat for n_samples
        spec_types = spec_types.expand(per_topo, -1)
        spec_values = spec_values.expand(per_topo, -1)
        spec_mask = spec_mask.expand(per_topo, -1)
        topology_idx_t = topology_idx_t.expand(per_topo)
        active_mask = active_mask.expand(per_topo, -1)
        bounds_min = bounds_min.expand(per_topo, -1)
        bounds_max = bounds_max.expand(per_topo, -1)

        with torch.no_grad():
            # Sample from flow + project
            soft_X, soft_A, soft_V, info = model.sample_with_projection(
                spec_types=spec_types,
                spec_values=spec_values,
                spec_mask=spec_mask,
                topology_idx=topology_idx_t,
                active_mask=active_mask,
                bounds_min=bounds_min,
                bounds_max=bounds_max,
            )

            # Discretize to CircuitGraphs (same as VCG.generate)
            pred_types = soft_X.argmax(dim=-1)
            pred_adj = (soft_A > 0.5).float()
            pred_values = torch.clamp(soft_V, min=bounds_min, max=bounds_max)

            # Look up topology name
            topo_name = "unknown"
            for name, idx in TOPOLOGY_TO_IDX.items():
                if idx == topo_idx:
                    topo_name = name
                    break

            for b in range(per_topo):
                n_active = int(active_mask[b].sum().item())
                graph = CircuitGraph(
                    topology=topo_name,
                    n_components=n_active,
                    node_types=pred_types[b].cpu(),
                    adjacency=pred_adj[b].cpu(),
                    values=pred_values[b].cpu(),
                    active_mask=active_mask[b].cpu(),
                    spec_types=spec_types[b].cpu(),
                    spec_values=spec_values[b].cpu(),
                    spec_mask=spec_mask[b].cpu(),
                    value_bounds_min=bounds_min[b].cpu(),
                    value_bounds_max=bounds_max[b].cpu(),
                )
                validity = check_circuit_validity(graph)
                if validity["valid"]:
                    total_valid += 1
                total_generated += 1

    validity_rate = total_valid / max(total_generated, 1)
    return {
        "validity_rate": validity_rate,
        "total_generated": total_generated,
        "total_valid": total_valid,
        "n_topologies": len(topo_items),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Tokenizer
    tokenizer = CircuitTokenizer()

    # Load pre-trained VCG model
    logger.info(f"Loading VCG from {args.vcg_checkpoint}")
    vcg_ckpt = torch.load(args.vcg_checkpoint, map_location="cpu", weights_only=False)
    vcg_config = VCGConfig.from_dict(vcg_ckpt["config"])
    vcg_model = ValidCircuitGenModel(vcg_config)
    vcg_model.load_state_dict(vcg_ckpt["model"])
    logger.info(f"VCG loaded: {vcg_model.count_parameters():,} params, val_loss={vcg_ckpt.get('val_loss', '?')}")

    # CCFM config
    flow_config = FlowMatchingConfig(
        latent_dim=args.latent_dim,
        flow_d_model=args.flow_d_model,
        flow_n_layers=args.flow_n_layers,
        spec_d_model=vcg_config.d_model,
        n_sample_steps=args.n_sample_steps,
        guidance_strength=args.guidance_strength,
        vcg_config=vcg_config,
    )

    # Create CCFM model with pre-trained VCG components
    model = ConstrainedFlowMatchingModel(flow_config, vcg_model=vcg_model)
    model = model.to(device)
    n_total = model.count_parameters()
    logger.info(f"CCFM model: {n_total:,} total parameters (device={device})")

    # Dataset
    logger.info(f"Loading data from {args.data}...")
    dataset = CircuitGraphDataset(
        args.data, tokenizer, vcg_config, valid_only=args.valid_only,
    )
    logger.info(f"Dataset: {len(dataset)} circuits")

    # Train/val split (90/10)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Train: {n_train}, Val: {n_val}")

    # Save config
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(flow_config.to_dict(), f, indent=2)

    # Train
    logger.info(f"\n{'='*60}")
    logger.info(f"Training CCFM for {args.epochs} epochs")
    logger.info(f"  Freeze VCG: {not args.unfreeze_vcg}")
    logger.info(f"  Flow layers: {args.flow_n_layers}, dim: {args.flow_d_model}")
    logger.info(f"  Guidance: λ={args.guidance_strength}, start_t=0.3")
    logger.info(f"{'='*60}\n")

    result = train_flow_matching(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_vcg=not args.unfreeze_vcg,
        device=device,
        output_dir=args.output,
        log_interval=args.log_interval,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"  Best val_loss: {result['best_val_loss']:.4f}")
    logger.info(f"  Final train_loss: {result['final_train_loss']:.4f}")
    logger.info(f"{'='*60}")

    # Evaluate generation quality
    logger.info("\nEvaluating CCFM generation quality...")
    best_model = ConstrainedFlowMatchingModel.load(
        output_dir / "best_ccfm.pt", device=device,
    )
    gen_stats = evaluate_ccfm_generation(
        best_model, dataset, tokenizer,
        n_samples=args.eval_samples, device=device,
    )
    logger.info(
        f"CCFM Validity: {gen_stats['validity_rate']*100:.1f}% "
        f"({gen_stats['total_valid']}/{gen_stats['total_generated']} "
        f"across {gen_stats['n_topologies']} topologies)"
    )


if __name__ == "__main__":
    main()
