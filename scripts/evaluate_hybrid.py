#!/usr/bin/env python3
"""Evaluate hybrid generation pipeline (VCG, CCFM, and ranked union)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from arcs.hybrid_pipeline import HybridGenerator, evaluate_generator, summarize_eval_results
from arcs.simulate import ALL_TEST_SPECS
from arcs.flow_matching import ConstrainedFlowMatchingModel
from arcs.valid_circuit_gen import ValidCircuitGenModel, VCGConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_vcg(path: str, device: torch.device) -> ValidCircuitGenModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = VCGConfig.from_dict(ckpt["config"])
    model = ValidCircuitGenModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid generator")
    parser.add_argument("--vcg", type=str, default="checkpoints/vcg/best_model.pt")
    parser.add_argument("--ccfm", type=str, default="checkpoints/ccfm/best_ccfm.pt")
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--output", type=str, default="results/hybrid_phase14.json")
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    logger.info(f"Device: {device}")

    if not Path(args.vcg).exists() or not Path(args.ccfm).exists():
        raise FileNotFoundError("Required checkpoints missing for hybrid evaluation")

    vcg_model = _load_vcg(args.vcg, device)
    ccfm_model = ConstrainedFlowMatchingModel.load(args.ccfm, device=device)
    ccfm_model.eval()

    hybrid = HybridGenerator(vcg_model=vcg_model, ccfm_model=ccfm_model, device=device)

    def vcg_fn(topology: str, specs: dict[str, float]):
        return hybrid.generate_from_vcg(topology, specs, n_candidates=args.n_candidates)

    def ccfm_fn(topology: str, specs: dict[str, float]):
        return hybrid.generate_from_ccfm(topology, specs, n_candidates=args.n_candidates)

    def ranked_fn(topology: str, specs: dict[str, float]):
        best = hybrid.generate_best(
            topology,
            specs,
            n_candidates_per_source=args.n_candidates,
            sources=["vcg", "ccfm"],
        )
        return [best]

    logger.info("Evaluating VCG-only...")
    vcg_res = evaluate_generator(vcg_fn, test_specs=ALL_TEST_SPECS, label="vcg")
    vcg_sum = summarize_eval_results(vcg_res, label="vcg")

    logger.info("Evaluating CCFM-only...")
    ccfm_res = evaluate_generator(ccfm_fn, test_specs=ALL_TEST_SPECS, label="ccfm")
    ccfm_sum = summarize_eval_results(ccfm_res, label="ccfm")

    logger.info("Evaluating Hybrid ranked union (VCG+CCFM)...")
    hybrid_res = evaluate_generator(ranked_fn, test_specs=ALL_TEST_SPECS, label="hybrid")
    hybrid_sum = summarize_eval_results(hybrid_res, label="hybrid")

    output = {
        "device": str(device),
        "n_candidates_per_source": args.n_candidates,
        "summary": {
            "vcg": vcg_sum,
            "ccfm": ccfm_sum,
            "hybrid": hybrid_sum,
        },
        "per_topology": {
            "vcg": {k: vars(v) for k, v in vcg_res.items()},
            "ccfm": {k: vars(v) for k, v in ccfm_res.items()},
            "hybrid": {k: vars(v) for k, v in hybrid_res.items()},
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    logger.info("Saved hybrid evaluation: %s", out_path)
    logger.info("Hybrid sim_valid=%.1f%%, reward=%.3f", 100 * hybrid_sum["sim_valid_rate"], hybrid_sum["mean_reward"])


if __name__ == "__main__":
    main()
