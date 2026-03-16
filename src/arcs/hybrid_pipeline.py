"""Hybrid Circuit Generation Pipeline.

Bridges VCG/CCFM graph-based generators with ARCS's SPICE simulation
infrastructure, enabling:

1. **VCG→SPICE**: Convert VCG-generated CircuitGraphs to SPICE netlists,
   simulate, and obtain performance metrics.

2. **CCFM→SPICE**: Flow matching circuits through the same pipeline.

3. **Multi-Source Ranking**: Generate candidates from ARCS (autoregressive),
   VCG (constrained VAE), and/or CCFM (flow matching), rank them by
   SPICE-simulated performance, and return the best.

4. **Standardized Evaluation**: Evaluate any generator on a common benchmark
   of circuit specifications across all topologies.

Usage:
    from arcs.hybrid_pipeline import HybridGenerator, vcg_graph_to_spice

    # Single-source
    result = vcg_graph_to_spice(graph, runner)

    # Multi-source ranking
    hybrid = HybridGenerator(arcs_model=model, vcg_model=vcg)
    best = hybrid.generate_best(topology="buck", specs={"vin": 12, "vout": 5, ...})
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from arcs.evaluate import DecodedCircuit, decode_generated_sequence
from arcs.simulate import (
    simulate_decoded_circuit,
    compute_reward,
    SimulationOutcome,
    COMPONENT_TO_PARAM,
    TIER1_TEST_SPECS,
    TIER2_TEST_SPECS,
    ALL_TEST_SPECS,
)
from arcs.spice import NGSpiceRunner
from arcs.tokenizer import CircuitTokenizer
from arcs.valid_circuit_gen import (
    CircuitGraph,
    ValidCircuitGenModel,
    VCGConfig,
    graph_to_token_sequence,
    check_circuit_validity,
    TOPOLOGY_TO_IDX,
    NODE_TYPE_TO_IDX,
    IDX_TO_NODE_TYPE,
    SPEC_TO_IDX,
    SPEC_TYPES,
)
from arcs.model_enhanced import TOPOLOGY_ADJACENCY

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# VCG/CCFM → SPICE Pipeline
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GeneratedCircuit:
    """A generated circuit with optional simulation results."""

    source: str                           # "arcs", "vcg", "ccfm"
    topology: str
    graph: Optional[CircuitGraph] = None  # for VCG/CCFM
    decoded: Optional[DecodedCircuit] = None
    outcome: Optional[SimulationOutcome] = None
    reward: float = 0.0
    validity: Optional[dict] = None       # structural validity check
    gen_time_ms: float = 0.0


def _apply_topology_skeleton(graph: CircuitGraph) -> CircuitGraph:
    """Return a copy repaired to topology reference adjacency and node types."""
    adj_pairs = TOPOLOGY_ADJACENCY.get(graph.topology)
    comp_list = COMPONENT_TO_PARAM.get(graph.topology)
    if not adj_pairs or not comp_list:
        return graph

    n = graph.n_components
    new_adj = torch.zeros_like(graph.adjacency)
    for i, j in adj_pairs:
        if i < n and j < n:
            new_adj[i, j] = 1.0
            new_adj[j, i] = 1.0

    new_types = graph.node_types.clone()
    for i, (comp_type, _) in enumerate(comp_list[:n]):
        new_types[i] = NODE_TYPE_TO_IDX.get(comp_type, new_types[i].item())

    return CircuitGraph(
        topology=graph.topology,
        n_components=graph.n_components,
        node_types=new_types,
        adjacency=new_adj,
        values=graph.values,
        active_mask=graph.active_mask,
        spec_types=graph.spec_types,
        spec_values=graph.spec_values,
        spec_mask=graph.spec_mask,
        value_bounds_min=graph.value_bounds_min,
        value_bounds_max=graph.value_bounds_max,
    )


def vcg_graph_to_spice(
    graph: CircuitGraph,
    runner: NGSpiceRunner,
    tokenizer: Optional[CircuitTokenizer] = None,
) -> tuple[DecodedCircuit, SimulationOutcome, float]:
    """Convert a VCG/CCFM CircuitGraph to SPICE simulation results.

    Pipeline: CircuitGraph → token sequence → DecodedCircuit → SPICE → reward

    Args:
        graph: Generated circuit graph
        runner: ngspice runner instance
        tokenizer: CircuitTokenizer (created if not provided)

    Returns:
        decoded: DecodedCircuit with topology, components, etc.
        outcome: SimulationOutcome with metrics
        reward: scalar reward score
    """
    if tokenizer is None:
        tokenizer = CircuitTokenizer()

    # Convert graph → tokens → decoded circuit
    token_ids = graph_to_token_sequence(graph, tokenizer)
    decoded = decode_generated_sequence(token_ids, tokenizer)

    # Simulate
    if decoded.valid_structure:
        outcome = simulate_decoded_circuit(decoded, runner)
        reward = compute_reward(decoded, outcome)
    else:
        outcome = SimulationOutcome(
            success=False,
            valid=False,
            metrics={},
            error="Invalid decoded structure",
            sim_time=0.0,
        )
        reward = 0.0

    return decoded, outcome, reward


def _prepare_vcg_input(
    topology: str,
    specs: dict[str, float],
    config: VCGConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build VCG/CCFM input tensors from topology + specs.

    Creates the batch dict needed for VCG.generate() or CCFM.sample().
    """
    from arcs.simulate import COMPONENT_TO_PARAM
    from arcs.templates import POWER_CONVERTER_BOUNDS, SIGNAL_CIRCUIT_BOUNDS
    from arcs.valid_circuit_gen import (
        _ALL_BOUNDS, NODE_TYPE_TO_IDX, TOPOLOGY_TO_IDX,
        TOPOLOGY_ADJACENCY, LOG_VAL_MIN, LOG_VAL_MAX,
    )
    import math

    N = config.max_nodes
    S = config.max_specs

    # Topology
    topo_idx = TOPOLOGY_TO_IDX.get(topology, 0)
    topology_idx = torch.tensor([topo_idx], dtype=torch.long, device=device)

    # Get component list for this topology
    comp_list = COMPONENT_TO_PARAM.get(topology, [])
    n_comp = min(len(comp_list), N)

    # Node types
    node_types = torch.zeros(1, N, dtype=torch.long, device=device)
    for i, (comp_type, _) in enumerate(comp_list[:N]):
        idx = NODE_TYPE_TO_IDX.get(comp_type, 0)
        node_types[0, i] = idx

    # Active mask
    active_mask = torch.zeros(1, N, device=device)
    active_mask[0, :n_comp] = 1.0

    # Specs
    spec_types = torch.zeros(1, S, dtype=torch.long, device=device)
    spec_values = torch.zeros(1, S, device=device)
    spec_mask = torch.zeros(1, S, device=device)

    _OC_TO_SPEC = {
        "vin": "vin", "vout": "vout", "iout": "iout", "fsw": "fsw",
        "vin_amp": "vin", "freq_test": "cutoff_freq", "vcc": "vin",
    }
    si = 0
    for key, val in specs.items():
        spec_name = _OC_TO_SPEC.get(key, key)
        if spec_name in SPEC_TO_IDX and si < S:
            spec_types[0, si] = SPEC_TO_IDX[spec_name]
            spec_values[0, si] = math.log10(abs(val)) if val != 0 else 0.0
            spec_mask[0, si] = 1.0
            si += 1

    # Value bounds
    bounds_min = torch.full((1, N), LOG_VAL_MIN, device=device)
    bounds_max = torch.full((1, N), LOG_VAL_MAX, device=device)
    bounds_list = _ALL_BOUNDS.get(topology)
    if bounds_list:
        for i, (_, param_name) in enumerate(comp_list[:N]):
            for b in bounds_list:
                if b.name == param_name:
                    bounds_min[0, i] = math.log10(max(b.min_val, 1e-15))
                    bounds_max[0, i] = math.log10(max(b.max_val, 1e-15))
                    break

    return {
        "topology_idx": topology_idx,
        "node_types": node_types,
        "active_mask": active_mask,
        "spec_types": spec_types,
        "spec_values": spec_values,
        "spec_mask": spec_mask,
        "value_bounds_min": bounds_min,
        "value_bounds_max": bounds_max,
        "values": torch.zeros(1, N, device=device),
        "adjacency": torch.zeros(1, N, N, device=device),
        "n_components": torch.tensor([n_comp], dtype=torch.long, device=device),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Generator
# ═══════════════════════════════════════════════════════════════════════════


class HybridGenerator:
    """Multi-source circuit generator with SPICE-based ranking.

    Generates circuit candidates from one or more sources:
        - ARCS: autoregressive token generation (fast, flexible)
        - VCG:  constrained VAE (guaranteed-structure)
        - CCFM: constrained flow matching (novel, high quality)

    Candidates are ranked by SPICE simulation reward, and the best
    is returned. This exploits the complementary strengths of each
    approach.
    """

    def __init__(
        self,
        arcs_model=None,
        vcg_model: Optional[ValidCircuitGenModel] = None,
        ccfm_model=None,  # ConstrainedFlowMatchingModel
        reward_model=None,     # learned reward model for fast pre-filtering
        device: Optional[torch.device] = None,
    ):
        self.arcs_model = arcs_model
        self.vcg_model = vcg_model
        self.ccfm_model = ccfm_model
        self.reward_model = reward_model

        self.tokenizer = CircuitTokenizer()
        self.runner = NGSpiceRunner()

        if device is None:
            device = torch.device("cpu")
        self.device = device

    def _score_candidate_proxy(self, candidate: GeneratedCircuit) -> float:
        """Cheap proxy score used to pre-rank candidates before SPICE.

        Priority order:
          1) learned reward model (if provided and graph available)
          2) structural + template-consistency heuristic
        """
        if (
            self.reward_model is not None
            and candidate.graph is not None
        ):
            try:
                token_ids = graph_to_token_sequence(candidate.graph, self.tokenizer)
                inp = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                pred = self.reward_model.predict(inp)
                return float(pred.squeeze().item())
            except (ValueError, RuntimeError, IndexError) as e:
                logger.debug("Reward model prediction failed for %s: %s", candidate.topology, e)

        score = 0.0
        decoded = candidate.decoded
        if decoded is not None and decoded.valid_structure:
            score += 1.0

        expected = COMPONENT_TO_PARAM.get(candidate.topology, [])
        expected_n = len(expected)
        observed_n = len(decoded.components) if decoded is not None else 0
        if expected_n > 0:
            score += 1.0 - min(abs(observed_n - expected_n) / expected_n, 1.0)

        if decoded is not None and decoded.specs:
            n_specs = sum(1 for k in ["vin", "vout", "iout", "fsw"] if k in decoded.specs)
            score += n_specs / 4.0

        return float(score)

    def _simulate_candidate(self, candidate: GeneratedCircuit) -> GeneratedCircuit:
        """Simulate candidate in-place (if not simulated yet) and return it."""
        if candidate.outcome is not None:
            return candidate
        if candidate.graph is None:
            return candidate

        decoded, outcome, reward = vcg_graph_to_spice(
            candidate.graph, self.runner, self.tokenizer
        )
        candidate.decoded = decoded
        candidate.outcome = outcome
        candidate.reward = reward
        return candidate

    def generate_from_vcg(
        self,
        topology: str,
        specs: dict[str, float],
        n_candidates: int = 8,
        temperature: float = 1.0,
        simulate: bool = True,
    ) -> list[GeneratedCircuit]:
        """Generate circuits via VCG."""
        if self.vcg_model is None:
            return []

        config = self.vcg_model.config
        batch = _prepare_vcg_input(topology, specs, config, self.device)

        t0 = time.time()
        graphs, _ = self.vcg_model.generate(
            batch["spec_types"], batch["spec_values"], batch["spec_mask"],
            batch["topology_idx"], batch["active_mask"],
            batch["value_bounds_min"], batch["value_bounds_max"],
            n_samples=n_candidates,
            temperature=temperature,
        )
        gen_time = (time.time() - t0) * 1000

        results = []
        for graph in graphs:
            validity = check_circuit_validity(graph)
            if not validity["valid"]:
                repaired = _apply_topology_skeleton(graph)
                repaired_validity = check_circuit_validity(repaired)
                if repaired_validity["valid"]:
                    graph = repaired
                    validity = repaired_validity
            if simulate:
                decoded, outcome, reward = vcg_graph_to_spice(
                    graph, self.runner, self.tokenizer
                )
            else:
                token_ids = graph_to_token_sequence(graph, self.tokenizer)
                decoded = decode_generated_sequence(token_ids, self.tokenizer)
                outcome = None
                reward = 0.0
            results.append(GeneratedCircuit(
                source="vcg",
                topology=topology,
                graph=graph,
                decoded=decoded,
                outcome=outcome,
                reward=reward,
                validity=validity,
                gen_time_ms=gen_time / n_candidates,
            ))

        return results

    def generate_from_ccfm(
        self,
        topology: str,
        specs: dict[str, float],
        n_candidates: int = 8,
        temperature: float = 1.0,
        n_steps: int = 50,
        simulate: bool = True,
    ) -> list[GeneratedCircuit]:
        """Generate circuits via CCFM (Constrained Flow Matching)."""
        if self.ccfm_model is None:
            return []

        vcg_config = self.ccfm_model.flow_config.vcg_config
        if vcg_config is None:
            vcg_config = VCGConfig()
        batch = _prepare_vcg_input(topology, specs, vcg_config, self.device)

        # Repeat for n_candidates
        batch_rep = {
            k: v.repeat(n_candidates, *([1] * (v.dim() - 1)))
            for k, v in batch.items()
        }

        t0 = time.time()
        soft_X, soft_A, soft_V, info = self.ccfm_model.sample_with_projection(
            batch_rep["spec_types"], batch_rep["spec_values"],
            batch_rep["spec_mask"], batch_rep["topology_idx"],
            batch_rep["active_mask"], batch_rep["value_bounds_min"],
            batch_rep["value_bounds_max"],
            n_steps=n_steps,
            temperature=temperature,
        )
        gen_time = (time.time() - t0) * 1000

        # Discretize and create CircuitGraphs
        pred_types = soft_X.argmax(dim=-1)
        pred_adj = (soft_A > 0.5).float()
        pred_values = torch.clamp(
            soft_V,
            min=batch_rep["value_bounds_min"],
            max=batch_rep["value_bounds_max"],
        )

        results = []
        for b in range(n_candidates):
            n_active = int(batch_rep["active_mask"][b].sum().item())
            graph = CircuitGraph(
                topology=topology,
                n_components=n_active,
                node_types=pred_types[b].cpu(),
                adjacency=pred_adj[b].cpu(),
                values=pred_values[b].cpu(),
                active_mask=batch_rep["active_mask"][b].cpu(),
                spec_types=batch_rep["spec_types"][b].cpu(),
                spec_values=batch_rep["spec_values"][b].cpu(),
                spec_mask=batch_rep["spec_mask"][b].cpu(),
                value_bounds_min=batch_rep["value_bounds_min"][b].cpu(),
                value_bounds_max=batch_rep["value_bounds_max"][b].cpu(),
            )

            validity = check_circuit_validity(graph)
            if not validity["valid"]:
                repaired = _apply_topology_skeleton(graph)
                repaired_validity = check_circuit_validity(repaired)
                if repaired_validity["valid"]:
                    graph = repaired
                    validity = repaired_validity
            if simulate:
                decoded, outcome, reward = vcg_graph_to_spice(
                    graph, self.runner, self.tokenizer
                )
            else:
                token_ids = graph_to_token_sequence(graph, self.tokenizer)
                decoded = decode_generated_sequence(token_ids, self.tokenizer)
                outcome = None
                reward = 0.0
            results.append(GeneratedCircuit(
                source="ccfm",
                topology=topology,
                graph=graph,
                decoded=decoded,
                outcome=outcome,
                reward=reward,
                validity=validity,
                gen_time_ms=gen_time / n_candidates,
            ))

        return results

    def generate_best(
        self,
        topology: str,
        specs: dict[str, float],
        n_candidates_per_source: int = 8,
        sources: Optional[list[str]] = None,
        pre_rank_top_k: Optional[int] = None,
    ) -> GeneratedCircuit:
        """Generate from all available sources and return the best circuit.

        Args:
            topology: target circuit topology
            specs: target specifications
            n_candidates_per_source: candidates per generator
            sources: subset of ["arcs", "vcg", "ccfm"] (default: all available)

        Returns:
            Best GeneratedCircuit by reward score
        """
        if sources is None:
            sources = []
            if self.arcs_model is not None:
                sources.append("arcs")
            if self.vcg_model is not None:
                sources.append("vcg")
            if self.ccfm_model is not None:
                sources.append("ccfm")

        all_candidates: list[GeneratedCircuit] = []

        use_prerank = pre_rank_top_k is not None and pre_rank_top_k > 0

        if "vcg" in sources:
            all_candidates.extend(
                self.generate_from_vcg(
                    topology,
                    specs,
                    n_candidates_per_source,
                    simulate=not use_prerank,
                )
            )

        if "ccfm" in sources:
            all_candidates.extend(
                self.generate_from_ccfm(
                    topology,
                    specs,
                    n_candidates_per_source,
                    simulate=not use_prerank,
                )
            )

        if not all_candidates:
            return GeneratedCircuit(
                source="none", topology=topology, reward=0.0
            )

        if use_prerank and len(all_candidates) > pre_rank_top_k:
            for cand in all_candidates:
                cand.reward = self._score_candidate_proxy(cand)

            all_candidates.sort(key=lambda c: c.reward, reverse=True)
            shortlisted = all_candidates[: pre_rank_top_k]
            all_candidates = [self._simulate_candidate(c) for c in shortlisted]

        # Rank by (true) reward
        all_candidates.sort(key=lambda c: c.reward, reverse=True)
        best = all_candidates[0]

        logger.info(
            f"HybridGenerator: {len(all_candidates)} candidates from "
            f"{sources}, best={best.source} reward={best.reward:.3f}"
        )

        return best


# ═══════════════════════════════════════════════════════════════════════════
# Standardized Evaluation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    """Evaluation results for a single topology."""
    topology: str
    n_generated: int = 0
    n_struct_valid: int = 0
    n_sim_success: int = 0
    n_sim_valid: int = 0
    mean_reward: float = 0.0
    mean_efficiency: float = 0.0
    mean_vout_error: float = 100.0
    gen_time_ms: float = 0.0


def evaluate_generator(
    generator_fn,
    test_specs: Optional[dict] = None,
    n_per_spec: int = 10,
    label: str = "generator",
) -> dict[str, EvalResult]:
    """Standardized evaluation benchmark for any circuit generator.

    Args:
        generator_fn: callable(topology, specs) → list[GeneratedCircuit]
        test_specs: dict of topology → list[spec_dicts] (default: ALL_TEST_SPECS)
        n_per_spec: generate this many per spec
        label: name for logging

    Returns:
        dict of topology → EvalResult
    """
    if test_specs is None:
        test_specs = ALL_TEST_SPECS

    # Normalize list-form specs [(topology, spec_dict), ...] into
    # dict-form {topology: [spec_dict, ...]}.
    if isinstance(test_specs, list):
        grouped_specs: dict[str, list[dict[str, float]]] = defaultdict(list)
        for topology, spec in test_specs:
            grouped_specs[topology].append(spec)
        test_specs = grouped_specs

    results: dict[str, EvalResult] = {}

    for topology, specs_list in test_specs.items():
        topo_result = EvalResult(topology=topology)
        rewards = []
        efficiencies = []
        vout_errors = []

        for specs in specs_list:
            circuits = generator_fn(topology, specs)

            for circ in circuits:
                topo_result.n_generated += 1
                topo_result.gen_time_ms += circ.gen_time_ms

                if circ.decoded and circ.decoded.valid_structure:
                    topo_result.n_struct_valid += 1

                if circ.outcome and circ.outcome.success:
                    topo_result.n_sim_success += 1
                    if circ.outcome.valid:
                        topo_result.n_sim_valid += 1

                    eff = circ.outcome.metrics.get("efficiency", 0)
                    verr = circ.outcome.metrics.get("vout_error_pct", 100)
                    efficiencies.append(eff)
                    vout_errors.append(verr)

                rewards.append(circ.reward)

        if rewards:
            topo_result.mean_reward = float(np.mean(rewards))
        if efficiencies:
            topo_result.mean_efficiency = float(np.mean(efficiencies))
        if vout_errors:
            topo_result.mean_vout_error = float(np.mean(vout_errors))

        n = topo_result.n_generated or 1
        logger.info(
            f"[{label}] {topology}: "
            f"struct={topo_result.n_struct_valid / n:.0%} "
            f"sim_ok={topo_result.n_sim_success / n:.0%} "
            f"sim_valid={topo_result.n_sim_valid / n:.0%} "
            f"reward={topo_result.mean_reward:.3f}"
        )

        results[topology] = topo_result

    return results


def summarize_eval_results(
    results: dict[str, EvalResult],
    label: str = "",
) -> dict[str, float]:
    """Aggregate per-topology results into summary metrics."""
    total_gen = sum(r.n_generated for r in results.values())
    total_struct = sum(r.n_struct_valid for r in results.values())
    total_sim = sum(r.n_sim_success for r in results.values())
    total_valid = sum(r.n_sim_valid for r in results.values())

    n = max(total_gen, 1)
    summary = {
        "n_topologies": len(results),
        "n_generated": total_gen,
        "struct_valid_rate": total_struct / n,
        "sim_success_rate": total_sim / n,
        "sim_valid_rate": total_valid / n,
        "mean_reward": float(np.mean([r.mean_reward for r in results.values()])),
        "mean_gen_time_ms": float(np.mean(
            [r.gen_time_ms / max(r.n_generated, 1) for r in results.values()]
        )),
    }

    if label:
        logger.info(f"Summary [{label}]: {summary}")

    return summary
