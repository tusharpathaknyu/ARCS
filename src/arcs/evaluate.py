"""ARCS evaluation: validate generated circuits via SPICE simulation.

Evaluation metrics:
  1. Structural validity — % with proper START/TOPO/SEP/END structure
  2. Simulation success  — % that produce valid SPICE netlists and converge
  3. Simulation validity  — % physically plausible (eff>0, vout_err<50%, etc.)
  4. Spec compliance     — avg vout error, efficiency, gain, etc.
  5. Diversity           — uniqueness of topologies and component combos
  6. Quality             — average reward score

Usage:
    PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/best_model.pt
    PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/best_model.pt --simulate
    PYTHONPATH=src python -m arcs.evaluate --checkpoint checkpoints/best_model.pt --simulate --tier 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from arcs.model import ARCSModel, ARCSConfig
from arcs.model_enhanced import load_model
from arcs.tokenizer import CircuitTokenizer, TokenType
from arcs.train import generate_from_specs
from arcs.simulate import (
    simulate_decoded_circuit,
    compute_reward,
    normalize_topology,
    SimulationOutcome,
    TIER1_TEST_SPECS,
    TIER2_TEST_SPECS,
    ALL_TEST_SPECS,
)
from arcs.spice import NGSpiceRunner
from arcs.templates import _TIER1_NAMES, _TIER2_NAMES


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

    # Simulation metrics (only populated when --simulate is used)
    n_sim_attempted: int = 0
    n_sim_success: int = 0
    n_sim_valid: int = 0
    sim_success_rate: float = 0.0
    sim_valid_rate: float = 0.0
    avg_reward: float = 0.0
    avg_sim_time: float = 0.0

    # Power converter metrics (Tier 1)
    avg_vout_error: float = 0.0
    avg_efficiency: float = 0.0
    avg_ripple: float = 0.0
    n_power_simulated: int = 0

    # Signal circuit metrics (Tier 2)
    avg_gain_db: float = 0.0
    n_signal_simulated: int = 0

    # Per-topology sim breakdown
    per_topology_sim: dict[str, dict] = None  # type: ignore

    def to_dict(self) -> dict:
        d = {
            "n_generated": self.n_generated,
            "n_valid_structure": self.n_valid_structure,
            "validity_rate": self.validity_rate,
            "topology_distribution": self.topology_distribution,
            "avg_n_components": self.avg_n_components,
            "component_type_distribution": self.component_type_distribution,
            "unique_component_combos": self.unique_component_combos,
            "diversity_score": self.diversity_score,
        }
        if self.n_sim_attempted > 0:
            d.update({
                "n_sim_attempted": self.n_sim_attempted,
                "n_sim_success": self.n_sim_success,
                "n_sim_valid": self.n_sim_valid,
                "sim_success_rate": self.sim_success_rate,
                "sim_valid_rate": self.sim_valid_rate,
                "avg_reward": round(self.avg_reward, 3),
                "avg_sim_time": round(self.avg_sim_time, 3),
                "avg_vout_error": round(self.avg_vout_error, 2),
                "avg_efficiency": round(self.avg_efficiency, 4),
                "avg_ripple": round(self.avg_ripple, 4),
                "n_power_simulated": self.n_power_simulated,
                "avg_gain_db": round(self.avg_gain_db, 2),
                "n_signal_simulated": self.n_signal_simulated,
                "per_topology_sim": self.per_topology_sim,
            })
        return d


def evaluate_generated_circuits(
    circuits: list[DecodedCircuit],
    sim_results: list[SimulationOutcome] | None = None,
    rewards: list[float] | None = None,
) -> EvalResults:
    """Compute evaluation metrics over a batch of generated circuits.

    Args:
        circuits: List of decoded circuits
        sim_results: Optional list of simulation outcomes (same length as circuits)
        rewards: Optional list of reward scores
    """
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

    result = EvalResults(
        n_generated=n,
        n_valid_structure=n_valid,
        validity_rate=n_valid / max(n, 1),
        topology_distribution=dict(topo_counts),
        avg_n_components=float(avg_comp),
        component_type_distribution=dict(all_types),
        unique_component_combos=len(combos),
        diversity_score=diversity,
    )

    # --- Simulation metrics ---
    if sim_results is not None:
        assert len(sim_results) == n
        result.n_sim_attempted = n

        successes = [s for s in sim_results if s.success]
        valids = [s for s in sim_results if s.valid]
        result.n_sim_success = len(successes)
        result.n_sim_valid = len(valids)
        result.sim_success_rate = len(successes) / max(n, 1)
        result.sim_valid_rate = len(valids) / max(n, 1)
        result.avg_sim_time = (
            np.mean([s.sim_time for s in sim_results if s.sim_time > 0])
            if any(s.sim_time > 0 for s in sim_results) else 0.0
        )

        if rewards is not None:
            result.avg_reward = float(np.mean(rewards))

        # Power converter metrics
        power_verrs, power_effs, power_rips = [], [], []
        # Signal circuit metrics
        signal_gains = []

        # Per-topology breakdown
        topo_stats: dict[str, dict] = {}

        for circ, sim in zip(circuits, sim_results):
            topo = normalize_topology(circ.topology) if circ.topology else ""
            if topo not in topo_stats:
                topo_stats[topo] = {
                    "n": 0, "sim_success": 0, "sim_valid": 0,
                    "vout_errors": [], "efficiencies": [], "gains": [],
                    "rewards": [],
                }
            ts = topo_stats[topo]
            ts["n"] += 1

            if sim.success:
                ts["sim_success"] += 1
                if sim.valid:
                    ts["sim_valid"] += 1

                if topo in _TIER1_NAMES:
                    verr = sim.metrics.get("vout_error_pct")
                    eff = sim.metrics.get("efficiency")
                    rip = sim.metrics.get("ripple_ratio")
                    if verr is not None:
                        power_verrs.append(verr)
                        ts["vout_errors"].append(verr)
                    if eff is not None:
                        power_effs.append(eff)
                        ts["efficiencies"].append(eff)
                    if rip is not None:
                        power_rips.append(rip)
                else:
                    gain = sim.metrics.get("gain_db", sim.metrics.get("gain_dc"))
                    if gain is not None:
                        signal_gains.append(gain)
                        ts["gains"].append(gain)

        result.n_power_simulated = len(power_verrs)
        result.avg_vout_error = float(np.mean(power_verrs)) if power_verrs else 0.0
        result.avg_efficiency = float(np.mean(power_effs)) if power_effs else 0.0
        result.avg_ripple = float(np.mean(power_rips)) if power_rips else 0.0
        result.n_signal_simulated = len(signal_gains)
        result.avg_gain_db = float(np.mean(signal_gains)) if signal_gains else 0.0

        # Compress per-topology stats
        per_topo_summary: dict[str, dict] = {}
        for topo, ts in topo_stats.items():
            if not topo:
                continue
            summary: dict[str, Any] = {
                "n": ts["n"],
                "sim_success": ts["sim_success"],
                "sim_valid": ts["sim_valid"],
                "sim_success_rate": ts["sim_success"] / max(ts["n"], 1),
                "sim_valid_rate": ts["sim_valid"] / max(ts["n"], 1),
            }
            if ts["vout_errors"]:
                summary["avg_vout_error"] = round(float(np.mean(ts["vout_errors"])), 2)
            if ts["efficiencies"]:
                summary["avg_efficiency"] = round(float(np.mean(ts["efficiencies"])), 4)
            if ts["gains"]:
                summary["avg_gain_db"] = round(float(np.mean(ts["gains"])), 2)
            per_topo_summary[topo] = summary

        result.per_topology_sim = per_topo_summary

    return result


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
    simulate: bool = False,
    tier: int | None = None,
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
        simulate:    If True, run SPICE simulation on generated circuits
        tier:        Filter to tier (1=power, 2=signal, None=all)

    Returns:
        EvalResults with aggregate metrics (including simulation if enabled)
    """
    model.eval()
    circuits: list[DecodedCircuit] = []

    # Select test specs based on tier
    if tier == 1:
        test_specs = TIER1_TEST_SPECS
    elif tier == 2:
        test_specs = TIER2_TEST_SPECS
    else:
        test_specs = ALL_TEST_SPECS

    for i in range(n_samples):
        if conditioned:
            # Cycle through test specs
            topo, specs = test_specs[i % len(test_specs)]
            # Build prefix
            prefix_ids = [tokenizer.start_id]
            topo_key = f"TOPO_{topo.upper()}"
            # Handle sallen_key_lowpass → TOPO_SALLEN_KEY_LOWPASS
            # But tokenizer uses LP/HP/BP abbreviations
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
            prefix = torch.tensor([prefix_ids], device=device)
        else:
            # Unconditional: just START
            prefix = torch.tensor([[tokenizer.start_id]], device=device)

        output = model.generate(
            prefix,
            max_new_tokens=80,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
        )
        decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
        circuits.append(decoded)

    # Run SPICE simulation if requested
    sim_results: list[SimulationOutcome] | None = None
    rewards: list[float] | None = None

    if simulate:
        print(f"Simulating {len(circuits)} circuits...")
        runner = NGSpiceRunner()
        sim_results = []
        rewards = []
        n_done = 0
        for circ in circuits:
            outcome = simulate_decoded_circuit(circ, runner=runner)
            sim_results.append(outcome)

            # Compute reward
            target = circ.specs if circ.specs else None
            r = compute_reward(circ, outcome, target)
            rewards.append(r)

            n_done += 1
            if n_done % 20 == 0:
                n_success = sum(1 for s in sim_results if s.success)
                n_valid = sum(1 for s in sim_results if s.valid)
                print(
                    f"  [{n_done}/{len(circuits)}] "
                    f"sim_success={n_success}/{n_done} ({n_success/n_done:.0%}) "
                    f"sim_valid={n_valid}/{n_done} ({n_valid/n_done:.0%})"
                )

    return evaluate_generated_circuits(circuits, sim_results, rewards)


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
    parser.add_argument("--simulate", action="store_true",
                        help="Run SPICE simulation on generated circuits")
    parser.add_argument("--tier", type=int, default=None, choices=[1, 2],
                        help="Filter to tier (1=power, 2=signal, None=all)")
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

    # Load model (auto-detects model type from checkpoint)
    model, config, model_type = load_model(args.checkpoint, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loaded {model_type} model from {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    tokenizer = CircuitTokenizer()

    # Evaluate
    mode = "unconditioned" if args.unconditioned else "conditioned"
    tier_str = f"tier {args.tier}" if args.tier else "all tiers"
    sim_str = "+SPICE" if args.simulate else "structure-only"
    print(f"\nGenerating {args.n_samples} circuits ({mode}, {tier_str}, {sim_str})...")
    t0 = time.time()
    results = generate_and_evaluate(
        model, tokenizer, device,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioned=not args.unconditioned,
        simulate=args.simulate,
        tier=args.tier,
    )
    dt = time.time() - t0

    # Report
    print(f"\n{'=' * 60}")
    print(f"ARCS Evaluation Results ({dt:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Generated:          {results.n_generated}")
    print(f"Valid structure:     {results.n_valid_structure} ({results.validity_rate:.1%})")
    print(f"Avg components:     {results.avg_n_components:.1f}")
    print(f"Unique combos:      {results.unique_component_combos}")
    print(f"Diversity score:    {results.diversity_score:.3f}")

    if results.n_sim_attempted > 0:
        print(f"\n--- SPICE Simulation ---")
        print(f"Sim success:        {results.n_sim_success}/{results.n_sim_attempted} ({results.sim_success_rate:.1%})")
        print(f"Sim valid:          {results.n_sim_valid}/{results.n_sim_attempted} ({results.sim_valid_rate:.1%})")
        print(f"Avg reward:         {results.avg_reward:.3f}/8.0")
        print(f"Avg sim time:       {results.avg_sim_time:.2f}s")

        if results.n_power_simulated > 0:
            print(f"\n--- Power Converters (n={results.n_power_simulated}) ---")
            print(f"Avg Vout error:     {results.avg_vout_error:.1f}%")
            print(f"Avg efficiency:     {results.avg_efficiency:.1%}")
            print(f"Avg ripple ratio:   {results.avg_ripple:.4f}")

        if results.n_signal_simulated > 0:
            print(f"\n--- Signal Circuits (n={results.n_signal_simulated}) ---")
            print(f"Avg gain (dB):      {results.avg_gain_db:.1f}")

        if results.per_topology_sim:
            print(f"\n--- Per-Topology Breakdown ---")
            for topo in sorted(results.per_topology_sim.keys()):
                ts = results.per_topology_sim[topo]
                line = f"  {topo:24s}: {ts['n']:3d} gen, {ts['sim_success']:3d} sim_ok ({ts['sim_success_rate']:.0%}), {ts['sim_valid']:3d} valid ({ts['sim_valid_rate']:.0%})"
                if "avg_vout_error" in ts:
                    line += f", verr={ts['avg_vout_error']:.1f}%"
                if "avg_efficiency" in ts:
                    line += f", eff={ts['avg_efficiency']:.1%}"
                if "avg_gain_db" in ts:
                    line += f", gain={ts['avg_gain_db']:.1f}dB"
                print(line)

    print(f"\nTopology distribution:")
    for topo, count in sorted(results.topology_distribution.items()):
        print(f"  {topo}: {count}")
    print(f"\nComponent type distribution:")
    for comp, count in sorted(results.component_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  {comp}: {count}")

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
