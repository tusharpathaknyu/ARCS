"""ARCS Demo CLI — interactive circuit design from specifications.

Generate a complete analog circuit from specs in ~20ms:

    $ python -m arcs.demo
    $ python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1
    $ python -m arcs.demo --topology boost --vin 5 --vout 12 --iout 0.5 --simulate
    $ python -m arcs.demo --interactive

Examples:
    # Quick design
    PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1

    # With SPICE validation
    PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1 --simulate

    # Interactive mode
    PYTHONPATH=src python -m arcs.demo --interactive

    # Multiple candidates
    PYTHONPATH=src python -m arcs.demo --topology boost --vin 5 --vout 12 --iout 0.5 -n 5 --simulate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from arcs.model import ARCSModel, ARCSConfig
from arcs.tokenizer import CircuitTokenizer
from arcs.evaluate import decode_generated_sequence, DecodedCircuit
from arcs.simulate import (
    simulate_decoded_circuit,
    compute_reward,
    normalize_topology,
    SimulationOutcome,
    _TIER1_NAMES,
)
from arcs.spice import NGSpiceRunner
from arcs.templates import get_topology, _TIER2_NAMES


# ---------------------------------------------------------------------------
# Available topologies and their spec keys
# ---------------------------------------------------------------------------

TOPOLOGY_SPECS = {
    # Tier 1: Power converters
    "buck":       {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    "boost":      {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    "buck_boost": {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    "cuk":        {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    "sepic":      {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    "flyback":    {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    "forward":    {"required": ["vin", "vout", "iout"], "optional": ["fsw"]},
    # Tier 2: Amplifiers
    "inverting_amp":       {"required": ["vin"], "optional": ["cutoff_freq"]},
    "noninverting_amp":    {"required": ["vin"], "optional": ["cutoff_freq"]},
    "instrumentation_amp": {"required": ["vin"], "optional": ["cutoff_freq"]},
    "differential_amp":    {"required": ["vin"], "optional": ["cutoff_freq"]},
    # Tier 2: Filters
    "sallen_key_lowpass":  {"required": [], "optional": ["cutoff_freq"]},
    "sallen_key_highpass": {"required": [], "optional": ["cutoff_freq"]},
    "sallen_key_bandpass": {"required": [], "optional": ["cutoff_freq"]},
    # Tier 2: Oscillators
    "wien_bridge": {"required": [], "optional": []},
    "colpitts":    {"required": [], "optional": ["vin"]},
}


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[ARCSModel, CircuitTokenizer]:
    """Load the ARCS model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ARCSConfig.from_dict(checkpoint["config"])
    model = ARCSModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, CircuitTokenizer()


def generate_circuit(
    model: ARCSModel,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    topology: str,
    specs: dict[str, float],
    temperature: float = 0.8,
    top_k: int = 50,
) -> tuple[DecodedCircuit, float]:
    """Generate a single circuit and return (decoded, inference_time_ms)."""
    # Build spec prefix
    prefix_ids = [tokenizer.start_id]

    # Handle topology token naming
    _topo_to_token = {
        "sallen_key_lowpass": "TOPO_SALLEN_KEY_LP",
        "sallen_key_highpass": "TOPO_SALLEN_KEY_HP",
        "sallen_key_bandpass": "TOPO_SALLEN_KEY_BP",
    }
    topo_key = _topo_to_token.get(topology, f"TOPO_{topology.upper()}")

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

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            prefix,
            max_new_tokens=80,
            temperature=temperature,
            top_k=top_k,
        )
    dt_ms = (time.perf_counter() - t0) * 1000

    decoded = decode_generated_sequence(output[0].tolist(), tokenizer)
    return decoded, dt_ms


def format_circuit(
    decoded: DecodedCircuit,
    sim_outcome: Optional[SimulationOutcome] = None,
    reward: Optional[float] = None,
    inference_ms: float = 0.0,
) -> str:
    """Format a generated circuit for display."""
    lines = []
    lines.append(f"  Topology:   {decoded.topology}")
    lines.append(f"  Structure:  {'VALID' if decoded.valid_structure else 'INVALID'}")

    if decoded.specs:
        specs_str = ", ".join(f"{k}={v:.4g}" for k, v in decoded.specs.items())
        lines.append(f"  Specs:      {specs_str}")

    if decoded.components:
        lines.append(f"  Components ({len(decoded.components)}):")
        for comp_type, comp_val in decoded.components:
            lines.append(f"    {comp_type:<16s} = {_format_value(comp_val)}")

    lines.append(f"  Inference:  {inference_ms:.1f}ms")

    if sim_outcome is not None:
        lines.append("")
        if not sim_outcome.success:
            lines.append(f"  SPICE:      FAILED ({sim_outcome.error})")
        else:
            lines.append(f"  SPICE:      {'VALID' if sim_outcome.valid else 'CONVERGED (invalid)'}")
            lines.append(f"  Sim time:   {sim_outcome.sim_time:.2f}s")

            topo = normalize_topology(decoded.topology) if decoded.topology else ""
            if topo in _TIER1_NAMES:
                m = sim_outcome.metrics
                lines.append(f"  Vout error: {m.get('vout_error_pct', '?'):.1f}%")
                lines.append(f"  Efficiency: {m.get('efficiency', 0):.1%}")
                lines.append(f"  Ripple:     {m.get('ripple_ratio', 0):.4f}")
            else:
                m = sim_outcome.metrics
                gain = m.get("gain_db", m.get("gain_dc"))
                if gain is not None:
                    lines.append(f"  Gain:       {gain:.1f} dB")
                bw = m.get("bw_3db")
                if bw is not None:
                    lines.append(f"  Bandwidth:  {bw:.0f} Hz")
                vosc = m.get("vosc_pp")
                if vosc is not None:
                    lines.append(f"  Osc Vpp:    {vosc:.3f} V")

    if reward is not None:
        lines.append(f"  Reward:     {reward:.2f}/8.0")

    return "\n".join(lines)


def _format_value(val: float) -> str:
    """Format a component value with SI prefix."""
    prefixes = [
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "μ"),
        (1e-3, "m"),
        (1, ""),
        (1e3, "k"),
        (1e6, "M"),
    ]
    if val == 0:
        return "0"
    abs_val = abs(val)
    for threshold, prefix in reversed(prefixes):
        if abs_val >= threshold:
            scaled = val / threshold
            if abs(scaled - round(scaled)) < 0.01:
                return f"{round(scaled)}{prefix}"
            return f"{scaled:.2g}{prefix}"
    return f"{val:.4g}"


def interactive_mode(
    model: ARCSModel,
    tokenizer: CircuitTokenizer,
    device: torch.device,
    do_simulate: bool = False,
):
    """Interactive REPL for circuit generation."""
    runner = NGSpiceRunner() if do_simulate else None

    print("\n" + "=" * 60)
    print("  ARCS Interactive Circuit Designer")
    print("  Type 'help' for commands, 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            line = input("\narcs> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        if line.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if line.lower() == "help":
            print("""
Commands:
  <topology> [key=value ...]  Generate a circuit
    Example: buck vin=12 vout=5 iout=1
    Example: boost vin=5 vout=12 iout=0.5 fsw=200000
    Example: inverting_amp vin=0.1
    Example: colpitts

  list          Show available topologies
  simulate on   Enable SPICE simulation
  simulate off  Disable SPICE simulation
  quit          Exit
""")
            continue

        if line.lower() == "list":
            print("\nAvailable topologies:")
            print("  Power converters: buck, boost, buck_boost, cuk, sepic, flyback, forward")
            print("  Amplifiers:       inverting_amp, noninverting_amp, instrumentation_amp, differential_amp")
            print("  Filters:          sallen_key_lowpass, sallen_key_highpass, sallen_key_bandpass")
            print("  Oscillators:      wien_bridge, colpitts")
            continue

        if line.lower() == "simulate on":
            runner = NGSpiceRunner()
            do_simulate = True
            print("  SPICE simulation: ON")
            continue
        if line.lower() == "simulate off":
            runner = None
            do_simulate = False
            print("  SPICE simulation: OFF")
            continue

        # Parse: topology [key=value ...]
        parts = line.split()
        topo = parts[0].lower()
        if topo not in TOPOLOGY_SPECS:
            print(f"  Unknown topology '{topo}'. Type 'list' for options.")
            continue

        specs: dict[str, float] = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    specs[k.lower()] = float(v)
                except ValueError:
                    print(f"  Invalid value: {part}")
                    continue

        # Check required specs
        missing = [s for s in TOPOLOGY_SPECS[topo]["required"] if s not in specs]
        if missing:
            # Use defaults for power converters
            defaults = {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}
            for m in missing:
                if m in defaults:
                    specs[m] = defaults[m]
                    print(f"  Using default {m}={defaults[m]}")

        # Generate
        decoded, dt_ms = generate_circuit(model, tokenizer, device, topo, specs)

        # Simulate if enabled
        sim_outcome = None
        reward = None
        if do_simulate and decoded.valid_structure:
            sim_outcome = simulate_decoded_circuit(decoded, runner=runner)
            reward = compute_reward(decoded, sim_outcome, specs)

        print(f"\n--- Generated Circuit ({dt_ms:.0f}ms) ---")
        print(format_circuit(decoded, sim_outcome, reward, dt_ms))
        print("---")


def main():
    parser = argparse.ArgumentParser(
        description="ARCS Demo — Generate circuits from specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick buck converter
  PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1

  # With SPICE simulation
  PYTHONPATH=src python -m arcs.demo --topology boost --vin 5 --vout 12 --iout 0.5 --simulate

  # Multiple candidates, pick the best
  PYTHONPATH=src python -m arcs.demo --topology buck --vin 12 --vout 5 --iout 1 -n 5 --simulate

  # Interactive mode
  PYTHONPATH=src python -m arcs.demo --interactive
""",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/arcs_rl_v2/best_rl_model.pt",
                        help="Model checkpoint path")
    parser.add_argument("--topology", type=str, default=None,
                        help="Circuit topology (e.g., buck, boost, inverting_amp)")
    parser.add_argument("--vin", type=float, default=None, help="Input voltage")
    parser.add_argument("--vout", type=float, default=None, help="Output voltage")
    parser.add_argument("--iout", type=float, default=None, help="Output current")
    parser.add_argument("--fsw", type=float, default=None, help="Switching frequency")
    parser.add_argument("--cutoff-freq", type=float, default=None, help="Cutoff frequency")
    parser.add_argument("-n", "--num-designs", type=int, default=1,
                        help="Number of candidate designs to generate")
    parser.add_argument("--simulate", action="store_true",
                        help="Run SPICE simulation on generated circuits")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive mode")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
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

    # Load model
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading ARCS model from {args.checkpoint}...")
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"  Ready (device={device})")

    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, device, args.simulate)
        return

    # Single-shot mode
    if args.topology is None:
        parser.error("--topology is required (or use --interactive)")

    topo = args.topology.lower()
    if topo not in TOPOLOGY_SPECS:
        print(f"Error: Unknown topology '{topo}'")
        print(f"Available: {', '.join(sorted(TOPOLOGY_SPECS.keys()))}")
        sys.exit(1)

    # Collect specs from CLI args
    specs: dict[str, float] = {}
    spec_map = {
        "vin": args.vin,
        "vout": args.vout,
        "iout": args.iout,
        "fsw": args.fsw,
        "cutoff_freq": args.cutoff_freq,
    }
    for k, v in spec_map.items():
        if v is not None:
            specs[k] = v

    # Defaults for power converters
    if topo in _TIER1_NAMES:
        defaults = {"vin": 12.0, "vout": 5.0, "iout": 1.0, "fsw": 100000}
        for k, v in defaults.items():
            if k not in specs:
                specs[k] = v

    runner = NGSpiceRunner() if args.simulate else None

    # Generate
    print(f"\nGenerating {args.num_designs} {topo} circuit(s)...")
    specs_str = ", ".join(f"{k}={v:.4g}" for k, v in specs.items())
    print(f"  Specs: {specs_str}")
    print()

    designs = []
    for i in range(args.num_designs):
        decoded, dt_ms = generate_circuit(
            model, tokenizer, device, topo, specs,
            temperature=args.temperature, top_k=args.top_k,
        )

        sim_outcome = None
        reward = None
        if args.simulate and decoded.valid_structure:
            sim_outcome = simulate_decoded_circuit(decoded, runner=runner)
            reward = compute_reward(decoded, sim_outcome, specs)

        designs.append((decoded, sim_outcome, reward, dt_ms))

        if not args.json:
            tag = f"[{i+1}/{args.num_designs}]" if args.num_designs > 1 else ""
            print(f"--- Design {tag} ---")
            print(format_circuit(decoded, sim_outcome, reward, dt_ms))
            print()

    # If multiple designs, rank by reward
    if args.num_designs > 1 and args.simulate:
        ranked = sorted(
            enumerate(designs),
            key=lambda x: x[1][2] if x[1][2] is not None else -1,
            reverse=True,
        )
        if not args.json:
            print("=" * 50)
            print("RANKING (by reward):")
            for rank, (idx, (dec, sim, rew, dt)) in enumerate(ranked, 1):
                valid_str = "VALID" if sim and sim.valid else "invalid"
                rew_str = f"{rew:.2f}" if rew is not None else "N/A"
                print(f"  #{rank}: Design {idx+1} — reward={rew_str}/8.0 ({valid_str})")
            best_idx = ranked[0][0]
            print(f"\nBest design: #{best_idx + 1}")

    # JSON output
    if args.json:
        json_out = []
        for decoded, sim_outcome, reward, dt_ms in designs:
            entry = {
                "topology": decoded.topology,
                "valid_structure": decoded.valid_structure,
                "components": [
                    {"type": ct, "value": cv} for ct, cv in decoded.components
                ],
                "specs": decoded.specs,
                "inference_ms": round(dt_ms, 1),
            }
            if sim_outcome is not None:
                entry["sim_success"] = sim_outcome.success
                entry["sim_valid"] = sim_outcome.valid
                entry["sim_metrics"] = sim_outcome.metrics
                entry["sim_time"] = sim_outcome.sim_time
            if reward is not None:
                entry["reward"] = round(reward, 3)
            json_out.append(entry)
        print(json.dumps(json_out, indent=2))


if __name__ == "__main__":
    main()
