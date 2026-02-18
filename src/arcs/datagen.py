"""Data generation pipeline for ARCS.

Sweeps component parameter space, runs SPICE simulations,
and stores (topology, parameters, metrics) tuples as training data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from tqdm import tqdm

from arcs.spice import NGSpiceRunner, SimulationResult
from arcs.templates import TopologyTemplate, get_topology, get_all_topologies


@dataclass
class CircuitSample:
    """A single data point: topology + parameters + simulation results."""

    topology: str
    parameters: dict[str, float]
    operating_conditions: dict[str, float]
    metrics: dict[str, float]
    valid: bool  # Did the simulation converge and produce reasonable output?
    sim_time: float
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CircuitSample":
        return cls(**d)


def compute_derived_metrics(
    raw_metrics: dict[str, float],
    operating_conditions: dict[str, float],
    topology_name: str,
) -> dict[str, float]:
    """Compute derived performance metrics from raw simulation outputs.

    Returns enriched metrics dict with efficiency, regulation error, etc.
    """
    derived = dict(raw_metrics)  # Copy raw

    # Compute power from measured currents and voltages
    vout_avg = abs(raw_metrics.get("vout_avg", 0))
    iout_avg = abs(raw_metrics.get("iout_avg", 0))
    vin_val = abs(operating_conditions.get("vin", 0))
    iin_avg = abs(raw_metrics.get("iin_avg", 0))
    vout_target = operating_conditions.get("vout", 0)

    pout = vout_avg * iout_avg
    pin = vin_val * iin_avg
    derived["pout"] = pout
    derived["pin"] = pin

    # Efficiency
    if pin > 1e-9:
        derived["efficiency"] = pout / pin
    else:
        derived["efficiency"] = 0.0

    # Output voltage regulation error (%)
    # Compare absolute values to handle inverting topologies (buck-boost: vout < 0)
    if abs(vout_target) > 1e-9:
        derived["vout_error_pct"] = abs(vout_avg - abs(vout_target)) / abs(vout_target) * 100
    else:
        derived["vout_error_pct"] = 100.0

    # Ripple ratio (ripple / vout)
    ripple = abs(raw_metrics.get("vout_ripple", 0))
    if abs(vout_avg) > 1e-9:
        derived["ripple_ratio"] = ripple / abs(vout_avg)
    else:
        derived["ripple_ratio"] = 1.0

    return derived


def is_valid_result(
    metrics: dict[str, float],
    operating_conditions: dict[str, float],
) -> bool:
    """Check if simulation result is physically plausible.

    We keep ALL results (valid & invalid) in the dataset,
    but this flag helps label them for the model.
    """
    efficiency = metrics.get("efficiency", 0)
    vout_error = metrics.get("vout_error_pct", 100)
    ripple_ratio = metrics.get("ripple_ratio", 1.0)

    # Basic sanity checks
    if efficiency <= 0 or efficiency > 1.0:
        return False
    if vout_error > 50:  # More than 50% off target
        return False
    if ripple_ratio > 0.5:  # More than 50% ripple
        return False

    return True


class DataGenerator:
    """Generates circuit training data by sweeping parameters and simulating."""

    def __init__(
        self,
        output_dir: str | Path = "data/raw",
        ngspice_path: str = "ngspice",
        timeout: int = 30,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runner = NGSpiceRunner(ngspice_path=ngspice_path, timeout=timeout)
        self.rng = np.random.default_rng(seed)

    def generate_for_topology(
        self,
        topology_name: str,
        n_samples: int = 2000,
        snap_to_e_series: bool = True,
        e_series: int = 24,
    ) -> list[CircuitSample]:
        """Generate n_samples for a given topology.

        Args:
            topology_name: Name of the topology (e.g., 'buck', 'boost').
            n_samples: Number of random parameter sets to simulate.
            snap_to_e_series: Snap component values to standard E-series values.
            e_series: Which E-series to use (12, 24).

        Returns:
            List of CircuitSample objects (both valid and invalid).
        """
        template = get_topology(topology_name)
        samples: list[CircuitSample] = []
        valid_count = 0
        fail_count = 0

        print(f"\n{'='*60}")
        print(f"Generating {n_samples} samples for {topology_name.upper()}")
        print(f"Operating conditions: {template.operating_conditions}")
        print(f"{'='*60}")

        for i in tqdm(range(n_samples), desc=topology_name):
            # Sample random parameters
            params = template.sample_parameters(self.rng)

            # Optionally snap to E-series values
            if snap_to_e_series:
                for bound in template.component_bounds:
                    if bound.unit in ("H", "F", "Ω"):  # Only snap R, L, C
                        params[bound.name] = bound.snap_to_e_series(
                            params[bound.name], series=e_series
                        )

            # Generate netlist and simulate
            try:
                netlist = template.generate_netlist(params)
                result = self.runner.run(netlist)

                if result.success and result.metrics:
                    metrics = compute_derived_metrics(
                        result.metrics, template.operating_conditions, topology_name
                    )
                    valid = is_valid_result(metrics, template.operating_conditions)
                    if valid:
                        valid_count += 1

                    sample = CircuitSample(
                        topology=topology_name,
                        parameters=params,
                        operating_conditions=template.operating_conditions,
                        metrics=metrics,
                        valid=valid,
                        sim_time=result.sim_time_seconds,
                    )
                else:
                    fail_count += 1
                    sample = CircuitSample(
                        topology=topology_name,
                        parameters=params,
                        operating_conditions=template.operating_conditions,
                        metrics=result.metrics or {},
                        valid=False,
                        sim_time=result.sim_time_seconds,
                        error_message=result.error_message or "No metrics extracted",
                    )

            except Exception as e:
                fail_count += 1
                sample = CircuitSample(
                    topology=topology_name,
                    parameters=params,
                    operating_conditions=template.operating_conditions,
                    metrics={},
                    valid=False,
                    sim_time=0.0,
                    error_message=str(e),
                )

            samples.append(sample)

        print(f"\nResults: {valid_count} valid, {len(samples) - valid_count} invalid "
              f"({fail_count} simulation failures)")

        return samples

    def generate_all(
        self,
        n_samples_per_topology: int = 2000,
        topologies: list[str] | None = None,
    ) -> dict[str, list[CircuitSample]]:
        """Generate data for all (or specified) topologies.

        Returns:
            Dict mapping topology name to list of samples.
        """
        if topologies is None:
            topologies = ["buck", "boost", "buck_boost", "cuk", "sepic", "flyback", "forward"]

        all_samples = {}
        total_start = time.time()

        for topo_name in topologies:
            samples = self.generate_for_topology(topo_name, n_samples_per_topology)
            all_samples[topo_name] = samples

            # Save incrementally
            self._save_topology(topo_name, samples)

        elapsed = time.time() - total_start
        total = sum(len(s) for s in all_samples.values())
        valid = sum(sum(1 for s in samples if s.valid) for samples in all_samples.values())

        print(f"\n{'='*60}")
        print(f"TOTAL: {total} samples ({valid} valid) in {elapsed:.1f}s")
        print(f"Saved to: {self.output_dir}")
        print(f"{'='*60}")

        return all_samples

    def _save_topology(self, name: str, samples: list[CircuitSample]) -> None:
        """Save samples for one topology to JSONL file."""
        outfile = self.output_dir / f"{name}.jsonl"
        with open(outfile, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")

    @staticmethod
    def load_topology(filepath: str | Path) -> list[CircuitSample]:
        """Load samples from a JSONL file."""
        samples = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    samples.append(CircuitSample.from_dict(json.loads(line)))
        return samples

    @staticmethod
    def load_all(data_dir: str | Path) -> dict[str, list[CircuitSample]]:
        """Load all topology data from a directory."""
        data_dir = Path(data_dir)
        all_samples = {}
        for filepath in sorted(data_dir.glob("*.jsonl")):
            name = filepath.stem
            all_samples[name] = DataGenerator.load_topology(filepath)
        return all_samples


def main():
    """CLI entry point for data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="ARCS Data Generation Pipeline")
    parser.add_argument(
        "--topologies", nargs="+",
        default=["buck", "boost", "buck_boost", "cuk", "sepic", "flyback", "forward"],
        help="Topologies to generate data for",
    )
    parser.add_argument("--samples", type=int, default=2000, help="Samples per topology")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=int, default=30, help="ngspice timeout (sec)")
    parser.add_argument("--no-e-series", action="store_true", help="Don't snap to E-series")

    args = parser.parse_args()

    generator = DataGenerator(
        output_dir=args.output,
        timeout=args.timeout,
        seed=args.seed,
    )
    generator.generate_all(
        n_samples_per_topology=args.samples,
        topologies=args.topologies,
    )


if __name__ == "__main__":
    main()
