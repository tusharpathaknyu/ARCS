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
from arcs.templates import (
    TopologyTemplate, get_topology, get_all_topologies,
    _TIER1_NAMES, _TIER2_NAMES,
)


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

    Dispatches to power-converter or signal-processing logic based on
    topology tier.
    """
    if topology_name in _TIER1_NAMES:
        return _compute_power_metrics(raw_metrics, operating_conditions)
    else:
        return _compute_signal_metrics(raw_metrics, operating_conditions, topology_name)


def _compute_power_metrics(
    raw_metrics: dict[str, float],
    operating_conditions: dict[str, float],
) -> dict[str, float]:
    """Tier 1: power converter derived metrics."""
    derived = dict(raw_metrics)

    vout_avg = abs(raw_metrics.get("vout_avg", 0))
    iout_avg = abs(raw_metrics.get("iout_avg", 0))
    vin_val = abs(operating_conditions.get("vin", 0))
    iin_avg = abs(raw_metrics.get("iin_avg", 0))
    vout_target = operating_conditions.get("vout", 0)

    pout = vout_avg * iout_avg
    pin = vin_val * iin_avg
    derived["pout"] = pout
    derived["pin"] = pin

    if pin > 1e-9:
        derived["efficiency"] = pout / pin
    else:
        derived["efficiency"] = 0.0

    if abs(vout_target) > 1e-9:
        derived["vout_error_pct"] = abs(vout_avg - abs(vout_target)) / abs(vout_target) * 100
    else:
        derived["vout_error_pct"] = 100.0

    ripple = abs(raw_metrics.get("vout_ripple", 0))
    if abs(vout_avg) > 1e-9:
        derived["ripple_ratio"] = ripple / abs(vout_avg)
    else:
        derived["ripple_ratio"] = 1.0

    return derived


def _compute_signal_metrics(
    raw_metrics: dict[str, float],
    operating_conditions: dict[str, float],
    topology_name: str,
) -> dict[str, float]:
    """Tier 2: signal-processing derived metrics (amplifiers, filters, oscillators).

    Computes bandwidth / cutoff frequency from the multi-frequency VDB probes
    (vdb_0..vdb_7) measured at [10, 100, 1k, 10k, 100k, 1M, 10M, 50M] Hz.
    """
    derived = dict(raw_metrics)

    # Gain in dB
    gain_db = raw_metrics.get("gain_db", raw_metrics.get("gain_dc", None))
    if gain_db is not None:
        derived["gain_linear"] = 10 ** (gain_db / 20)

    # --- Estimate -3 dB bandwidth / cutoff from probe frequencies ---
    probe_freqs = [10, 100, 1e3, 10e3, 100e3, 1e6, 10e6, 50e6]
    probe_gains = []
    for i, f in enumerate(probe_freqs):
        v = raw_metrics.get(f"vdb_{i}")
        if v is not None:
            probe_gains.append((f, v))

    if len(probe_gains) >= 2:
        # Find passband gain (maximum gain among probes)
        peak_freq, peak_gain = max(probe_gains, key=lambda x: x[1])
        derived["peak_gain_db"] = peak_gain
        derived["peak_freq"] = peak_freq
        threshold = peak_gain - 3.0

        # Find -3 dB crossing by linear interpolation on dB scale
        # For LP / amps: look for gain dropping below threshold going up in freq
        for j in range(len(probe_gains) - 1):
            f1, g1 = probe_gains[j]
            f2, g2 = probe_gains[j + 1]
            if g1 >= threshold and g2 < threshold:
                # Interpolate (log-frequency, linear dB)
                if abs(g1 - g2) > 0.01:
                    ratio = (threshold - g1) / (g2 - g1)
                    import math
                    fc = 10 ** (math.log10(f1) + ratio * (math.log10(f2) - math.log10(f1)))
                else:
                    fc = (f1 + f2) / 2
                derived["bw_3db"] = fc
                break

        # For HP filters: look for gain rising above threshold going up in freq
        if "bw_3db" not in derived:
            for j in range(len(probe_gains) - 1):
                f1, g1 = probe_gains[j]
                f2, g2 = probe_gains[j + 1]
                if g1 < threshold and g2 >= threshold:
                    if abs(g2 - g1) > 0.01:
                        ratio = (threshold - g1) / (g2 - g1)
                        import math
                        fc = 10 ** (math.log10(f1) + ratio * (math.log10(f2) - math.log10(f1)))
                    else:
                        fc = (f1 + f2) / 2
                    derived["bw_3db"] = fc
                    break

    return derived


def is_valid_result(
    metrics: dict[str, float],
    operating_conditions: dict[str, float],
    topology_name: str = "",
) -> bool:
    """Check if simulation result is physically plausible.

    We keep ALL results (valid & invalid) in the dataset,
    but this flag helps label them for the model.
    """
    if topology_name in _TIER1_NAMES or not topology_name:
        return _is_valid_power(metrics, operating_conditions)
    else:
        return _is_valid_signal(metrics, operating_conditions, topology_name)


def _is_valid_power(metrics: dict[str, float], operating_conditions: dict[str, float]) -> bool:
    """Power converter validity checks."""
    efficiency = metrics.get("efficiency", 0)
    vout_error = metrics.get("vout_error_pct", 100)
    ripple_ratio = metrics.get("ripple_ratio", 1.0)

    if efficiency <= 0 or efficiency > 1.0:
        return False
    if vout_error > 50:
        return False
    if ripple_ratio > 0.5:
        return False
    return True


def _is_valid_signal(
    metrics: dict[str, float],
    operating_conditions: dict[str, float],
    topology_name: str,
) -> bool:
    """Signal-processing validity checks."""
    amp_types = {"inverting_amp", "noninverting_amp", "instrumentation_amp", "differential_amp"}
    filter_types = {"sallen_key_lowpass", "sallen_key_highpass", "sallen_key_bandpass"}
    osc_types = {"wien_bridge", "colpitts"}

    if topology_name in amp_types:
        gain_db = metrics.get("gain_db", metrics.get("gain_dc"))
        if gain_db is None:
            return False
        if abs(gain_db) > 120:  # >120 dB unrealistic
            return False
        return True

    if topology_name in filter_types:
        # Must have gain_dc (passband or reference) and bw_3db from post-processing
        gain_dc = metrics.get("gain_dc")
        if gain_dc is None:
            return False
        bw = metrics.get("bw_3db")
        if bw is not None and bw <= 0:
            return False
        return True

    if topology_name in osc_types:
        vosc_pp = metrics.get("vosc_pp", 0)
        if vosc_pp < 0.01:
            return False
        return True

    return len(metrics) > 0


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
                    valid = is_valid_result(metrics, template.operating_conditions, topology_name)
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
            topologies = _TIER1_NAMES + _TIER2_NAMES

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
        default=None,  # Will use all tiers if None
        help="Topologies to generate data for (default: all tiers)",
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
