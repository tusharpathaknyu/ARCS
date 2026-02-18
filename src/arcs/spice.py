"""ARCS SPICE simulation interface.

Wraps ngspice for running circuit simulations and extracting metrics.
"""

from __future__ import annotations

import subprocess
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SimulationResult:
    """Result of a SPICE simulation."""

    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    raw_output: str = ""
    error_message: str = ""
    netlist_path: str = ""
    sim_time_seconds: float = 0.0


class NGSpiceRunner:
    """Runs ngspice simulations and extracts performance metrics."""

    def __init__(
        self,
        ngspice_path: str = "ngspice",
        timeout: int = 30,
        temp_dir: Optional[str] = None,
    ):
        self.ngspice_path = ngspice_path
        self.timeout = timeout
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def run(self, netlist: str, measure_names: list[str] | None = None) -> SimulationResult:
        """Run a SPICE netlist and extract .measure results.

        Args:
            netlist: Complete SPICE netlist string.
            measure_names: Expected .measure statement names to extract.

        Returns:
            SimulationResult with metrics and status.
        """
        import time

        # Write netlist to temp file
        netlist_path = os.path.join(self.temp_dir, f"arcs_sim_{os.getpid()}.cir")
        with open(netlist_path, "w") as f:
            f.write(netlist)

        output_path = netlist_path + ".out"
        start = time.time()
        try:
            result = subprocess.run(
                [self.ngspice_path, "-b", "-o", output_path, netlist_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start

            # Read ngspice output file (measure results go here, not stdout)
            ngspice_output = ""
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    ngspice_output = f.read()

            # Combine all output sources for parsing
            output = ngspice_output + "\n" + result.stdout + result.stderr
            metrics = self._parse_measures(output, measure_names)

            # Check for convergence errors
            if "no convergence" in output.lower() or "singular matrix" in output.lower():
                return SimulationResult(
                    success=False,
                    metrics=metrics,
                    raw_output=output,
                    error_message="Simulation did not converge",
                    netlist_path=netlist_path,
                    sim_time_seconds=elapsed,
                )

            success = result.returncode == 0 and len(metrics) > 0
            return SimulationResult(
                success=success,
                metrics=metrics,
                raw_output=output,
                netlist_path=netlist_path,
                sim_time_seconds=elapsed,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return SimulationResult(
                success=False,
                error_message=f"Simulation timed out after {self.timeout}s",
                netlist_path=netlist_path,
                sim_time_seconds=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            return SimulationResult(
                success=False,
                error_message=str(e),
                netlist_path=netlist_path,
                sim_time_seconds=elapsed,
            )
        finally:
            # Clean up temp files
            for path in [netlist_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)

    def _parse_measures(
        self, output: str, measure_names: list[str] | None = None
    ) -> dict[str, float]:
        """Parse .measure results from ngspice output."""
        metrics = {}
        # ngspice outputs measures like:
        #   vout_avg            =  1.954011e+01 from=  4.000000e-04 to=...
        for line in output.split("\n"):
            match = re.match(
                r"\s*(\w+)\s+=\s+([+-]?\d+\.?\d*(?:e[+-]?\d+)?)",
                line,
                re.IGNORECASE,
            )
            if match:
                name = match.group(1).lower()
                try:
                    value = float(match.group(2))
                    metrics[name] = value
                except ValueError:
                    continue

        return metrics

    def run_with_rawfile(self, netlist: str) -> SimulationResult:
        """Run simulation and parse binary rawfile for waveform data.

        This is more robust than parsing stdout for complex analyses.
        """
        netlist_path = os.path.join(self.temp_dir, f"arcs_sim_{os.getpid()}.cir")
        raw_path = netlist_path.replace(".cir", ".raw")

        # Add rawfile output to netlist if not present
        if ".raw" not in netlist:
            lines = netlist.strip().split("\n")
            # Insert before .end
            for i, line in enumerate(lines):
                if line.strip().lower() == ".end":
                    lines.insert(i, f".save all")
                    break
            netlist = "\n".join(lines)

        with open(netlist_path, "w") as f:
            f.write(netlist)

        import time

        start = time.time()
        try:
            result = subprocess.run(
                [self.ngspice_path, "-b", "-r", raw_path, netlist_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start

            output = result.stdout + result.stderr
            metrics = self._parse_measures(output)

            success = result.returncode == 0
            return SimulationResult(
                success=success,
                metrics=metrics,
                raw_output=output,
                netlist_path=netlist_path,
                sim_time_seconds=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return SimulationResult(
                success=False,
                error_message=f"Simulation timed out after {self.timeout}s",
                netlist_path=netlist_path,
                sim_time_seconds=elapsed,
            )
        finally:
            for p in [netlist_path, raw_path]:
                if os.path.exists(p):
                    os.remove(p)
