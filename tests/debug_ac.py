"""Debug: write a Tier 2 netlist to file and run ngspice manually."""
from arcs.templates import get_topology
import numpy as np, subprocess

rng = np.random.default_rng(42)
t = get_topology("inverting_amp")
params = t.sample_parameters(rng)
netlist = t.generate_netlist(params)

# Write netlist
path = "/tmp/arcs_debug_inv_amp.cir"
with open(path, "w") as f:
    f.write(netlist)

print("=== NETLIST ===")
print(netlist)
print("=== RUNNING NGSPICE ===")
result = subprocess.run(
    ["ngspice", "-b", path],
    capture_output=True, text=True, timeout=30
)
print("STDOUT (last 3000 chars):")
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR (last 2000 chars):")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print(f"Return code: {result.returncode}")
