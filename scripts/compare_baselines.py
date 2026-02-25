"""Generate comparison table between ARCS models and baselines."""
import json

# Load baseline results
with open("results/baseline_random_search.json") as f:
    rs = json.load(f)
with open("results/baseline_ga.json") as f:
    ga = json.load(f)

# Load ARCS model results
with open("results/sim_combined_cond.json") as f:
    combined = json.load(f)
with open("results/sim_rl_best_cond.json") as f:
    rl_best = json.load(f)

print("=" * 75)
print("ARCS vs BASELINES - Full Comparison Table")
print("=" * 75)
print()

header = f"{'Method':<30s} {'Sims/Design':>12s} {'SimSuccess':>12s} {'SimValid':>10s} {'AvgReward':>10s}"
print(header)
print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

# Random Search
print(f"{'Random Search (200 trials)':<30s} {'200':>12s} {rs['sim_success_rate']:>11.1%} {rs['sim_valid_rate']:>9.1%} {rs['avg_reward']:>10.3f}")

# GA
print(f"{'GA (pop=30, gen=20, 630 eval)':<30s} {'630':>12s} {ga['sim_success_rate']:>11.1%} {ga['sim_valid_rate']:>9.1%} {ga['avg_reward']:>10.3f}")

# ARCS Combined (supervised only)
print(f"{'ARCS Combined (supervised)':<30s} {'1 (0.02s)':>12s} {combined['sim_success_rate']:>11.1%} {combined['sim_valid_rate']:>9.1%} {combined['avg_reward']:>10.3f}")

# ARCS RL v2 Best
print(f"{'ARCS RL v2 Best':<30s} {'1 (0.02s)':>12s} {rl_best['sim_success_rate']:>11.1%} {rl_best['sim_valid_rate']:>9.1%} {rl_best['avg_reward']:>10.3f}")

print()
print("Key insight: Baselines achieve higher absolute reward by running")
print("200-630 SPICE simulations per design. ARCS generates a single design")
print("in ~20ms with no simulation loop, making it 10,000x faster at inference.")
print()

# Compute cost comparison
rs_time = rs["wall_time"]
ga_time = ga["wall_time"]
n_specs = rs["n_specs"]
rs_time_per = rs_time / n_specs
ga_time_per = ga_time / n_specs
arcs_time_per = 0.02  # ~20ms per design

print(f"Cost per design: RS={rs_time_per:.1f}s, GA={ga_time_per:.1f}s, ARCS={arcs_time_per:.3f}s")
print(f"Speedup: ARCS is {rs_time_per/arcs_time_per:.0f}x faster than RS, {ga_time_per/arcs_time_per:.0f}x faster than GA")
print()

# Per-topology comparison
print("=" * 75)
print("PER-TOPOLOGY COMPARISON")
print("=" * 75)
print()
header2 = f"{'Topology':<25s} {'RS AvgR':>8s} {'GA AvgR':>8s}  {'RS Vld':>7s} {'GA Vld':>7s} {'ARCS Vld':>8s}   {'ARCS OK':>7s}"
print(header2)
print(f"{'-'*25} {'-'*8} {'-'*8}  {'-'*7} {'-'*7} {'-'*8}   {'-'*7}")

rs_topo = rs["per_topology"]
ga_topo = ga["per_topology"]

# ARCS per-topology from per_topology_sim field
arcs_topo = rl_best.get("per_topology_sim", {})
arcs_comb_topo = combined.get("per_topology_sim", {})

for topo in sorted(rs_topo.keys()):
    rs_r = rs_topo[topo]["avg_reward"]
    ga_r = ga_topo.get(topo, {}).get("avg_reward", 0)
    rs_v = rs_topo[topo]["sim_valid"]
    ga_v = ga_topo.get(topo, {}).get("sim_valid", 0)
    
    arcs_v = arcs_topo.get(topo, {}).get("sim_valid_rate", 0)
    arcs_s = arcs_topo.get(topo, {}).get("sim_success_rate", 0)
    
    print(f"{topo:<25s} {rs_r:>8.3f} {ga_r:>8.3f}  {rs_v:>7.1%} {ga_v:>7.1%} {arcs_v:>7.1%}   {arcs_s:>6.1%}")

print()
print("Note: Baselines sample directly in parameter space with known correct")
print("topology + component count. ARCS must predict both from scratch.")
print()

# Summary table for paper (Markdown)
print("=" * 75)
print("MARKDOWN TABLE FOR PAPER")
print("=" * 75)
print()
print("| Method | Sims/Design | Sim Success | Sim Valid | Avg Reward | Wall Time |")
print("|--------|-------------|-------------|-----------|------------|-----------|")
print(f"| Random Search (N=200) | 200 | {rs['sim_success_rate']:.1%} | {rs['sim_valid_rate']:.1%} | {rs['avg_reward']:.3f} | {rs['wall_time']:.0f}s |")
print(f"| Genetic Algorithm | 630 | {ga['sim_success_rate']:.1%} | {ga['sim_valid_rate']:.1%} | {ga['avg_reward']:.3f} | {ga['wall_time']:.0f}s |")
print(f"| ARCS (supervised) | 1 | {combined['sim_success_rate']:.1%} | {combined['sim_valid_rate']:.1%} | {combined['avg_reward']:.3f} | ~0.02s |")
print(f"| ARCS + RL (best) | 1 | {rl_best['sim_success_rate']:.1%} | {rl_best['sim_valid_rate']:.1%} | {rl_best['avg_reward']:.3f} | ~0.02s |")
