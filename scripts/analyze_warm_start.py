#!/usr/bin/env python3
"""Analyze warm-start experiment results."""
import json

d = json.load(open("results/warm_start.json"))

for key in ["arcs_only", "ga_cold", "ga_warm"]:
    rewards = [x["reward"] for x in d[key]]
    times = [x["time"] for x in d[key]]
    sims = [x["sims"] for x in d[key]]
    avg_r = sum(rewards) / len(rewards)
    avg_t = sum(times) / len(times)
    avg_s = sum(sims) / len(sims)
    print(f"{key}: avg_reward={avg_r:.3f}, avg_time={avg_t:.1f}s, avg_sims={avg_s:.0f}")

# Non-fallback warm-start
warm_nf = [x for x in d["ga_warm"] if not x.get("fallback")]
cold_matched = [x for x in d["ga_cold"] if x["topology"] in [w["topology"] for w in warm_nf]]
print(f"\nWarm (no fallback, {len(warm_nf)} topos): avg_reward={sum(x['reward'] for x in warm_nf)/len(warm_nf):.3f}, avg_sims={sum(x['sims'] for x in warm_nf)/len(warm_nf):.0f}")
print(f"Cold (same topos): avg_reward={sum(x['reward'] for x in cold_matched)/len(cold_matched):.3f}, avg_sims={sum(x['sims'] for x in cold_matched)/len(cold_matched):.0f}")

# Per topology comparison
print(f"\n{'Topology':<25} {'ARCS':>8} {'GA-Cold':>8} {'GA-Warm':>8} {'Warm/Cold%':>10}")
for a, c, w in zip(d["arcs_only"], d["ga_cold"], d["ga_warm"]):
    pct = w["reward"] / c["reward"] * 100 if c["reward"] > 0 else 0
    fb = " (fb)" if w.get("fallback") else ""
    print(f"{a['topology']:<25} {a['reward']:>8.2f} {c['reward']:>8.2f} {w['reward']:>8.2f} {pct:>9.1f}%{fb}")
