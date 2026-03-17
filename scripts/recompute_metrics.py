#!/usr/bin/env python3
"""Re-compute derived metrics and re-label valid flags for all JSONL data.

Some topologies (especially extended power converters) had raw simulation
metrics (vout_avg, iout_avg, etc.) but missing derived metrics (efficiency,
vout_error_pct, ripple_ratio) because the original datagen didn't compute
them. This script:
1. Computes derived metrics from raw metrics + operating conditions
2. Re-labels valid/invalid using corrected is_valid_result()
3. Overwrites the JSONL files in place

Usage:
    PYTHONPATH=src python scripts/recompute_metrics.py --data data/combined_v2
"""
import argparse
import json
from pathlib import Path

from arcs.datagen import compute_derived_metrics, is_valid_result


def recompute_file(path: Path) -> dict:
    """Re-compute derived metrics and re-label a single JSONL file."""
    lines = path.read_text().strip().split("\n")
    stats = {
        "total": 0, "old_valid": 0, "new_valid": 0,
        "metrics_added": 0, "changed": 0,
    }

    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        sample = json.loads(line)
        stats["total"] += 1

        old_valid = sample.get("valid", False)
        if old_valid:
            stats["old_valid"] += 1

        topology = sample.get("topology", "")
        raw_metrics = sample.get("metrics", {})
        oc = sample.get("operating_conditions", {})

        # Check if derived metrics are missing
        has_derived = (
            "efficiency" in raw_metrics
            or "gain_db" in raw_metrics
            or "vosc_pp" in raw_metrics
            or "iref" in raw_metrics
        )

        if not has_derived and raw_metrics:
            # Re-compute derived metrics from raw data
            try:
                derived = compute_derived_metrics(raw_metrics, oc, topology)
                # Merge derived into existing metrics (don't overwrite raw)
                for k, v in derived.items():
                    if k not in raw_metrics:
                        raw_metrics[k] = v
                        stats["metrics_added"] += 1
                sample["metrics"] = raw_metrics
            except Exception as e:
                pass  # keep original metrics if computation fails

        # Re-label valid flag
        new_valid = is_valid_result(raw_metrics, oc, topology)
        if new_valid:
            stats["new_valid"] += 1
        if old_valid != new_valid:
            stats["changed"] += 1
        sample["valid"] = new_valid

        new_lines.append(json.dumps(sample))

    path.write_text("\n".join(new_lines) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/combined_v2")
    args = parser.parse_args()

    data_dir = Path(args.data)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    print(f"Re-computing metrics for {len(jsonl_files)} files in {data_dir}")
    print(f"{'Topology':<30} {'Total':>6} {'OldValid':>9} {'NewValid':>9} {'MetricsAdded':>13} {'Changed':>8}")
    print("-" * 82)

    total = {"total": 0, "old_valid": 0, "new_valid": 0, "metrics_added": 0, "changed": 0}

    for f in jsonl_files:
        topology = f.stem
        stats = recompute_file(f)
        for k in total:
            total[k] += stats[k]

        marker = " <<<" if stats["changed"] > 0 or stats["metrics_added"] > 0 else ""
        print(f"{topology:<30} {stats['total']:>6} {stats['old_valid']:>9} "
              f"{stats['new_valid']:>9} {stats['metrics_added']:>13} {stats['changed']:>8}{marker}")

    print("-" * 82)
    print(f"{'TOTAL':<30} {total['total']:>6} {total['old_valid']:>9} "
          f"{total['new_valid']:>9} {total['metrics_added']:>13} {total['changed']:>8}")
    print(f"\nValid: {total['old_valid']} → {total['new_valid']}")
    print(f"Metrics added: {total['metrics_added']}")
    print(f"Labels changed: {total['changed']}")


if __name__ == "__main__":
    main()
