#!/usr/bin/env python3
"""Re-label valid/invalid flags in JSONL data using corrected validation logic.

This script re-processes all JSONL files in data/combined_v2/ and updates the
'valid' field using the fixed is_valid_result() from datagen.py, which now
properly validates all 34 topologies instead of falling through to
len(metrics)>0 for 18 of them.

Usage:
    PYTHONPATH=src python scripts/relabel_valid.py --data data/combined_v2
"""
import argparse
import json
from pathlib import Path

from arcs.datagen import is_valid_result


def relabel_file(path: Path) -> dict:
    """Re-label a single JSONL file. Returns stats."""
    lines = path.read_text().strip().split("\n")
    stats = {"total": 0, "old_valid": 0, "new_valid": 0, "changed": 0}

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
        metrics = sample.get("metrics", {})
        oc = sample.get("operating_conditions", {})

        new_valid = is_valid_result(metrics, oc, topology)
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

    print(f"Re-labeling {len(jsonl_files)} files in {data_dir}")
    print(f"{'Topology':<30} {'Total':>6} {'OldValid':>9} {'NewValid':>9} {'Changed':>8}")
    print("-" * 70)

    total_stats = {"total": 0, "old_valid": 0, "new_valid": 0, "changed": 0}

    for f in jsonl_files:
        topology = f.stem
        stats = relabel_file(f)

        for k in total_stats:
            total_stats[k] += stats[k]

        marker = " <<<" if stats["changed"] > 0 else ""
        print(f"{topology:<30} {stats['total']:>6} {stats['old_valid']:>9} "
              f"{stats['new_valid']:>9} {stats['changed']:>8}{marker}")

    print("-" * 70)
    print(f"{'TOTAL':<30} {total_stats['total']:>6} {total_stats['old_valid']:>9} "
          f"{total_stats['new_valid']:>9} {total_stats['changed']:>8}")

    pct_change = total_stats["changed"] / max(total_stats["total"], 1) * 100
    print(f"\n{total_stats['changed']} samples changed ({pct_change:.1f}%)")
    print(f"Valid: {total_stats['old_valid']} → {total_stats['new_valid']}")


if __name__ == "__main__":
    main()
