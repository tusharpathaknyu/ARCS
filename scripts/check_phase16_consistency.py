#!/usr/bin/env python3
"""Validate metric consistency across results, README, and paper.

Updated for Phase 17+ (34 topologies, v3 models). Falls back to Phase 14
result files if present, otherwise checks README structural patterns only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PAPER = ROOT / "paper" / "arcs_paper.tex"
PHASE14 = ROOT / "results" / "phase14_comparison.json"
HYBRID = ROOT / "results" / "hybrid_phase14.json"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _require(text: str, pattern: str, label: str, errors: list[str]) -> None:
    if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is None:
        errors.append(label)


def main() -> int:
    readme_text = README.read_text()
    errors: list[str] = []

    # ---- README structural checks (topology-count agnostic) ----

    # Must have a VCG v3 row OR legacy VCG row in graph model table
    _require(
        readme_text,
        r"\|\s*VCG\s*(v3\s*)?\(VAE\)\s*\|\s*4\.0M\s*\|\s*100\.0%\s*\|\s*\d+/\d+\s*\|",
        "README missing VCG graph-model row with 100% validity",
        errors,
    )
    _require(
        readme_text,
        r"\|\s*CCFM\s*(v3\s*)?\(Flow Matching\)\s*\|\s*7\.6M\s*\|\s*100\.0%\s*\|\s*\d+/\d+\s*\|",
        "README missing CCFM graph-model row with 100% validity",
        errors,
    )

    # Must mention 34 topologies somewhere
    _require(
        readme_text,
        r"34 topolog",
        "README does not mention 34 topologies",
        errors,
    )

    # ---- Phase 14 result file checks (if files exist) ----
    phase14 = _load_json(PHASE14)
    hybrid = _load_json(HYBRID)

    if phase14 and hybrid:
        hybrid_summary = hybrid["summary"]["hybrid"]
        vcg_summary = hybrid["summary"]["vcg"]
        ccfm_summary = hybrid["summary"]["ccfm"]

        vcg_hybrid_simvalid = f"{100.0 * vcg_summary['sim_valid_rate']:.1f}"
        ccfm_hybrid_simvalid = f"{100.0 * ccfm_summary['sim_valid_rate']:.1f}"
        hybrid_simvalid = f"{100.0 * hybrid_summary['sim_valid_rate']:.1f}"
        hybrid_reward = f"{hybrid_summary['mean_reward']:.3f}"

        # Check paper if it exists
        if PAPER.exists():
            paper_text = PAPER.read_text()
            hybrid_simvalid_num_paper = (
                str(int(round(100.0 * hybrid_summary["sim_valid_rate"])))
                if abs(100.0 * hybrid_summary["sim_valid_rate"] - round(100.0 * hybrid_summary["sim_valid_rate"])) < 1e-9
                else hybrid_simvalid
            )
            vcg_reward_paper = f"{vcg_summary['mean_reward']:.2f}"
            ccfm_reward_paper = f"{ccfm_summary['mean_reward']:.2f}"
            hybrid_reward_paper = f"{hybrid_summary['mean_reward']:.2f}"

            _require(
                paper_text,
                rf"VCG-only\s*&\s*100\\%\s*&\s*100\\%\s*&\s*{vcg_hybrid_simvalid}\\%\s*&\s*{vcg_reward_paper}",
                "Paper VCG-only hybrid row does not match hybrid_phase14",
                errors,
            )
            _require(
                paper_text,
                rf"CCFM-only\s*&\s*100\\%\s*&\s*100\\%\s*&\s*{ccfm_hybrid_simvalid}\\%\s*&\s*{ccfm_reward_paper}",
                "Paper CCFM-only hybrid row does not match hybrid_phase14",
                errors,
            )
            _require(
                paper_text,
                rf"\\textbf\{{Hybrid \(VCG\+CCFM\)\}}\s*&\s*\\textbf\{{100\\%\}}\s*&\s*\\textbf\{{100\\%\}}\s*&\s*\\textbf\{{{hybrid_simvalid_num_paper}\\%\}}\s*&\s*\\textbf\{{{hybrid_reward_paper}\}}",
                "Paper hybrid row does not match hybrid_phase14 (sim-valid/reward)",
                errors,
            )

    if errors:
        print("[FAIL] Phase 16 consistency check failed:")
        for err in errors:
            print(f" - {err}")
        return 1

    print("[PASS] Phase 16 consistency check passed")
    print(f" - README graph validity: VCG=100.0%, CCFM=100.0%")
    if hybrid:
        hybrid_summary = hybrid["summary"]["hybrid"]
        vcg_summary = hybrid["summary"]["vcg"]
        ccfm_summary = hybrid["summary"]["ccfm"]
        print(
            " - Hybrid summary: "
            f"vcg_sim_valid={_pct(vcg_summary['sim_valid_rate'])}, "
            f"ccfm_sim_valid={_pct(ccfm_summary['sim_valid_rate'])}, "
            f"hybrid_sim_valid={_pct(hybrid_summary['sim_valid_rate'])}, "
            f"reward={hybrid_summary['mean_reward']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
