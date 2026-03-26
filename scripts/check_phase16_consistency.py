#!/usr/bin/env python3
"""Validate metric consistency across results, README, and paper.

Updated for v5 models (34 topologies). Uses hybrid_v5.json as the
canonical results file. Falls back to hybrid_phase14.json if present.
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
# Prefer v5 results; fall back to phase14
HYBRID = ROOT / "results" / "hybrid_v5.json"
if not HYBRID.exists():
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

    # Must have a VCG row in graph model table (v3, v4, or v5)
    _require(
        readme_text,
        r"\|\s*VCG\s*(v[345]\s*)?\(VAE\)\s*\|\s*4\.0M\s*\|\s*100\.0%\s*\|\s*\d+/\d+\s*\|",
        "README missing VCG graph-model row with 100% validity",
        errors,
    )
    _require(
        readme_text,
        r"\|\s*CCFM\s*(v[345]\s*)?\(Flow Matching\)\s*\|\s*7\.6M\s*\|\s*100\.0%\s*\|\s*\d+/\d+\s*\|",
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
            vcg_reward_paper = f"{vcg_summary['mean_reward']:.2f}"
            ccfm_reward_paper = f"{ccfm_summary['mean_reward']:.2f}"
            hybrid_reward_paper = f"{hybrid_summary['mean_reward']:.2f}"

            # Check that paper mentions VCG, CCFM, and Hybrid results
            _require(
                paper_text,
                rf"VCG-only\s*&\s*100\\%\s*&.*&.*&\s*{vcg_reward_paper}",
                "Paper VCG-only hybrid row does not match results",
                errors,
            )
            _require(
                paper_text,
                rf"CCFM-only\s*&\s*100\\%\s*&.*&.*&\s*{ccfm_reward_paper}",
                "Paper CCFM-only hybrid row does not match results",
                errors,
            )
            _require(
                paper_text,
                rf"Hybrid.*VCG\+CCFM.*{hybrid_reward_paper}",
                "Paper hybrid row does not match results (reward)",
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
