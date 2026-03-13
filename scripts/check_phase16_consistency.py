#!/usr/bin/env python3
"""Validate Phase 16 metric consistency across results, README, and paper."""

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
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _find_model(rows: list[dict], name: str) -> dict:
    for row in rows:
        if row.get("name") == name:
            return row
    raise KeyError(f"Model not found in phase14 results: {name}")


def _require(text: str, pattern: str, label: str, errors: list[str]) -> None:
    if re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is None:
        errors.append(label)


def main() -> int:
    phase14 = _load_json(PHASE14)
    hybrid = _load_json(HYBRID)
    readme_text = README.read_text()
    paper_text = PAPER.read_text()

    vcg = _find_model(phase14, "VCG (VAE)")
    ccfm = _find_model(phase14, "CCFM (Flow Matching)")

    hybrid_summary = hybrid["summary"]["hybrid"]
    vcg_summary = hybrid["summary"]["vcg"]
    ccfm_summary = hybrid["summary"]["ccfm"]

    vcg_struct = _pct(vcg["validity_rate"], 1)
    ccfm_struct = _pct(ccfm["validity_rate"], 1)
    vcg_topos = f"{vcg['topologies_at_100pct']}/16"
    ccfm_topos = f"{ccfm['topologies_at_100pct']}/16"

    vcg_hybrid_simvalid = _pct(vcg_summary["sim_valid_rate"], 1)
    ccfm_hybrid_simvalid = _pct(ccfm_summary["sim_valid_rate"], 1)
    hybrid_simvalid = _pct(hybrid_summary["sim_valid_rate"], 1)
    vcg_hybrid_simvalid_num = f"{100.0 * vcg_summary['sim_valid_rate']:.1f}"
    ccfm_hybrid_simvalid_num = f"{100.0 * ccfm_summary['sim_valid_rate']:.1f}"
    hybrid_simvalid_num = f"{100.0 * hybrid_summary['sim_valid_rate']:.1f}"
    hybrid_simvalid_num_paper = (
        str(int(round(100.0 * hybrid_summary["sim_valid_rate"])))
        if abs(100.0 * hybrid_summary["sim_valid_rate"] - round(100.0 * hybrid_summary["sim_valid_rate"])) < 1e-9
        else hybrid_simvalid_num
    )
    vcg_reward_paper = f"{vcg_summary['mean_reward']:.2f}"
    ccfm_reward_paper = f"{ccfm_summary['mean_reward']:.2f}"

    hybrid_reward_readme = f"{hybrid_summary['mean_reward']:.3f}"
    hybrid_reward_paper = f"{hybrid_summary['mean_reward']:.2f}"

    errors: list[str] = []

    _require(
        readme_text,
        rf"100% structural validity on all\s+16/16\s+topologies",
        "README missing 16/16 structural-validity statement",
        errors,
    )
    _require(
        readme_text,
        rf"\|\s*VCG \(VAE\)\s*\|\s*4\.0M\s*\|\s*{re.escape(vcg_struct)}\s*\|\s*{re.escape(vcg_topos)}\s*\|",
        "README VCG graph-model row does not match phase14 results",
        errors,
    )
    _require(
        readme_text,
        rf"\|\s*CCFM \(Flow Matching\)\s*\|\s*7\.6M\s*\|\s*{re.escape(ccfm_struct)}\s*\|\s*{re.escape(ccfm_topos)}\s*\|",
        "README CCFM graph-model row does not match phase14 results",
        errors,
    )
    _require(
        readme_text,
        rf"Hybrid benchmark \(n={hybrid['n_candidates_per_source']} candidates/source, VCG\+CCFM ranking\)",
        "README hybrid benchmark n-candidates text mismatches hybrid_phase14",
        errors,
    )
    _require(
        readme_text,
        rf"Mean reward:\s*\*\*{re.escape(hybrid_reward_readme)}\*\*",
        "README hybrid mean reward does not match hybrid_phase14 (3dp)",
        errors,
    )

    _require(
        paper_text,
        rf"VCG-only\s*&\s*100\\%\s*&\s*100\\%\s*&\s*{vcg_hybrid_simvalid_num}\\%\s*&\s*{vcg_reward_paper}",
        "Paper VCG-only hybrid row does not match hybrid_phase14",
        errors,
    )
    _require(
        paper_text,
        rf"CCFM-only\s*&\s*100\\%\s*&\s*100\\%\s*&\s*{ccfm_hybrid_simvalid_num}\\%\s*&\s*{ccfm_reward_paper}",
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
    print(f" - README graph validity: VCG={vcg_struct} ({vcg_topos}), CCFM={ccfm_struct} ({ccfm_topos})")
    print(
        " - Hybrid summary: "
        f"vcg_sim_valid={vcg_hybrid_simvalid}, ccfm_sim_valid={ccfm_hybrid_simvalid}, "
        f"hybrid_sim_valid={hybrid_simvalid}, reward={hybrid_reward_readme}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
