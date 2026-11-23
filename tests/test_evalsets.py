"""Eval sets stay valid, and the offline verifier cases are executable here:
the inline chapter/state of every `verifier_*` case is run through the real
deterministic machine checks and must produce the expected outcome."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bookforge.config import Settings
from bookforge.schemas import coerce_analysis, coerce_assets_manifest
from bookforge.tools import qa_checks

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))
from validate_evalset import validate_evalset  # noqa: E402


@pytest.mark.parametrize("name", ["bookforge.evalset.json", "bookforge-comprehensive.evalset.json"])
def test_evalset_files_are_valid(name):
    ok, errors = validate_evalset(EVAL_DIR / name)
    assert ok, errors


def _comprehensive() -> dict:
    return json.loads((EVAL_DIR / "bookforge-comprehensive.evalset.json").read_text(encoding="utf-8"))


def test_comprehensive_covers_every_stage():
    ids = {c["eval_id"] for c in _comprehensive()["eval_cases"]}
    prefixes = {i.split("_")[0] for i in ids}
    assert {
        "intake", "media", "analyst", "assets", "writer", "verifier", "critic",
        "refiner", "cache", "compiler", "orchestration", "metrics", "config", "e2e",
    } <= prefixes
    # the old free-form tool protocol is gone from the evals too
    assert not any("compile_chapter" in i or "approve_chapter" in i for i in ids)


def _verifier_cases():
    return [c for c in _comprehensive()["eval_cases"] if c["eval_id"].startswith("verifier_")]


@pytest.mark.parametrize("case", _verifier_cases(), ids=lambda c: c["eval_id"])
def test_verifier_eval_cases_run_offline(case, tmp_path):
    state = case["session_input"]["state"]
    analysis = coerce_analysis(state["analysis_json"])
    assets = coerce_assets_manifest(state["assets_manifest"])
    settings = Settings(_env_file=None, compile_latex=False, workspace_root=tmp_path)
    report = qa_checks.run_machine_checks(
        state["chapter_tex"], analysis, assets, state["transcript"], None, None, settings, 1
    )
    expected = case["conversation"][0]["final_response"]["parts"][0]["text"]
    if "passed" in expected:
        assert report.passed, [d.message for d in report.defects]
    else:
        assert not report.passed
        codes = {d.code for d in report.blocking}
        if "missing_asset" in case["eval_id"]:
            assert "unknown_asset" in codes
        if "unsupported_number" in case["eval_id"]:
            assert "unsupported_number" in codes
        if "missing_sections" in case["eval_id"]:
            assert "missing_section" in codes
