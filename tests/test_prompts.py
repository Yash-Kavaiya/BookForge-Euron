"""Instruction templates must not contain accidental ADK state placeholders,
and must carry the grounding contract the deterministic verifier enforces.

ADK injects session state via {identifier}. A literal TikZ/LaTeX example
like `{label}` raises KeyError: Context variable not found: `label`.
"""

from __future__ import annotations

import re

from bookforge.prompts import (
    ANALYST_INSTRUCTION,
    CRITIC_INSTRUCTION,
    REFINER_INSTRUCTION,
    WRITER_INSTRUCTION,
)
from bookforge.tools.qa_checks import DATA_SECTION, MODELS_SECTION, REQUIRED_SECTIONS

# Same pattern ADK uses in google.adk.utils.instructions_utils
_TEMPLATE_VAR_PATTERN = re.compile(r"{+[^{}]*}+")

ALLOWED_PLACEHOLDERS = {
    "video_title",
    "transcript",
    "analysis_json",
    "assets_manifest",
    "chapter_tex",
    "critique",
    "qa_report",
}

# Every placeholder a prompt uses must be seeded into state before the agent
# runs (orchestrator reset / upstream stage outputs), otherwise ADK raises.
EXPECTED_PLACEHOLDERS = {
    "analyst": {"video_title", "transcript"},
    "writer": {"video_title", "analysis_json", "assets_manifest"},
    "critic": {"video_title", "qa_report", "analysis_json", "chapter_tex", "assets_manifest"},
    "refiner": {"chapter_tex", "critique", "assets_manifest", "analysis_json"},
}

INSTRUCTIONS = {
    "analyst": ANALYST_INSTRUCTION,
    "writer": WRITER_INSTRUCTION,
    "critic": CRITIC_INSTRUCTION,
    "refiner": REFINER_INSTRUCTION,
}


def required_placeholders(template: str) -> set[str]:
    found: set[str] = set()
    for match in _TEMPLATE_VAR_PATTERN.finditer(template):
        name = match.group().lstrip("{").rstrip("}").strip()
        if name.endswith("?"):
            continue
        if name.isidentifier():
            found.add(name)
    return found


def test_writer_has_no_literal_label_placeholder():
    found = required_placeholders(WRITER_INSTRUCTION)
    assert "label" not in found, (
        "WRITER_INSTRUCTION contains `{label}`, which ADK treats as a "
        "session-state injection and crashes chapter writing with "
        "'Context variable not found: label.'"
    )


def test_all_instruction_placeholders_are_allowlisted():
    unexpected: dict[str, set[str]] = {}
    for name, text in INSTRUCTIONS.items():
        extra = required_placeholders(text) - ALLOWED_PLACEHOLDERS
        if extra:
            unexpected[name] = extra
    assert unexpected == {}, f"Accidental ADK placeholders: {unexpected}"


def test_each_prompt_uses_exactly_its_expected_placeholders():
    for name, text in INSTRUCTIONS.items():
        assert required_placeholders(text) == EXPECTED_PLACEHOLDERS[name], name


def test_no_literal_latex_braces_anywhere():
    """Any {word} that is not a known placeholder would crash injection."""
    for name, text in INSTRUCTIONS.items():
        for match in _TEMPLATE_VAR_PATTERN.finditer(text):
            inner = match.group().strip("{}").strip()
            assert inner in ALLOWED_PLACEHOLDERS, f"{name}: literal braces {match.group()!r}"


def test_prompts_carry_the_grounding_contract():
    for text in (ANALYST_INSTRUCTION, WRITER_INSTRUCTION, REFINER_INSTRUCTION):
        assert "only source of facts" in text.lower()
    assert "never round" in ANALYST_INSTRUCTION.lower()
    assert "evidence_timestamps" in ANALYST_INSTRUCTION
    assert "exactly as it appears in the analysis" in WRITER_INSTRUCTION
    assert "verbatim" in WRITER_INSTRUCTION
    assert "faithfulness" in CRITIC_INSTRUCTION.lower()
    assert "unsupported" in REFINER_INSTRUCTION.lower()


def test_writer_and_refiner_name_every_section_the_verifier_requires():
    for title in (*REQUIRED_SECTIONS, DATA_SECTION, MODELS_SECTION):
        assert f'"{title}"' in WRITER_INSTRUCTION, title
    for title in REQUIRED_SECTIONS:
        assert title in REFINER_INSTRUCTION, title


def test_critic_is_told_the_machine_gate_already_ran():
    assert "deterministic verifier" in CRITIC_INSTRUCTION.lower()
    assert "structured verdict" in CRITIC_INSTRUCTION.lower()
    # the critic must not re-flag things the verifier owns or the preamble defines
    for env in ("notebox", "keyinsight", "formulabox"):
        assert env in CRITIC_INSTRUCTION


def test_analyst_requires_english_output_for_non_english_transcripts():
    text = ANALYST_INSTRUCTION.lower()
    assert "hindi" in text, "should mention Hindi as a possible transcript language"
    assert "english" in text, "should require English output"
    assert "translat" in text, "should instruct faithful translation"
    assert "romanised" in text, "should keep names as spoken (romanised)"
    # Placeholder set must still be exactly the original two
    assert required_placeholders(ANALYST_INSTRUCTION) == {"video_title", "transcript"}, (
        "ANALYST_INSTRUCTION placeholders changed; no new {placeholder} was allowed"
    )
