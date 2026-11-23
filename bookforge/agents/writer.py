"""Chapter writer + refiner (LLM) with cache/gate callbacks.

writer   - skipped when the orchestrator resumed a previously written chapter
           (`chapter_tex_cached`), so a chapter that failed QA on an earlier
           run goes straight back into the QA loop instead of being rewritten.
refiner  - skipped (and the QA loop exits) when the LLM critic's structured
           verdict is `pass`; otherwise turns the verdict into the `critique`
           text it rewrites against.
"""

from __future__ import annotations

import logging

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from bookforge.agents.model_resolver import resolve_model
from bookforge.agents.verifier import CRITIQUE_REPORT_KEY, QA_MACHINE_OK_KEY
from bookforge.config import Settings
from bookforge.instrumentation import llm_callbacks
from bookforge.prompts import REFINER_INSTRUCTION, WRITER_INSTRUCTION
from bookforge.schemas import coerce_critique
from bookforge.tools import qa_checks

logger = logging.getLogger(__name__)

CHAPTER_TEX_CACHED_KEY = "chapter_tex_cached"
QA_NOTE_KEY = "qa_note"  # human-readable reason for the last writer/refiner skip


def _chapter_content(tex: str) -> types.Content:
    """Skip content for the writer/refiner.

    ADK stores a before_agent_callback's text under the agent's output_key, so
    the only safe thing to return from these two agents is the chapter body
    itself (cached or unchanged). Explanations go to `qa_note` in state; the
    verifier/orchestrator events tell the story to the user.
    """
    return types.Content(role="model", parts=[types.Part(text=tex)])


def reuse_written_chapter(callback_context: CallbackContext) -> types.Content | None:
    """before_agent_callback: resume a written-but-unverified chapter."""
    state = callback_context.state
    cached = str(state.get(CHAPTER_TEX_CACHED_KEY) or "")
    if not cached.strip():
        return None
    state["chapter_tex"] = cached
    state[CHAPTER_TEX_CACHED_KEY] = ""
    hits = list(state.get("cache_hits") or [])
    hits.append("chapter_tex")
    state["cache_hits"] = hits
    state[QA_NOTE_KEY] = (
        "Writer skipped: reusing the chapter written on a previous run; the QA "
        "loop will verify and refine it."
    )
    return _chapter_content(cached)


def gate_refiner_on_critique(callback_context: CallbackContext) -> types.Content | None:
    """before_agent_callback: exit the loop on a passing LLM verdict, else
    convert the structured critique into the refiner's defect list."""
    state = callback_context.state
    raw = state.get(CRITIQUE_REPORT_KEY)
    if not raw:
        # Critic produced nothing: `critique` already holds the machine defect
        # list (or the loop is in an inconsistent state, in which case
        # refining is still the safe move).
        return None
    try:
        report = coerce_critique(raw)
    except Exception as exc:  # noqa: BLE001 - never let a malformed verdict kill the chapter
        logger.warning("critique_report unparsable (%s); treating as fail", exc)
        state["critique"] = (
            "The critic's response could not be parsed. Re-check the chapter "
            f"against the analysis and fix any defects you find. Raw: {str(raw)[:1500]}"
        )
        return None

    if report.verdict == "pass" and not report.blocking and state.get(QA_MACHINE_OK_KEY):
        state["qa_verdict"] = "pass"
        state["critique"] = ""
        state[QA_NOTE_KEY] = (
            "Chapter approved by the LLM critic after passing machine verification."
            + (f" {report.summary}" if report.summary else "")
        )
        callback_context.actions.escalate = True
        return _chapter_content(str(state.get("chapter_tex") or ""))

    if not report.defects:
        # A skipped critic emits a defect-less fail: the machine defect list in
        # `critique` is already the refiner's to-do list.
        return None
    defects = report.blocking or report.defects
    state["critique"] = qa_checks.format_defects(
        defects,
        header=(
            "LLM critic verdict: FAIL."
            + (f" {report.summary}" if report.summary else "")
            + " Fix every item below while preserving everything else."
        ),
    )
    return None


def make_writer_agent(settings: Settings) -> LlmAgent:
    callbacks = llm_callbacks()
    callbacks["before_agent_callback"] = [
        reuse_written_chapter,
        *callbacks["before_agent_callback"],
    ]
    return LlmAgent(
        name="chapter_writer",
        model=resolve_model(settings.model_for_stage("writer"), settings),
        description=(
            "Composes the full LaTeX chapter body from the validated analysis "
            "and the on-disk assets manifest."
        ),
        instruction=WRITER_INSTRUCTION,
        output_key="chapter_tex",
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.temperature_for_stage("writer"),
        ),
        disallow_transfer_to_parent=True,
        **callbacks,
    )


def make_refiner_agent(settings: Settings) -> LlmAgent:
    callbacks = llm_callbacks()
    callbacks["before_agent_callback"] = [
        gate_refiner_on_critique,
        *callbacks["before_agent_callback"],
    ]
    return LlmAgent(
        name="chapter_refiner",
        model=resolve_model(settings.model_for_stage("refiner"), settings),
        description=(
            "Rewrites the chapter to fix every defect found by the verifier or "
            "critic while preserving what already works."
        ),
        instruction=REFINER_INSTRUCTION,
        output_key="chapter_tex",
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.temperature_for_stage("refiner"),
        ),
        disallow_transfer_to_parent=True,
        **callbacks,
    )
