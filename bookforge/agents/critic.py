"""Chapter QA loop: deterministic verifier -> LLM critic -> refiner, bounded.

Iteration shape (LoopAgent, max_iterations = qa_max_iterations):

    chapter_verifier  machine checks + pdflatex; approves, or writes the exact
                      defect list to `critique` and marks qa_machine_ok=False
    chapter_critic    LLM judgement (structured CritiqueReport). Runs ONLY when
                      machine checks passed and settings.qa_llm_critic_mode
                      still asks for it; otherwise skipped via callback.
    chapter_refiner   rewrites to fix `critique`. Skipped (and the loop exits)
                      when the critic's verdict is pass.

The LLM critic therefore never pays to re-discover mechanical defects, and a
clean chapter costs at most one critic call per chapter by default.
"""

from __future__ import annotations

import logging

from google.adk.agents import LlmAgent, LoopAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from bookforge.agents.model_resolver import resolve_model
from bookforge.agents.verifier import (
    CRITIQUE_REPORT_KEY,
    QA_CRITIC_RUNS_KEY,
    QA_MACHINE_OK_KEY,
    ChapterVerifierAgent,
    llm_critic_needed,
)
from bookforge.agents.writer import make_refiner_agent
from bookforge.config import Settings, get_settings
from bookforge.instrumentation import llm_callbacks
from bookforge.prompts import CRITIC_INSTRUCTION
from bookforge.schemas import CritiqueReport

logger = logging.getLogger(__name__)


def _skip(verdict: str, summary: str) -> types.Content:
    """Skip content for an LlmAgent with output_schema/output_key.

    ADK saves a before_agent_callback's text into the agent's output_key (and
    validates it against output_schema), so the skip message must itself be a
    valid CritiqueReport. The refiner gate treats a defect-less report as
    "nothing new": a fail keeps the machine defect list, a pass escalates.
    """
    report = CritiqueReport(verdict=verdict, summary=summary, defects=[])  # type: ignore[arg-type]
    return types.Content(role="model", parts=[types.Part(text=report.model_dump_json())])


def skip_critic_unless_machine_ok(
    callback_context: CallbackContext,
) -> types.Content | None:
    """before_agent_callback: only spend an LLM call when the machine gate is clean."""
    state = callback_context.state
    if not state.get(QA_MACHINE_OK_KEY):
        return _skip(
            "fail", "LLM critic skipped: machine verification already produced the fix list."
        )
    settings = get_settings()
    runs = int(state.get(QA_CRITIC_RUNS_KEY) or 0)
    if not llm_critic_needed(settings.qa_llm_critic_mode, runs):
        return _skip("pass", "LLM critic skipped: machine checks passed and critic budget is spent.")
    state[QA_CRITIC_RUNS_KEY] = runs + 1
    return None


def make_critic_agent(settings: Settings) -> LlmAgent:
    callbacks = llm_callbacks()
    callbacks["before_agent_callback"] = [
        skip_critic_unless_machine_ok,
        *callbacks["before_agent_callback"],
    ]
    return LlmAgent(
        name="chapter_critic",
        model=resolve_model(settings.model_for_stage("critic"), settings),
        description=(
            "Judges faithfulness, completeness and pedagogy of a chapter that "
            "already passed machine verification; emits a structured verdict."
        ),
        instruction=CRITIC_INSTRUCTION,
        output_schema=CritiqueReport,
        output_key=CRITIQUE_REPORT_KEY,
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.temperature_for_stage("critic"),
        ),
        disallow_transfer_to_parent=True,
        **callbacks,
    )


def make_qa_loop(settings: Settings) -> LoopAgent:
    return LoopAgent(
        name="chapter_qa_loop",
        description=(
            "Bounded verify-and-refine loop: deterministic verifier, LLM critic "
            "(only on a clean machine pass), refiner."
        ),
        sub_agents=[
            ChapterVerifierAgent(),
            make_critic_agent(settings),
            make_refiner_agent(settings),
        ],
        max_iterations=settings.qa_max_iterations,
    )
