"""ChapterVerifierAgent: the deterministic gate at the top of the QA loop.

Runs the machine checks (structure, LaTeX balance, asset/label integrity,
unsupported numbers, promised equations/concepts/diagrams, pdflatex) and
decides, without an LLM:

* clean + LLM critic not needed  -> qa_verdict=pass, escalate (loop exits)
* clean + LLM critic needed      -> qa_machine_ok=True, critic runs next
* defects                        -> critique = exact defect list, critic is
                                    skipped, refiner runs next
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from bookforge.config import get_settings
from bookforge.instrumentation import agent_callbacks
from bookforge.schemas import (
    QAReport,
    VideoRecord,
    coerce_analysis,
    coerce_assets_manifest,
)
from bookforge.tools import latex as latex_tools
from bookforge.tools import qa_checks
from bookforge.tools.workspace import Workspace

logger = logging.getLogger(__name__)

QA_REPORT_KEY = "qa_report"
QA_MACHINE_OK_KEY = "qa_machine_ok"
QA_ITERATION_KEY = "qa_iteration"
QA_CRITIC_RUNS_KEY = "qa_llm_critic_runs"
CRITIQUE_REPORT_KEY = "critique_report"


def llm_critic_needed(mode: str, critic_runs: int) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    return critic_runs < 1  # "once"


class ChapterVerifierAgent(BaseAgent):
    name: str = "chapter_verifier"
    description: str = (
        "Deterministically verifies the chapter (structure, asset and label "
        "integrity, unsupported numbers, promised equations, pdflatex) and "
        "either approves it, hands it to the LLM critic, or sends an exact "
        "defect list to the refiner."
    )

    def __init__(self, **kwargs) -> None:
        kwargs = {**agent_callbacks(), **kwargs}
        super().__init__(**kwargs)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        settings = get_settings()
        state = ctx.session.state
        record = VideoRecord.model_validate(state["current_video"])
        ws = Workspace(settings.workspace_root_abs, state["channel_slug"])
        iteration = int(state.get(QA_ITERATION_KEY) or 0) + 1
        critic_runs = int(state.get(QA_CRITIC_RUNS_KEY) or 0)

        tex = latex_tools.sanitize_chapter_tex(state.get("chapter_tex", ""))
        analysis = None
        assets = None
        try:
            analysis = coerce_analysis(state["analysis_json"])
        except Exception:  # noqa: BLE001 - verifier must never crash the loop
            logger.warning("verifier: analysis_json unreadable; coverage checks skipped")
        try:
            assets = coerce_assets_manifest(state["assets_manifest"])
        except Exception:  # noqa: BLE001
            logger.warning("verifier: assets_manifest unreadable; asset checks skipped")

        report: QAReport = await asyncio.to_thread(
            qa_checks.run_machine_checks,
            tex,
            analysis,
            assets,
            state.get("transcript", ""),
            ws,
            record,
            settings,
            iteration,
        )
        if tex.strip():
            Workspace.write_text(ws.chapter_dir(record) / "chapter.tex", tex)
        Workspace.write_text(
            ws.chapter_dir(record) / f"qa_report_{iteration}.json",
            report.model_dump_json(indent=2),
        )

        delta: dict = {
            QA_REPORT_KEY: report.model_dump(mode="json"),
            QA_ITERATION_KEY: iteration,
            CRITIQUE_REPORT_KEY: "",  # stale LLM verdicts must never leak across iterations
            "chapter_tex": tex,
        }
        escalate = False
        n_block = len(report.blocking)
        n_minor = len(report.defects) - n_block

        if report.passed:
            if llm_critic_needed(settings.qa_llm_critic_mode, critic_runs):
                delta[QA_MACHINE_OK_KEY] = True
                delta["critique"] = ""
                text = (
                    f"QA iteration {iteration}: machine checks passed "
                    f"({n_minor} minor notes). Handing to the LLM critic."
                )
            else:
                delta[QA_MACHINE_OK_KEY] = True
                delta["qa_verdict"] = "pass"
                delta["critique"] = ""
                escalate = True
                text = (
                    f"QA iteration {iteration}: machine checks passed "
                    f"({n_minor} minor notes); chapter approved."
                )
        else:
            delta[QA_MACHINE_OK_KEY] = False
            delta["qa_verdict"] = "fail"
            delta["critique"] = qa_checks.format_defects(
                report.defects,
                header=(
                    f"Machine verification (iteration {iteration}) found "
                    f"{n_block} blocking defect(s). Fix every item below."
                ),
            )
            compile_note = (
                "" if report.compile_ok is None else f", compile_ok={report.compile_ok}"
            )
            text = (
                f"QA iteration {iteration}: {n_block} blocking / {n_minor} minor "
                f"defect(s){compile_note}. Sending exact fixes to the refiner "
                f"(LLM critic skipped)."
            )

        # Build the delta completely first: EventActions copies the dict.
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            actions=EventActions(state_delta=delta, escalate=escalate),
        )
