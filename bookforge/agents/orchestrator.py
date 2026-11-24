"""BookProductionAgent (resumable per-video loop) + root graph builder.

Media for upcoming chapters is prefetched in parallel (thread pool) while the
current chapter runs the LLM/QA stages. The ADK session graph itself stays
sequential so session state cannot be clobbered.

Per chapter the loop also:
* resumes a written-but-unverified chapter straight into the QA loop,
* re-runs the deterministic machine checks on the final text (the last
  refiner output is otherwise never verified) before deciding the status,
* snapshots stage timings / LLM usage into chapters/<NN>/metrics.json and the
  run-level run_metrics.json, and forwards them to MLflow when tracing is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from bookforge.agents.analyst import make_analyst_agent
from bookforge.agents.assets import VisualAssetAgent
from bookforge.agents.compiler import BookCompilerAgent
from bookforge.agents.critic import make_qa_loop
from bookforge.agents.intake import ChannelIntakeAgent
from bookforge.agents.media import MediaAcquisitionAgent
from bookforge.agents.verifier import (
    CRITIQUE_REPORT_KEY,
    QA_CRITIC_RUNS_KEY,
    QA_ITERATION_KEY,
    QA_MACHINE_OK_KEY,
    QA_REPORT_KEY,
    llm_critic_needed,
)
from bookforge.agents.writer import (
    CHAPTER_TEX_CACHED_KEY,
    QA_NOTE_KEY,
    make_writer_agent,
)
from bookforge.config import Settings, get_settings
from bookforge.instrumentation import (
    LLM_USAGE_KEY,
    STAGE_TIMINGS_KEY,
    stage_seconds,
    summarize_usage,
)
from bookforge.integrations.mlflow_integration import (
    log_chapter_metrics,
    log_run_metrics,
)
from bookforge.progress import ProgressTracker, format_duration
from bookforge.schemas import (
    ChapterMetrics,
    QAReport,
    VideoRecord,
    coerce_analysis,
    coerce_assets_manifest,
)
from bookforge.tools import latex as latex_tools
from bookforge.tools import qa_checks
from bookforge.tools.media_acquire import acquire_media_bundle
from bookforge.tools.workspace import Workspace
from bookforge.tools.workspace_lock import safe_update_video

logger = logging.getLogger(__name__)

# State keys reset at the start of every chapter so nothing leaks between videos.
_PER_CHAPTER_RESET: dict[str, object] = {
    "qa_verdict": "unknown",
    "chapter_tex": "",
    "critique": "",
    CRITIQUE_REPORT_KEY: "",
    QA_REPORT_KEY: "",
    QA_MACHINE_OK_KEY: False,
    QA_ITERATION_KEY: 0,
    QA_CRITIC_RUNS_KEY: 0,
    CHAPTER_TEX_CACHED_KEY: "",
    QA_NOTE_KEY: "",
    STAGE_TIMINGS_KEY: {},
    LLM_USAGE_KEY: {},
    "cache_hits": [],
}


def _text_event(author: str, text: str) -> Event:
    return Event(
        author=author,
        content=types.Content(role="model", parts=[types.Part(text=text)]),
    )


class BookProductionAgent(BaseAgent):
    """Runs chapter_pipeline once per pending video, with checkpointing.

    A video that raises is marked `failed` in the manifest and the loop
    continues — one broken video never blocks the book. Re-running the agent
    skips every video already `verified`.
    """

    name: str = "book_production"
    description: str = "Resumable loop: one book chapter per pending video."
    pipeline: SequentialAgent

    def __init__(self, pipeline: SequentialAgent, **kwargs) -> None:
        super().__init__(pipeline=pipeline, sub_agents=[pipeline], **kwargs)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        settings = get_settings()
        state = ctx.session.state
        ws = Workspace(settings.workspace_root_abs, state["channel_slug"])
        pending = ws.pending_videos()
        tracker = ProgressTracker(len(pending))
        prefetch_n = max(1, settings.media_prefetch_concurrency)

        yield _text_event(
            self.name,
            f"Producing {len(pending)} chapters. "
            f"Media prefetch x{prefetch_n} in parallel. "
            f"Initial ETA ~{format_duration(tracker.eta_sec())} "
            f"(~{format_duration(tracker.average_chapter_sec())}/chapter).",
        )

        prefetch_task = asyncio.create_task(
            self._prefetch_media(ws, pending[1:], prefetch_n)
        )
        try:
            async for event in self._run_sequential(ctx, ws, pending, tracker, settings):
                yield event
        finally:
            if not prefetch_task.done():
                prefetch_task.cancel()
                try:
                    await prefetch_task
                except (asyncio.CancelledError, Exception):
                    pass

        run_metrics = tracker.run_metrics()
        Workspace.write_text(
            ws.root / "run_metrics.json", json.dumps(run_metrics, indent=2)
        )
        log_run_metrics(run_metrics, run_name=f"bookforge-{state['channel_slug']}")

        yield _text_event(self.name, tracker.summary())
        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "production_done": True,
                    "run_metrics": {k: v for k, v in run_metrics.items() if k != "chapters"},
                }
            ),
        )

    async def _prefetch_media(
        self, ws: Workspace, records: list[VideoRecord], concurrency: int
    ) -> None:
        if not records:
            return
        settings = get_settings()
        sem = asyncio.Semaphore(max(1, concurrency))

        async def one(record: VideoRecord) -> None:
            async with sem:
                try:
                    await asyncio.to_thread(
                        acquire_media_bundle, record, ws, settings
                    )
                    logger.info("prefetched media for %s", record.video_id)
                except Exception:
                    logger.exception("media prefetch failed for %s", record.video_id)

        await asyncio.gather(*(one(r) for r in records), return_exceptions=True)

    async def _final_verification(
        self,
        state,
        ws: Workspace,
        record: VideoRecord,
        tex: str,
        settings: Settings,
    ) -> tuple[str, QAReport | None]:
        """Decide the manifest status for the final chapter text.

        The QA loop's last refiner output is never re-verified inside the loop,
        so unless the loop already approved the chapter we run the machine
        checks once more here (deterministic, no LLM).
        """
        if state.get("qa_verdict") == "pass":
            return "verified", None
        analysis = assets = None
        try:
            analysis = coerce_analysis(state["analysis_json"])
            assets = coerce_assets_manifest(state["assets_manifest"])
        except Exception:  # noqa: BLE001 - checks degrade gracefully without them
            pass
        report = await asyncio.to_thread(
            qa_checks.run_machine_checks,
            tex,
            analysis,
            assets,
            state.get("transcript", ""),
            ws,
            record,
            settings,
            int(state.get(QA_ITERATION_KEY) or 0) + 1,
        )
        Workspace.write_text(
            ws.chapter_dir(record) / "qa_report_final.json",
            report.model_dump_json(indent=2),
        )
        critic_runs = int(state.get(QA_CRITIC_RUNS_KEY) or 0)
        if report.passed and not llm_critic_needed(settings.qa_llm_critic_mode, critic_runs):
            state["qa_verdict"] = "pass"
            return "verified", report
        return "written", report

    def _chapter_metrics(
        self,
        state,
        record: VideoRecord,
        status: str,
        wall_sec: float,
        final_report: QAReport | None,
    ) -> ChapterMetrics:
        calls, prompt, completion = summarize_usage(state.get(LLM_USAGE_KEY))
        critic_calls = int(state.get(QA_CRITIC_RUNS_KEY) or 0)
        defects = 0
        if final_report is not None:
            defects = len(final_report.blocking)
        else:
            raw = state.get(QA_REPORT_KEY)
            if isinstance(raw, dict):
                defects = int(raw.get("metrics", {}).get("blocking_defects", 0) or 0)
        return ChapterMetrics(
            video_id=record.video_id,
            chapter_number=record.chapter_number,
            status=status,
            wall_sec=round(wall_sec, 2),
            stage_sec=stage_seconds(state.get(STAGE_TIMINGS_KEY)),
            llm_calls=calls,
            prompt_tokens=prompt,
            completion_tokens=completion,
            qa_iterations=int(state.get(QA_ITERATION_KEY) or 0),
            qa_llm_critic_calls=critic_calls,
            qa_defects_final=defects,
            cache_hits=list(state.get("cache_hits") or []),
        )

    async def _run_sequential(
        self,
        ctx: InvocationContext,
        ws: Workspace,
        pending: list[VideoRecord],
        tracker: ProgressTracker,
        settings: Settings,
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        for index, record in enumerate(pending):
            state["current_video"] = record.model_dump()
            state["video_id"] = record.video_id
            state["video_title"] = record.title
            for key, value in _PER_CHAPTER_RESET.items():
                state[key] = value.copy() if isinstance(value, (dict, list)) else value

            resumed = ""
            if settings.reuse_written_chapters and record.status == "written":
                existing = ws.chapter_dir(record) / "chapter.tex"
                if existing.exists():
                    cached = latex_tools.sanitize_chapter_tex(Workspace.read_text(existing))
                    if cached.strip():
                        state[CHAPTER_TEX_CACHED_KEY] = cached
                        resumed = " Resuming previously written chapter in the QA loop."

            yield _text_event(
                self.name,
                tracker.start_chapter(index, record.title, extra=resumed.strip()),
            )

            status = "failed"
            final_report: QAReport | None = None
            chapter_started = time.monotonic()
            try:
                async for event in self.pipeline.run_async(ctx):
                    yield event

                tex = latex_tools.sanitize_chapter_tex(state.get("chapter_tex", ""))
                if not tex.strip():
                    raise RuntimeError("pipeline produced an empty chapter")
                try:
                    tex, added = latex_tools.ensure_figures_included(
                        tex, coerce_assets_manifest(state.get("assets_manifest"))
                    )
                    if added:
                        logger.info("inserted %d visual(s) the writer left out: %s",
                                    len(added), ", ".join(added))
                except Exception:  # noqa: BLE001 - never lose a chapter over this
                    logger.exception("could not insert missing visuals for %s", record.video_id)
                Workspace.write_text(ws.chapter_dir(record) / "chapter.tex", tex)

                status, final_report = await self._final_verification(
                    state, ws, record, tex, settings
                )
                await safe_update_video(ws, record.video_id, status=status, error="")
                notes = []
                if state.get(QA_NOTE_KEY):
                    notes.append(str(state[QA_NOTE_KEY]))
                if final_report is not None and status == "written":
                    notes.append(
                        f"Final machine check: {len(final_report.blocking)} blocking "
                        f"defect(s) remain after {settings.qa_max_iterations} QA iterations."
                    )
                hits = list(state.get("cache_hits") or [])
                if hits:
                    notes.append("Cache hits: " + ", ".join(hits) + ".")
                yield _text_event(
                    self.name,
                    " ".join([tracker.finish_chapter(index, record.title, status), *notes]),
                )
            except Exception as exc:
                logger.exception("chapter pipeline failed for %s", record.video_id)
                await safe_update_video(
                    ws, record.video_id, status="failed", error=str(exc)[:500]
                )
                yield _text_event(
                    self.name,
                    tracker.finish_chapter(index, record.title, "failed")
                    + f" Error: {exc}",
                )
            finally:
                wall = time.monotonic() - chapter_started
                metrics = self._chapter_metrics(state, record, status, wall, final_report)
                tracker.record_chapter(metrics)
                try:
                    Workspace.write_text(
                        ws.chapter_dir(record) / "metrics.json",
                        metrics.model_dump_json(indent=2),
                    )
                except OSError:
                    logger.debug("could not write metrics.json for %s", record.video_id)
                log_chapter_metrics(metrics.model_dump(mode="json"))


def build_chapter_pipeline(settings: Settings) -> SequentialAgent:
    return SequentialAgent(
        name="chapter_pipeline",
        description="One video -> one verified book chapter.",
        sub_agents=[
            MediaAcquisitionAgent(),
            make_analyst_agent(settings),
            VisualAssetAgent(),
            make_writer_agent(settings),
            make_qa_loop(settings),
        ],
    )


def build_root_agent(settings: Settings | None = None) -> SequentialAgent:
    """The BookForge root agent (used by `adk run`, `adk web`, and the CLI)."""
    settings = settings or get_settings()
    return SequentialAgent(
        name="bookforge",
        description=(
            "Turns a YouTube channel URL into a compiled LaTeX book: "
            "intake -> per-video chapters (media, analysis, assets, writing, "
            "QA loop) -> final book compilation."
        ),
        sub_agents=[
            ChannelIntakeAgent(),
            BookProductionAgent(pipeline=build_chapter_pipeline(settings)),
            BookCompilerAgent(),
        ],
    )
