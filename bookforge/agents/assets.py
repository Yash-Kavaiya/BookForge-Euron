"""VisualAssetAgent: turn the analysis spec into real files on disk.

Deterministic by design: charts are rendered by matplotlib from the analyst's
validated data (never hand-drawn by an LLM), tables become booktabs fragments,
and the best frames are copied into the chapter figures directory.

All rendering runs in a thread pool (charts, tables and frame curation are
independent), and the event loop is never blocked, so media prefetch for the
next chapters keeps flowing while this stage works.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from bookforge.agents.analyst import analysis_cache_key, store_cached_analysis
from bookforge.config import Settings, get_settings
from bookforge.instrumentation import agent_callbacks
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    FigureAsset,
    MediaBundle,
    TableAsset,
    VideoRecord,
    coerce_analysis,
)
from bookforge.tools import frames as frame_tools
from bookforge.tools import latex as latex_tools
from bookforge.tools import visuals
from bookforge.tools.workspace import Workspace
from bookforge.tools.workspace_lock import safe_update_video

logger = logging.getLogger(__name__)


def _mm_ss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _section_for(text: str, analysis: ChapterAnalysis) -> str:
    """Concept heading that best matches a visual's title/caption (or "")."""
    words = {w for w in visuals._norm(text).split() if len(w) > 3}
    best, best_overlap = "", 0
    for concept in analysis.concepts:
        overlap = len(words & {w for w in visuals._norm(concept.heading).split() if len(w) > 3})
        if overlap > best_overlap:
            best, best_overlap = concept.heading, overlap
    return best


def render_assets(
    analysis: ChapterAnalysis,
    record: VideoRecord,
    ws: Workspace,
    settings: Settings,
    transcript: str = "",
) -> tuple[AssetsManifest, int]:
    """Render every table/chart/diagram/frame asset for one chapter.

    Blocking and thread-safe. Every visual is optional: a chart whose numbers
    are not stated in the transcript, or a diagram/chart that fails to render,
    is dropped and logged; the chapter never fails because of a visual.
    Returns (manifest, curated_frame_count).
    """
    vdir = ws.video_dir(record.video_id)
    chapter_dir = ws.chapter_dir(record)
    nn = record.chapter_number
    caps = settings.visual_caps

    def rel(path: Path) -> str:
        return path.relative_to(ws.root).as_posix()

    def table_job(i: int, spec) -> TableAsset:
        label = f"tab:ch{nn:02d}-{i}"
        frag = latex_tools.render_table_fragment(
            spec, chapter_dir / "tables" / f"table_{i}.tex", label=label
        )
        return TableAsset(tex_file=rel(frag), caption=spec.caption, label=label)

    def chart_job(i: int, spec) -> FigureAsset | None:
        if transcript:
            ok, source_ts, reason = visuals.verify_chart_against_transcript(spec, transcript)
            if not ok:
                logger.warning("chart '%s' dropped (ch %d): %s", spec.title, nn, reason)
                return None
        else:
            source_ts = spec.source_timestamp
        try:
            pdf = latex_tools.render_chart(spec, chapter_dir / "figures" / f"chart_{i}.pdf")
        except Exception:  # noqa: BLE001 - drop the chart, keep the chapter
            logger.exception("chart '%s' dropped (ch %d): render failed", spec.title, nn)
            return None
        return FigureAsset(
            filename=rel(pdf),
            kind="chart",
            caption=visuals.chart_caption(spec, source_ts),
            label=f"fig:ch{nn:02d}-chart{i}",
            section=_section_for(f"{spec.title} {spec.caption}", analysis),
        )

    def diagram_job(i: int, spec) -> FigureAsset | None:
        pdf = visuals.render_diagram(
            spec, chapter_dir / "figures" / f"diagram_{i}.pdf", engine=settings.diagram_engine
        )
        if pdf is None:
            return None
        return FigureAsset(
            filename=rel(pdf),
            kind="diagram",
            caption=spec.caption or spec.title,
            label=f"fig:ch{nn:02d}-diagram{i}",
            section=_section_for(f"{spec.title} {spec.caption}", analysis),
        )

    def frames_job() -> list[FigureAsset]:
        media_json = vdir / "media.json"
        if not media_json.exists():
            logger.warning("media.json missing for %s; no frames curated", record.video_id)
            return []
        bundle = MediaBundle.model_validate_json(media_json.read_text(encoding="utf-8"))
        # Fewer, more informative screenshots: best-scoring frame (slides,
        # code, diagrams beat talking heads) in each timeline bucket.
        scored = [
            (f.timestamp_sec, visuals.frame_information_score(Path(f.file)), f)
            for f in bundle.frames
        ]
        best = visuals.pick_informative(scored, caps["frames"])
        curated = frame_tools.curate_frames(best, vdir, chapter_dir / "figures", len(best) or 1)
        overrides = analysis.frame_captions_dict
        out: list[FigureAsset] = []
        for j, frame in enumerate(curated, start=1):
            caption = overrides.get(frame.file) or (
                f"Video frame at {_mm_ss(frame.timestamp_sec)} illustrating "
                f"the segment discussed in this chapter."
            )
            out.append(
                FigureAsset(
                    filename=rel(chapter_dir / "figures" / frame.file),
                    kind="frame",
                    caption=caption,
                    label=f"fig:ch{nn:02d}-frame{j}",
                )
            )
        return out

    workers = max(1, settings.asset_render_workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bf-assets") as pool:
        table_futs = [pool.submit(table_job, i, s) for i, s in enumerate(analysis.tables, 1)]
        chart_futs = [
            pool.submit(chart_job, i, s)
            for i, s in enumerate(analysis.charts[: caps["charts"]], 1)
        ]
        diagram_futs = [
            pool.submit(diagram_job, i, s)
            for i, s in enumerate(analysis.diagrams[: caps["diagrams"]], 1)
        ]
        frames_fut = pool.submit(frames_job)
        tables = [f.result() for f in table_futs]
        figures: list[FigureAsset] = [
            fig for fig in (f.result() for f in chart_futs + diagram_futs) if fig is not None
        ]
        try:
            frame_figs = frames_fut.result()
        except Exception:  # noqa: BLE001 - screenshots are optional
            logger.exception("frame curation failed for %s", record.video_id)
            frame_figs = []
    figures.extend(frame_figs)

    manifest = AssetsManifest(
        chapter_slug=chapter_dir.name,
        chapter_dir=rel(chapter_dir),
        figures=figures,
        tables=tables,
    )
    Workspace.write_text(chapter_dir / "assets.json", manifest.model_dump_json(indent=2))
    return manifest, len(frame_figs)


async def render_illustrations(
    analysis: ChapterAnalysis,
    record: VideoRecord,
    ws: Workspace,
    settings: Settings,
) -> list[FigureAsset]:
    """AI illustrations (euri-image-fast) for the sections the analyst chose.

    Sequential on purpose (EURI allows 2 concurrent requests and the chat
    agents share that budget). A failed image is dropped and logged.
    """
    cap = settings.visual_caps["illustrations"]
    if cap <= 0 or not analysis.illustrations:
        return []
    chapter_dir = ws.chapter_dir(record)
    nn = record.chapter_number
    out: list[FigureAsset] = []
    for i, spec in enumerate(analysis.illustrations[:cap], start=1):
        dest = chapter_dir / "figures" / f"illustration_{i}.png"
        path = await visuals.generate_image(
            visuals.illustration_prompt(spec, analysis.chapter_title),
            dest,
            api_key=settings.euri_api_key,
            api_base=settings.euri_api_base,
            model=settings.image_model,
            max_concurrency=settings.llm_max_concurrency,
        )
        if path is None:
            continue
        caption = spec.caption.strip().rstrip(".") + " (AI-generated illustration)."
        out.append(
            FigureAsset(
                filename=path.relative_to(ws.root).as_posix(),
                kind="illustration",
                caption=caption,
                label=f"fig:ch{nn:02d}-illus{i}",
                section=spec.section,
            )
        )
    return out


class VisualAssetAgent(BaseAgent):
    name: str = "visual_assets"
    description: str = (
        "Renders charts (matplotlib), booktabs table fragments and curated "
        "screenshots into the chapter directory and publishes an assets "
        "manifest the writer must reference verbatim."
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
        analysis = coerce_analysis(state["analysis_json"])
        ws = Workspace(settings.workspace_root_abs, state["channel_slug"])

        # Persist the (validated) analysis so a rerun never pays the analyst twice.
        if settings.reuse_cached_analysis:
            key = analysis_cache_key(state.get("transcript", ""), settings)
            await asyncio.to_thread(
                store_cached_analysis,
                ws.chapter_dir(record),
                key,
                analysis.model_dump(mode="json"),
            )

        manifest, n_frames = await asyncio.to_thread(
            render_assets, analysis, record, ws, settings, state.get("transcript", "")
        )
        illus = await render_illustrations(analysis, record, ws, settings)
        if illus:
            manifest.figures.extend(illus)
            Workspace.write_text(
                ws.chapter_dir(record) / "assets.json", manifest.model_dump_json(indent=2)
            )
        kinds: dict[str, int] = {}
        for fig in manifest.figures:
            kinds[fig.kind] = kinds.get(fig.kind, 0) + 1
        await safe_update_video(ws, record.video_id, status="assets", error="")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=f"Assets for chapter {record.chapter_number}: "
                        f"{len(manifest.figures)} figures ({n_frames} frames, "
                        f"{kinds.get('chart', 0)} charts, {kinds.get('diagram', 0)} diagrams, "
                        f"{kinds.get('illustration', 0)} illustrations), "
                        f"{len(manifest.tables)} tables."
                    )
                ],
            ),
            actions=EventActions(
                state_delta={"assets_manifest": manifest.model_dump(mode="json")}
            ),
        )
