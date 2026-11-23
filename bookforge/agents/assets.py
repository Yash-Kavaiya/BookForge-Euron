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
from bookforge.tools.workspace import Workspace
from bookforge.tools.workspace_lock import safe_update_video

logger = logging.getLogger(__name__)


def _mm_ss(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def render_assets(
    analysis: ChapterAnalysis,
    record: VideoRecord,
    ws: Workspace,
    settings: Settings,
) -> tuple[AssetsManifest, int]:
    """Render every table/chart/frame asset for one chapter (blocking, thread-safe).

    Returns (manifest, curated_frame_count). Safe to call from a worker thread.
    """
    vdir = ws.video_dir(record.video_id)
    chapter_dir = ws.chapter_dir(record)
    nn = record.chapter_number

    def rel(path: Path) -> str:
        return path.relative_to(ws.root).as_posix()

    def table_job(i: int, spec) -> TableAsset:
        label = f"tab:ch{nn:02d}-{i}"
        frag = latex_tools.render_table_fragment(
            spec, chapter_dir / "tables" / f"table_{i}.tex", label=label
        )
        return TableAsset(tex_file=rel(frag), caption=spec.caption, label=label)

    def chart_job(i: int, spec) -> FigureAsset:
        pdf = latex_tools.render_chart(spec, chapter_dir / "figures" / f"chart_{i}.pdf")
        return FigureAsset(
            filename=rel(pdf),
            kind="chart",
            caption=spec.caption,
            label=f"fig:ch{nn:02d}-chart{i}",
        )

    def frames_job() -> list[FigureAsset]:
        media_json = vdir / "media.json"
        if not media_json.exists():
            logger.warning("media.json missing for %s; no frames curated", record.video_id)
            return []
        bundle = MediaBundle.model_validate_json(media_json.read_text(encoding="utf-8"))
        curated = frame_tools.curate_frames(
            bundle.frames, vdir, chapter_dir / "figures", settings.max_frames_per_chapter
        )
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
        chart_futs = [pool.submit(chart_job, i, s) for i, s in enumerate(analysis.charts, 1)]
        frames_fut = pool.submit(frames_job)
        tables = [f.result() for f in table_futs]
        figures: list[FigureAsset] = [f.result() for f in chart_futs]
        frame_figs = frames_fut.result()
    figures.extend(frame_figs)

    manifest = AssetsManifest(
        chapter_slug=chapter_dir.name,
        chapter_dir=rel(chapter_dir),
        figures=figures,
        tables=tables,
    )
    Workspace.write_text(chapter_dir / "assets.json", manifest.model_dump_json(indent=2))
    return manifest, len(frame_figs)


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
            render_assets, analysis, record, ws, settings
        )
        await safe_update_video(ws, record.video_id, status="assets", error="")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=f"Assets for chapter {record.chapter_number}: "
                        f"{len(manifest.figures)} figures ({n_frames} frames, "
                        f"{len(analysis.charts)} charts), {len(manifest.tables)} tables."
                    )
                ],
            ),
            actions=EventActions(
                state_delta={"assets_manifest": manifest.model_dump(mode="json")}
            ),
        )
