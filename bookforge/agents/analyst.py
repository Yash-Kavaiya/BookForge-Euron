"""Transcript analyst (LLM, structured output) with an on-disk analysis cache.

The cache key covers the transcript text, the analyst model and the analyst
instruction, so a resumed run reuses the analysis only when nothing that
could change it has changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from bookforge.agents.model_resolver import resolve_model
from bookforge.config import Settings, get_settings
from bookforge.instrumentation import llm_callbacks
from bookforge.prompts import analyst_instruction
from bookforge.schemas import ChapterAnalysis, VideoRecord
from bookforge.tools.workspace import Workspace

logger = logging.getLogger(__name__)

ANALYSIS_CACHE_FILE = "analysis.json"


def analysis_cache_key(transcript: str, settings: Settings) -> str:
    h = hashlib.sha256()
    h.update((transcript or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(settings.model_for_stage("analyst").encode("utf-8"))
    h.update(b"\x00")
    h.update(analyst_instruction(settings.visual_caps).encode("utf-8"))
    return h.hexdigest()[:24]


def load_cached_analysis(chapter_dir: Path, key: str) -> dict | None:
    path = Path(chapter_dir) / ANALYSIS_CACHE_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("cache_key") != key:
        return None
    analysis = payload.get("analysis")
    try:
        ChapterAnalysis.model_validate(analysis)
    except Exception:  # noqa: BLE001 - a stale/invalid cache is just a miss
        return None
    return analysis


def store_cached_analysis(chapter_dir: Path, key: str, analysis: dict) -> Path:
    payload = {"cache_key": key, "analysis": analysis}
    return Workspace.write_text(
        Path(chapter_dir) / ANALYSIS_CACHE_FILE,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )


def reuse_cached_analysis(callback_context: CallbackContext) -> types.Content | None:
    """before_agent_callback: skip the LLM when this transcript was analysed before."""
    settings = get_settings()
    if not settings.reuse_cached_analysis:
        return None
    state = callback_context.state
    try:
        record = VideoRecord.model_validate(state["current_video"])
        ws = Workspace(settings.workspace_root_abs, state["channel_slug"])
    except Exception:  # noqa: BLE001 - outside the pipeline (e.g. adk web ad-hoc chat)
        return None
    key = analysis_cache_key(state.get("transcript", ""), settings)
    cached = load_cached_analysis(ws.chapter_dir(record), key)
    if cached is None:
        return None
    state["analysis_json"] = cached
    hits = list(state.get("cache_hits") or [])
    hits.append("analysis_json")
    state["cache_hits"] = hits
    state["analysis_cache_key"] = key
    logger.info("analyst skipped: cached analysis for %s (key %s)", record.video_id, key)
    # ADK saves this text under output_key (validated against output_schema),
    # so the skip content must be the analysis itself.
    return types.Content(
        role="model", parts=[types.Part(text=json.dumps(cached, ensure_ascii=False))]
    )


def make_analyst_agent(settings: Settings) -> LlmAgent:
    callbacks = llm_callbacks()
    callbacks["before_agent_callback"] = [
        reuse_cached_analysis,
        *callbacks["before_agent_callback"],
    ]
    return LlmAgent(
        name="transcript_analyst",
        model=resolve_model(settings.model_for_stage("analyst"), settings),
        description=(
            "Reads the timestamped transcript and emits a structured chapter "
            "analysis: objectives, concepts, table/chart/diagram specs, "
            "glossary and exercises."
        ),
        instruction=analyst_instruction(settings.visual_caps),
        output_schema=ChapterAnalysis,
        output_key="analysis_json",
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.temperature_for_stage("analyst"),
        ),
        disallow_transfer_to_parent=True,
        **callbacks,
    )
