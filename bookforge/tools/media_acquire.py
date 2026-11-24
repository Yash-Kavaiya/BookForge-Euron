"""Idempotent per-video media acquisition (download, transcript, frames).

Safe to call from a worker thread / prefetch task. Reuses existing files.

The independent halves run concurrently:

    download_video ─┬─> extract_frames -> dedupe_frames
                    └─> (audio -> whisper, only if no captions)
    fetch_captions ───> vtt_to_text

so a video's wall-clock cost is max(download+frames, captions) instead of the
sum, and the whisper fallback overlaps with frame extraction.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bookforge.config import Settings
from bookforge.schemas import FrameAsset, MediaBundle, VideoRecord
from bookforge.tools import frames as frame_tools
from bookforge.tools import youtube as yt
from bookforge.tools.workspace import Workspace

logger = logging.getLogger(__name__)


def _load_cached(vdir: Path, max_chars: int = 120_000) -> tuple[MediaBundle, str] | None:
    existing = vdir / "media.json"
    if not existing.exists():
        return None
    try:
        bundle = MediaBundle.model_validate_json(existing.read_text(encoding="utf-8"))
        if bundle.video_path.endswith(yt._PARTIAL_SUFFIXES):
            return None  # cached from an interrupted download: redo it
        transcript_text = yt.load_transcript_text(Path(bundle.transcript_path), max_chars)
    except (OSError, ValueError):
        return None
    if not transcript_text.strip():
        return None
    return bundle, transcript_text


def _frames_pipeline(video_path: Path, vdir: Path, settings: Settings) -> list[FrameAsset]:
    raw_frames = frame_tools.extract_frames(
        video_path, vdir / "frames_raw", settings.frame_interval_sec
    )
    return frame_tools.dedupe_frames(
        raw_frames,
        interval_sec=settings.frame_interval_sec,
        hash_threshold=settings.frame_phash_threshold,
        workers=settings.frame_hash_workers,
    )


def acquire_media_bundle_ex(
    record: VideoRecord,
    ws: Workspace,
    settings: Settings,
) -> tuple[MediaBundle, str, bool]:
    """Download + transcript + frames. Returns (bundle, transcript_text, cached)."""
    vdir = ws.video_dir(record.video_id)
    cached = _load_cached(vdir, settings.transcript_max_chars)
    if cached is not None:
        return cached[0], cached[1], True

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="bf-media") as pool:
        video_fut = pool.submit(
            yt.download_video, record.url, vdir, max_height=settings.download_resolution
        )
        captions_fut = pool.submit(yt.fetch_captions, record.url, vdir)

        video_path = video_fut.result()
        frames_fut = pool.submit(_frames_pipeline, video_path, vdir, settings)

        captions = captions_fut.result()
        if captions is not None:
            transcript_path = yt.vtt_to_text(captions, vdir)
            source = "captions"
        else:
            logger.info("no captions for %s; falling back to whisper", record.video_id)
            audio_path = yt.extract_audio(video_path, vdir)
            transcript_path = yt.whisper_transcribe(
                audio_path, vdir, model_size=settings.whisper_model
            )
            source = "whisper"

        deduped = frames_fut.result()

    bundle = MediaBundle(
        video_id=record.video_id,
        video_path=str(video_path),
        transcript_path=str(transcript_path),
        transcript_source=source,  # type: ignore[arg-type]
        frames=deduped,
    )
    transcript_text = yt.load_transcript_text(
        Path(transcript_path), settings.transcript_max_chars
    )
    if not transcript_text.strip():
        raise RuntimeError(f"empty transcript for {record.video_id}")
    # Write media.json last: its presence is the cache marker for reruns.
    Workspace.write_text(vdir / "media.json", bundle.model_dump_json(indent=2))
    return bundle, transcript_text, False


def acquire_media_bundle(
    record: VideoRecord,
    ws: Workspace,
    settings: Settings,
) -> tuple[MediaBundle, str]:
    """Backward-compatible wrapper: (bundle, transcript_text)."""
    bundle, transcript_text, _cached = acquire_media_bundle_ex(record, ws, settings)
    return bundle, transcript_text
