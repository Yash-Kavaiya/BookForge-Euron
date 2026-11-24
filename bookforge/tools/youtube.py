"""YouTube acquisition: channel listing, downloads, captions, whisper fallback.

All functions are synchronous, deterministic, and idempotent: if the target
file already exists, the download is skipped (cheap resume).
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL

logger = logging.getLogger(__name__)

_UNAVAILABLE_TITLES = {"[private video]", "[deleted video]", "[unavailable video]"}


# ---------------------------------------------------------------------------
# Channel listing
# ---------------------------------------------------------------------------


SourceKind = Literal["video", "playlist", "channel"]

_CHANNEL_PATH_RE = re.compile(r"^/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)", re.IGNORECASE)
_CHANNEL_TABS = ("videos", "featured", "streams", "playlists", "shorts", "about", "community", "live")
MORE_FROM_CHANNEL = "More from the channel"


def _clean_url(url: str) -> str:
    return re.sub(r"[.,;/\s]+$", "", (url or "").strip())


def classify_source_url(url: str) -> SourceKind:
    """Tell a single video, a playlist and a whole channel apart (pure)."""
    parsed = urlparse(_clean_url(url))
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    query = parse_qs(parsed.query)
    if "youtu.be" in host:
        return "playlist" if "list" in query else "video"
    if "list" in query and (path.rstrip("/") == "/playlist" or "v" not in query):
        return "playlist"
    if "list" in query:  # watch?v=..&list=.. -> the playlist
        return "playlist"
    if path.startswith("/watch") and "v" in query:
        return "video"
    if re.match(r"^/(?:shorts|live|embed)/[^/]+", path):
        return "video"
    if _CHANNEL_PATH_RE.match(path):
        return "channel"
    return "playlist"


def _channel_base(url: str) -> str:
    parsed = urlparse(_clean_url(url))
    m = _CHANNEL_PATH_RE.match(parsed.path or "")
    base_path = m.group(0) if m else (parsed.path or "").rstrip("/")
    scheme = parsed.scheme or "https"
    host = parsed.netloc or "www.youtube.com"
    return f"{scheme}://{host}{base_path}"


def channel_videos_url(url: str) -> str:
    """Channel URL (any tab, any query) -> its /videos tab."""
    return _channel_base(url) + "/videos"


def channel_playlists_url(url: str) -> str:
    """Channel URL (any tab, any query) -> its /playlists tab."""
    return _channel_base(url) + "/playlists"


def group_videos_by_part(
    videos: list[dict], membership: dict[str, str], fallback: str = MORE_FROM_CHANNEL
) -> list[dict]:
    """Set ``part_title`` from ``membership`` (video_id -> playlist title) and
    stable-group the videos by part, parts ordered by their first video.
    Pure: returns new dicts, the input is not mutated."""
    tagged = [{**v, "part_title": membership.get(v.get("video_id", ""), fallback)} for v in videos]
    order: list[str] = []
    for v in tagged:
        if v["part_title"] not in order and v["part_title"] != fallback:
            order.append(v["part_title"])
    order.append(fallback)  # loose videos always come last
    return [v for part in order for v in tagged if v["part_title"] == part]


_BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "extractor_args": {"youtube": {"player_client": ["android", "web", "ios"]}},
}


def _extract(url: str, **opts) -> dict | None:
    options = {**_BASE_OPTS, **{k: v for k, v in opts.items() if v is not None}}
    with YoutubeDL(options) as ydl:
        try:
            return ydl.extract_info(url, download=False)
        except Exception as exc:  # noqa: BLE001 - caller decides what empty means
            logger.warning("yt-dlp could not list %s: %s", url, type(exc).__name__)
            return None


def _entry_to_video(entry: dict) -> dict | None:
    video_id = entry.get("id") or ""
    title = entry.get("title")
    # Private / deleted / unavailable playlist items come back without a
    # title (or with a placeholder) and can never be downloaded.
    if not title or title.strip().lower() in _UNAVAILABLE_TITLES:
        logger.info("skipping unavailable playlist entry %s", video_id or "?")
        return None
    if not video_id:
        return None
    url = entry.get("url") or entry.get("webpage_url") or ""
    if "youtube.com" not in url and "youtu.be" not in url:
        url = f"https://www.youtube.com/watch?v={video_id}"
    return {
        "video_id": video_id,
        "title": title,
        "url": url,
        "duration_sec": int(entry.get("duration") or 0),
        "upload_date": entry.get("upload_date") or "",
        "part_title": "",
        "view_count": int(entry.get("view_count") or 0),
    }


def _filter_entries(entries, min_duration_sec: int, max_duration_sec: int) -> list[dict]:
    videos: list[dict] = []
    seen: set[str] = set()
    for entry in entries or []:
        if not entry:
            continue
        duration = int(entry.get("duration") or 0)
        if duration and not (min_duration_sec <= duration <= max_duration_sec):
            continue
        video = _entry_to_video(entry)
        # Playlists can list the same video twice — one chapter per video.
        if video is None or video["video_id"] in seen:
            continue
        seen.add(video["video_id"])
        videos.append(video)
    return videos


def _playlist_membership(channel_url: str, max_playlists: int) -> dict[str, str]:
    """video_id -> title of the first channel playlist containing it."""
    info = _extract(channel_playlists_url(channel_url), extract_flat="in_playlist", ignoreerrors=True)
    membership: dict[str, str] = {}
    lists = [e for e in (info or {}).get("entries") or [] if e and e.get("id") and e.get("title")]
    for pl in lists[: max(0, max_playlists)]:
        sub = _extract(
            f"https://www.youtube.com/playlist?list={pl['id']}",
            extract_flat="in_playlist",
            ignoreerrors=True,
        )
        for e in (sub or {}).get("entries") or []:
            if e and e.get("id") and e["id"] not in membership:
                membership[e["id"]] = pl["title"]
    return membership


def list_channel_videos(
    channel_url: str,
    max_videos: int | None = None,
    min_duration_sec: int = 0,
    max_duration_sec: int = 10**9,
    sort: str = "newest",
    group_by_playlists: bool = False,
    max_playlists: int = 25,
) -> tuple[str, list[dict]]:
    """Return (book source title, video dicts) for a video, playlist or channel URL.

    * video: one dict (duration filters do not apply to an explicit request).
    * playlist: playlist order, unavailable/duplicate/out-of-range filtered.
    * channel: ``sort="newest"`` takes the newest N and orders them oldest ->
      newest; ``sort="popular"`` takes the N most viewed, most viewed first.
      ``group_by_playlists`` sets ``part_title`` from the channel's playlists
      and groups chapters by part, loose videos last (never fails intake).
    Uses flat extraction: no downloads, one metadata pass per list.
    """
    cleaned_url = _clean_url(channel_url)
    kind = classify_source_url(cleaned_url)

    if kind == "video":
        info = _extract(cleaned_url, noplaylist=True, skip_download=True)
        if not info:
            return "Unknown video", []
        video = _entry_to_video({**info, "url": info.get("webpage_url") or ""})
        return (info.get("title") or "Unknown video"), ([video] if video else [])

    if kind == "channel":
        over = max_videos * 3 if (max_videos and sort != "popular") else None
        info = _extract(
            channel_videos_url(cleaned_url),
            extract_flat="in_playlist",
            ignoreerrors=True,
            playlistend=over,
        )
        raw_title = (info or {}).get("channel") or (info or {}).get("uploader") or (info or {}).get("title") or ""
        title = re.sub(r"\s*-\s*Videos$", "", raw_title).strip() or "Unknown channel"
        videos = _filter_entries((info or {}).get("entries"), min_duration_sec, max_duration_sec)
        if sort == "popular":
            videos = sorted(videos, key=lambda v: -v["view_count"])
            if max_videos:
                videos = videos[:max_videos]
        else:  # newest first on YouTube -> book reads oldest to newest
            if max_videos:
                videos = videos[:max_videos]
            videos = list(reversed(videos))
        if group_by_playlists and videos:
            try:
                membership = _playlist_membership(cleaned_url, max_playlists)
                if membership:
                    videos = group_videos_by_part(videos, membership)
            except Exception as exc:  # noqa: BLE001 - grouping is a nicety
                logger.warning("playlist grouping skipped for %s: %s", cleaned_url, exc)
                videos = [{**v, "part_title": ""} for v in videos]
        return title, videos

    # playlist (and anything else yt-dlp can list)
    normalized_url = cleaned_url
    if "/@" in normalized_url and not normalized_url.endswith("/videos"):
        normalized_url += "/videos"
    over = max_videos * 3 if max_videos else None  # over-fetch, filter below
    info = _extract(normalized_url, extract_flat="in_playlist", ignoreerrors=True, playlistend=over)
    if info is None and normalized_url != cleaned_url:
        info = _extract(cleaned_url, extract_flat="in_playlist", ignoreerrors=True, playlistend=over)
    title = (info or {}).get("title") or "Unknown channel"
    videos = _filter_entries((info or {}).get("entries"), min_duration_sec, max_duration_sec)
    if max_videos:
        videos = videos[:max_videos]
    return title, videos


# ---------------------------------------------------------------------------
# Download + audio
# ---------------------------------------------------------------------------


_PARTIAL_SUFFIXES = (".part", ".ytdl", ".temp", ".tmp")


def _finished_videos(dest_dir: Path) -> list[Path]:
    """Completed video files only: an interrupted yt-dlp run leaves
    ``video.mp4.part`` / ``.ytdl`` files that must not count as a cache hit."""
    return [
        p
        for p in sorted(dest_dir.glob("video.*"))
        if not p.name.endswith(_PARTIAL_SUFFIXES) and not re.match(r"video\.f\d+", p.name)
    ]


def download_video(url: str, dest_dir: Path, max_height: int = 480) -> Path:
    """Download a capped-resolution mp4; returns existing file on re-run."""
    dest_dir = Path(dest_dir)
    existing = _finished_videos(dest_dir)
    if existing:
        return existing[0]

    outtmpl = str(dest_dir / "video.%(ext)s")
    options = {
        "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "overwrites": False,
        # YouTube throttles long downloads (503s, short reads, timeouts):
        # retry and resume the .part file instead of failing the chapter.
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "socket_timeout": 30,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "ios"],
            }
        },
    }
    with YoutubeDL(options) as ydl:
        ydl.download([url])
    downloaded = _finished_videos(dest_dir)
    if not downloaded:
        raise RuntimeError(f"yt-dlp produced no file for {url}")
    return downloaded[0]


def extract_audio(video_path: Path, dest_dir: Path) -> Path:
    """16 kHz mono wav for whisper; idempotent."""
    dest = Path(dest_dir) / "audio.wav"
    if dest.exists():
        return dest
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dest),
    ]
    _run(cmd, f"audio extraction for {video_path}")
    return dest


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def _caption_files(dest_dir: Path) -> list[Path]:
    return sorted(dest_dir.glob("captions.*.vtt")) + sorted(dest_dir.glob("captions.vtt"))


def _download_captions(url: str, dest_dir: Path, langs: list[str]) -> Path | None:
    options = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "vtt",
        "outtmpl": str(dest_dir / "captions.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many exception types
        logger.warning("caption fetch (%s) failed for %s: %s", ",".join(langs), url, exc)
        return None
    found = _caption_files(dest_dir)
    return found[0] if found else None


def original_caption_langs(info: dict) -> list[str]:
    """Caption tracks in the video's spoken language, best first.

    YouTube's machine-translated English track ("en" derived from e.g. Hindi)
    is heavily rate-limited (HTTP 429), while the original-language track
    ("hi-orig" / "hi") downloads fine, and the LLM analyst can read it.
    """
    auto = info.get("automatic_captions") or {}
    manual = info.get("subtitles") or {}
    lang = (info.get("language") or "").strip()
    picks: list[str] = []
    if lang:
        base = lang.split("-")[0]
        for c in (f"{lang}-orig", lang, f"{base}-orig", base):
            if (c in auto or c in manual) and c not in picks:
                picks.append(c)
        if picks:
            # A known spoken language wins: other "<xx>-orig" tracks belong to
            # YouTube's auto-dubbed audio tracks (e.g. an Arabic dub of an
            # English talk), not to what the speaker actually said.
            return picks[:2]
        if base == "en":
            return []  # English talk whose English track is blocked -> Whisper
    picks += [k for k in auto if k.endswith("-orig") and k not in picks]
    return picks[:2]


def fetch_captions(
    url: str, dest_dir: Path, langs: tuple[str, ...] = ("en.*", "en")
) -> Path | None:
    """Fetch human or auto captions as vtt. Returns None when unavailable.

    English is preferred; when it is missing or blocked, fall back to the
    caption track in the video's original language.
    """
    dest_dir = Path(dest_dir)
    existing = _caption_files(dest_dir)
    if existing:
        return existing[0]

    found = _download_captions(url, dest_dir, list(langs))
    if found:
        return found
    try:
        with YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("caption track listing failed for %s: %s", url, exc)
        return None
    fallback = original_caption_langs(info)
    if not fallback:
        return None
    logger.info("English captions unavailable for %s; using %s", url, fallback[0])
    return _download_captions(url, dest_dir, fallback[:1])


def _load_wav_16k(audio_path: Path):
    """Our extracted audio (16 kHz mono s16 WAV) as float32 samples.

    Passing an array skips faster-whisper's PyAV decoder, whose
    ``av.open(..., metadata_errors=...)`` call breaks on PyAV >= 15. Falls back
    to the path for any other format.
    """
    import wave

    import numpy as np

    try:
        with wave.open(str(audio_path), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                return str(audio_path)
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, OSError, EOFError):
        return str(audio_path)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def whisper_transcribe(
    audio_path: Path, dest_dir: Path, model_size: str = "base"
) -> Path:
    """Local faster-whisper transcription with timestamps; idempotent."""
    dest = Path(dest_dir) / "transcript.txt"
    if dest.exists():
        return dest

    from faster_whisper import WhisperModel  # heavy import, deferred

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(_load_wav_16k(audio_path), vad_filter=True)
    lines = [f"[{_fmt_ts(seg.start)}] {seg.text.strip()}" for seg in segments]
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


_VTT_TS = re.compile(r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})[.,]\d+\s*-->")
_VTT_TAG = re.compile(r"<[^>]+>")


def vtt_to_text(vtt_path: Path, dest_dir: Path) -> Path:
    """Convert WebVTT captions to timestamped plain text; idempotent.

    Output lines look like: [00:01:12] the spoken words ...
    Consecutive duplicate lines (auto-caption scrolling artifacts) are dropped.
    """
    dest = Path(dest_dir) / "transcript.txt"
    if dest.exists():
        return dest

    lines_out: list[str] = []
    last_text = ""
    ts = "00:00:00"
    for raw in Path(vtt_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        match = _VTT_TS.search(line)
        if match:
            ts = f"{match.group('h')}:{match.group('m')}:{match.group('s')}"
            continue  # timestamp line; text follows on the next line(s)
        if line.isdigit():
            continue  # cue index
        text = _VTT_TAG.sub("", line).strip()
        if not text or text == last_text:
            continue
        lines_out.append(f"[{ts}] {text}")
        last_text = text
    dest.write_text("\n".join(lines_out), encoding="utf-8")
    return dest


def load_transcript_text(path: Path, max_chars: int = 120_000, windows: int = 6) -> str:
    """Read a transcript, keeping it under ``max_chars``.

    Long videos are not cut to head+tail (which loses the whole middle of a
    90-minute talk); instead ``windows`` evenly spaced excerpts are kept, each
    still carrying its timestamps, so the analyst sees the full arc.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    marker = "\n...[transcript excerpt gap]...\n"
    windows = max(1, windows)
    size = max(1, (max_chars - len(marker) * (windows - 1)) // windows)
    stride = (len(text) - size) / max(1, windows - 1) if windows > 1 else 0
    parts = []
    for i in range(windows):
        start = int(i * stride)
        chunk = text[start : start + size]
        # snap to whole lines so timestamps stay intact
        if start > 0 and "\n" in chunk:
            chunk = chunk[chunk.index("\n") + 1 :]
        if "\n" in chunk and start + size < len(text):
            chunk = chunk[: chunk.rindex("\n")]
        parts.append(chunk)
    return marker.join(parts)


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed: {proc.stderr[-2000:]}")
