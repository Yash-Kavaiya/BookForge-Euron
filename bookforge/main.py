"""BookForge CLI runner.

    python -m bookforge.main "https://www.youtube.com/@SomeChannel"
    python -m bookforge.main <url> --max-videos 2 --session my-run

Streams agent progress to stdout and exits non-zero when the book fails.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from google.adk.runners import Runner
from google.genai import types

from bookforge.agents.orchestrator import build_root_agent
from bookforge.config import Settings, get_settings
from bookforge.integrations.mlflow_integration import (
    ADK_WEB_USER_ID,
    create_session_service,
    flush_langfuse_tracing,
    flush_mlflow_tracing,
    setup_langfuse_tracing,
    setup_mlflow_tracing,
)

logger = logging.getLogger("bookforge")

# Automatically enable MLflow OpenTelemetry tracing targeting 'my-experiment'
setup_mlflow_tracing(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "my-experiment"),
)
# Langfuse is opt-in (BOOKFORGE_LANGFUSE_ENABLED / keys); no-op otherwise.
setup_langfuse_tracing()


def preflight_check(settings: Settings) -> list[str]:
    """Validate environment before running the pipeline. Returns warnings."""
    import shutil

    warnings = []

    # Credentials - require EURI_API_KEY for the Euron EURI API
    if not os.environ.get("EURI_API_KEY") and not os.environ.get("BOOKFORGE_EURI_API_KEY"):
        raise RuntimeError(
            "EURI_API_KEY is not set. Set it via the EURI_API_KEY environment variable "
            "(sign up at https://euron.one/euri)."
        )

    # ffmpeg
    if not shutil.which("ffmpeg"):
        warnings.append(
            "ffmpeg not found on PATH — frame extraction will fail. "
            "Install ffmpeg: https://ffmpeg.org/download.html"
        )

    # pdflatex
    if settings.compile_latex and not shutil.which("pdflatex"):
        warnings.append(
            "pdflatex not found on PATH but COMPILE_LATEX=true — "
            "PDF compilation will be skipped. Install TeX Live or MiKTeX, "
            "or set BOOKFORGE_COMPILE_LATEX=false."
        )
        settings.compile_latex = False

    # Workspace
    logger.info("workspace: %s", settings.workspace_root_abs)

    return warnings


# (flag, Settings field, help) for plain string options.
STUDIO_FLAGS: list[tuple[str, str, str]] = [
    ("--title", "book_title", "book title"),
    ("--subtitle", "book_subtitle", "book subtitle"),
    ("--author", "book_author", "author line on the title page and cover"),
    ("--header-left", "header_left", "left header text (default: chapter title)"),
    ("--header-right", "header_right", "right header text"),
    ("--footer", "footer_text", "left footer text (default: channel name)"),
    ("--logo", "logo", "logo file (PNG/JPG/PDF/SVG) for --logo-positions"),
    ("--logo-positions", "logo_positions", "comma list of title,header,footer"),
    ("--logo-title", "logo_title", "logo for the title page only"),
    ("--logo-header", "logo_header", "logo for the running header only"),
    ("--logo-footer", "logo_footer", "logo for the footer only"),
    ("--cover-image", "cover_image", "use this image as the cover instead of generating one"),
    ("--cover-prompt", "cover_prompt", "prompt for the generated cover image"),
]


def apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """CLI flags win over env vars. Mutates and returns ``settings``."""
    for _flag, dest, _help in STUDIO_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            setattr(settings, dest, value)
    for name in ("visuals", "diagram_engine", "channel_sort"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(settings, name, value)
    if getattr(args, "no_group_by_playlists", False):
        settings.channel_group_by_playlists = False
    if getattr(args, "no_cover", False):
        settings.cover_generate = False
    if getattr(args, "no_illustrations", False):
        settings.illustrations_enabled = False
    if getattr(args, "workspace", None):
        from pathlib import Path

        settings.workspace_root = Path(args.workspace)
    return settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bookforge",
        description="Turn a YouTube channel into a compiled LaTeX book.",
    )
    parser.add_argument(
        "channel_url",
        help="YouTube source: a video, a playlist, or a channel (@handle or /channel/ID)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="cap the number of videos (testing)",
    )
    parser.add_argument("--session", default=None, help="session id (default: random)")
    studio = parser.add_argument_group("BookForge Studio (each flag overrides its BOOKFORGE_* env var)")
    for flag, dest, help_text in STUDIO_FLAGS:
        studio.add_argument(flag, dest=dest, default=None, help=help_text)
    studio.add_argument("--visuals", choices=["light", "standard", "rich"], default=None,
                        help="visual density per chapter (BOOKFORGE_VISUALS)")
    studio.add_argument("--diagram-engine", choices=["tikz", "graphviz", "auto"], default=None,
                        help="diagram renderer (BOOKFORGE_DIAGRAM_ENGINE)")
    studio.add_argument("--channel-sort", choices=["newest", "popular"], default=None,
                        help="which channel videos to take (BOOKFORGE_CHANNEL_SORT)")
    studio.add_argument("--no-group-by-playlists", action="store_true",
                        help="channel books: do not group chapters into parts by playlist")
    studio.add_argument("--no-cover", action="store_true",
                        help="do not generate a cover image (BOOKFORGE_COVER_GENERATE=false)")
    studio.add_argument("--no-illustrations", action="store_true",
                        help="skip AI illustrations (BOOKFORGE_ILLUSTRATIONS_ENABLED=false)")
    studio.add_argument("--workspace", default=None,
                        help="output root directory (BOOKFORGE_WORKSPACE_ROOT)")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    # Load .env into os.environ so EURI_API_KEY is visible to
    # both the preflight check and the Euron EURI SDK.
    from dotenv import load_dotenv

    load_dotenv()

    settings = get_settings()
    if args.max_videos is not None:
        settings.max_videos = args.max_videos  # CLI override wins
    apply_cli_overrides(settings, args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    warnings = preflight_check(settings)
    for w in warnings:
        logger.warning(w)

    # Same SQLite store + user id as `adk web`, so the session appears in the UI.
    session_service = create_session_service()
    runner = Runner(
        agent=build_root_agent(settings),
        app_name="bookforge",
        session_service=session_service,
    )

    session_id = args.session or f"run-{uuid.uuid4().hex[:8]}"
    session = await session_service.create_session(
        app_name="bookforge", user_id=ADK_WEB_USER_ID, session_id=session_id
    )
    logger.info("session %s | workspace: %s", session_id, settings.workspace_root_abs)

    user_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                text=f"Create a complete book from this YouTube source: {args.channel_url}"
            )
        ],
    )

    exit_code = 0
    try:
        async for event in runner.run_async(
            user_id=ADK_WEB_USER_ID, session_id=session.id, new_message=user_msg
        ):
            if event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts).strip()
                if text:
                    print(f"[{event.author}] {text[:500]}", flush=True)
    except Exception as exc:
        logger.exception("pipeline aborted")
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        flush_mlflow_tracing()
        flush_langfuse_tracing()

    final = await session_service.get_session(
        app_name="bookforge", user_id=ADK_WEB_USER_ID, session_id=session_id
    )
    final_state = dict(final.state or {}) if final else {}
    book_pdf = final_state.get("book_pdf")
    slug = final_state.get("channel_slug", "")
    run_metrics = final_state.get("run_metrics") or {}
    if run_metrics:
        print(
            "\nRUN METRICS: "
            f"{run_metrics.get('chapters_done', 0)}/{run_metrics.get('chapters_total', 0)} chapters "
            f"({run_metrics.get('chapters_verified', 0)} verified, "
            f"{run_metrics.get('chapters_failed', 0)} failed) in "
            f"{run_metrics.get('wall_sec', 0)}s | LLM calls {run_metrics.get('llm_calls', 0)}, "
            f"tokens {run_metrics.get('prompt_tokens', 0)}+{run_metrics.get('completion_tokens', 0)}, "
            f"QA iterations {run_metrics.get('qa_iterations', 0)}, "
            f"critic calls {run_metrics.get('qa_llm_critic_calls', 0)}"
        )
        if slug:
            print(f"Detailed metrics: {settings.workspace_root_abs / slug / 'run_metrics.json'}")
    if book_pdf:
        print(f"\nBOOK READY: {settings.workspace_root_abs / slug / book_pdf}")
    elif exit_code == 0:
        print(
            "\nRun finished without a book PDF "
            "(compile disabled or all chapters failed).",
            file=sys.stderr,
        )
        exit_code = 2
    return exit_code


def cli_entry() -> None:
    main()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
