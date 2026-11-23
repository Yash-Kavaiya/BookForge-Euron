"""FastAPI app: web UI + streaming book generation on the Euron EURI API.

Routes
  GET  /              web UI
  GET  /api/health    liveness + whether EURI_API_KEY is configured (never its value)
  GET  /api/modes     generation modes and their sizes
  POST /api/generate  NDJSON stream of progress events, ending in a "done" event
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from bookforge import __version__
from bookforge.web.euri_client import EuriError, MissingApiKeyError
from bookforge.web.pipeline import MODES, BookRequest, BookResult, generate_book
from bookforge.web.settings import WebSettings

logger = logging.getLogger("bookforge.web")
logging.basicConfig(level=os.environ.get("BOOKFORGE_LOG_LEVEL", "INFO"))

STATIC_DIR = Path(__file__).parent / "static"
HEARTBEAT_S = 10
# Generations that may run at once in one function instance (each uses up to
# 2 concurrent EURI requests through the shared client semaphore).
_ACTIVE = asyncio.Semaphore(int(os.environ.get("BOOKFORGE_WEB_MAX_JOBS", "3")))

app = FastAPI(
    title="BookForge on Euron",
    version=__version__,
    description="Multi-agent book generator powered by the Euron EURI API.",
)


def _settings() -> WebSettings:
    return WebSettings.from_env()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> HTMLResponse:
    svg = (STATIC_DIR / "favicon.svg").read_text(encoding="utf-8")
    return HTMLResponse(svg, media_type="image/svg+xml")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    s = _settings()
    return {
        "status": "ok",
        "version": __version__,
        "euri_configured": s.configured,
        "euri_base_url": s.api_base,
        "models": {
            "planner": s.planner_model,
            "writer": s.writer_model,
            "deep_writer": s.deep_writer_model,
            "critic": s.critic_model,
            "image": s.image_model,
        },
        "access_code_required": bool(os.environ.get("BOOKFORGE_ACCESS_CODE")),
    }


@app.get("/api/modes")
async def modes() -> dict[str, Any]:
    return {name: vars(profile) for name, profile in MODES.items()}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "book"


def _bundle(result: BookResult, job_id: str) -> dict[str, Any]:
    """Write outputs to /tmp (the only writable path on Vercel) and build a zip."""
    slug = _slug(result.plan.title)
    meta = {
        "title": result.plan.title,
        "subtitle": result.plan.subtitle,
        "description": result.plan.description,
        "request": result.request.model_dump(),
        "chapters": [
            {"number": c.number, "title": c.plan.title, "status": c.status,
             "words": len(c.markdown.split())}
            for c in result.chapters
        ],
        "models": result.models,
        "usage": result.usage,
        "seconds": result.seconds,
        "cover_url": result.cover_url,
    }
    files = {
        f"{slug}/book.md": result.markdown,
        f"{slug}/book.tex": result.latex,
        f"{slug}/book.json": json.dumps(meta, indent=2),
        f"{slug}/README.txt": "Compile the PDF locally with:  pdflatex book.tex && pdflatex book.tex\n",
    }
    try:
        out_dir = Path(_settings().output_root) / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (out_dir / Path(name).name).write_text(content, encoding="utf-8")
    except OSError as exc:  # read-only FS etc. - downloads still work from the stream
        logger.warning("could not persist outputs: %s", exc)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return {
        "type": "done",
        "job_id": job_id,
        "slug": slug,
        "markdown": result.markdown,
        "latex": result.latex,
        "zip_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "meta": meta,
    }


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


@app.post("/api/generate")
async def generate(request: Request) -> Any:
    settings = _settings()
    if not settings.configured:
        return JSONResponse(
            status_code=503,
            content={"error": "EURI_API_KEY is not configured on the server. Add it as an "
                     "environment variable (get a key at https://euron.one/euri) and redeploy."},
        )
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Body must be JSON."})
    access_code = os.environ.get("BOOKFORGE_ACCESS_CODE")
    if access_code and (payload or {}).get("access_code") != access_code:
        return JSONResponse(status_code=401, content={"error": "A valid access code is required."})
    try:
        book_request = BookRequest.model_validate({k: v for k, v in payload.items() if k != "access_code"})
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first.get("loc", []))
        return JSONResponse(status_code=422, content={"error": f"{field}: {first.get('msg')}"})

    if _ACTIVE.locked():
        return JSONResponse(status_code=429, content={"error": "Busy generating other books. Try again in a minute."})

    job_id = uuid.uuid4().hex[:12]
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def emit(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def run() -> None:
        async with _ACTIVE:
            try:
                result = await generate_book(book_request, settings, emit)
                await queue.put(_bundle(result, job_id))
            except MissingApiKeyError as exc:
                await queue.put({"type": "error", "message": str(exc)})
            except EuriError as exc:
                await queue.put({"type": "error", "message": f"EURI error: {exc}"})
            except Exception:  # noqa: BLE001 - report, never leak internals
                logger.exception("generation %s failed", job_id)
                await queue.put({"type": "error", "message": "Generation failed unexpectedly. Please retry."})
            finally:
                await queue.put(None)

    async def stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run())
        yield _line({"type": "start", "job_id": job_id, "mode": book_request.mode})
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield _line({"type": "ping"})
                    continue
                if event is None:
                    break
                yield _line(event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
