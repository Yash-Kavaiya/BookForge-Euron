"""Offline tests for the web pipeline: EURI is replaced by an httpx MockTransport."""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import pytest

from bookforge.web.euri_client import EuriClient, EuriError, MissingApiKeyError, extract_json
from bookforge.web.latex_export import escape_latex, markdown_book_to_latex, markdown_to_latex
from bookforge.web.pipeline import (
    BookRequest,
    clean_chapter_markdown,
    generate_book,
    verify_chapter,
    ChapterPlan,
)
from bookforge.web.settings import WebSettings

CHAPTER_MD = (
    "An opening paragraph about the subject with enough words to count.\n\n"
    "## First idea\n\nSome prose here. " + "word " * 300 + "\n\n"
    "- point one\n- point two\n\n## Key Takeaways\n\n- takeaway one\n- takeaway two\n"
)


def fake_euri(record: dict | None = None, *, critic_verdict: str = "pass") -> httpx.MockTransport:
    """A fake EURI server that answers planner, writer, critic and image calls."""
    record = record if record is not None else {}
    record.setdefault("calls", [])
    record.setdefault("in_flight", 0)
    record.setdefault("max_in_flight", 0)

    async def handler(request: httpx.Request) -> httpx.Response:
        record["calls"].append(request.url.path)
        assert request.headers["authorization"] == "Bearer test-key"
        record["in_flight"] += 1
        record["max_in_flight"] = max(record["max_in_flight"], record["in_flight"])
        try:
            await asyncio.sleep(0.01)
            if request.url.path.endswith("/images/generations"):
                return httpx.Response(200, json={"data": [{"url": "https://img.example/cover.png"}]})
            body = json.loads(request.content)
            prompt = body["messages"][-1]["content"]
            if prompt.startswith("Plan a short book"):
                n = int(re.search(r"Exactly (\d+) chapters", prompt).group(1))
                content = json.dumps({
                    "title": "Tea & Time",
                    "subtitle": "A short history",
                    "description": "Two sentences. About tea.",
                    "chapters": [
                        {"title": f"Part {i}", "summary": "s", "key_points": ["a", "b", "c"]}
                        for i in range(1, n + 1)
                    ],
                })
            elif prompt.startswith("Review chapter"):
                content = json.dumps({"verdict": critic_verdict, "defects": ["Add an example."]})
            else:
                content = "# Chapter title repeated\n\n" + CHAPTER_MD
            return httpx.Response(200, json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            })
        finally:
            record["in_flight"] -= 1

    return httpx.MockTransport(handler)


def settings(**kw) -> WebSettings:
    return WebSettings(api_key="test-key", **kw)


def test_settings_default_to_euri(monkeypatch):
    monkeypatch.delenv("EURI_API_KEY", raising=False)
    s = WebSettings.from_env()
    assert s.api_base == "https://api.euron.one/api/v1/euri"
    assert s.writer_model.startswith("euri-")
    assert s.max_concurrency == 2
    assert not s.configured
    monkeypatch.setenv("EURI_API_KEY", "abc")
    monkeypatch.setenv("BOOKFORGE_WEB_WRITER_MODEL", "euri-chat-smart")
    s = WebSettings.from_env()
    assert s.configured and s.writer_model == "euri-chat-smart"
    assert "abc" not in repr(s)  # the key never shows up in reprs/logs


def test_client_requires_key():
    with pytest.raises(MissingApiKeyError):
        EuriClient(WebSettings(api_key=""))


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


async def test_quick_book_end_to_end_offline():
    record: dict = {}
    events: list[dict] = []

    async def emit(e):
        events.append(e)

    client = EuriClient(settings(), transport=fake_euri(record))
    async with client:
        result = await generate_book(BookRequest(topic="history of tea"), settings(), emit, client=client)

    assert result.plan.title == "Tea & Time"
    assert len(result.chapters) == 4  # quick mode
    assert all(c.status == "verified" for c in result.chapters)
    assert "Chapter title repeated" not in result.markdown  # writer's repeated H1 stripped
    assert result.markdown.count("# Chapter ") == 4
    assert r"\chapter{Part 1}" in result.latex and r"Tea \& Time" in result.latex
    assert record["max_in_flight"] <= 2  # EURI concurrency cap respected
    assert result.usage["calls"] == 5  # 1 plan + 4 chapters, no critic in quick mode
    kinds = [e["type"] for e in events]
    assert kinds[0] == "stage" and "outline" in kinds and kinds[-1] == "stage"


async def test_standard_mode_runs_critic_and_refiner():
    record: dict = {}
    client = EuriClient(settings(), transport=fake_euri(record, critic_verdict="revise"))
    async with client:
        result = await generate_book(
            BookRequest(topic="tea", mode="standard", chapters=2, cover=True), settings(), client=client
        )
    # plan + 2 x (write + critic + refine) + cover image
    assert result.usage["calls"] == 1 + 2 * 3
    assert record["calls"].count("/api/v1/euri/images/generations") == 1
    assert result.cover_url == "https://img.example/cover.png"
    assert all(c.status in {"verified", "revised"} for c in result.chapters)
    assert record["max_in_flight"] <= 2


async def test_retries_on_429_then_succeeds():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = EuriClient(settings(), transport=httpx.MockTransport(handler))
    import bookforge.web.euri_client as mod

    async def no_sleep(_):
        return None

    mod.asyncio.sleep, real = no_sleep, mod.asyncio.sleep
    try:
        async with client:
            assert await client.chat([{"role": "user", "content": "hi"}], model="euri-chat-fast") == "ok"
    finally:
        mod.asyncio.sleep = real
    assert attempts["n"] == 2


async def test_bad_key_is_not_retried():
    client = EuriClient(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    async with client:
        with pytest.raises(EuriError, match="rejected the API key"):
            await client.chat([{"role": "user", "content": "hi"}], model="euri-chat-fast")


def test_clean_and_verify_chapter():
    raw = "```markdown\n# Chapter 1: X\n\n" + CHAPTER_MD + "```"
    cleaned = clean_chapter_markdown(raw, "X")
    assert cleaned.startswith("An opening paragraph")
    assert verify_chapter(cleaned, ChapterPlan(title="X"), 300) == []
    issues = verify_chapter("short text", ChapterPlan(title="X"), 300)
    assert any("too short" in i for i in issues)
    assert any("Key Takeaways" in i for i in issues)


def test_latex_export_escapes_and_structures():
    assert escape_latex("50% & $5 #1 a_b") == r"50\% \& \$5 \#1 a\_b"
    tex = markdown_to_latex("## Head\n\n**bold** and *it* `x_y`\n\n- a\n- b\n\n1. one\n\n```\nraw_code\n```")
    assert r"\section{Head}" in tex
    assert r"\textbf{bold}" in tex and r"\emph{it}" in tex and r"\texttt{x\_y}" in tex
    assert tex.count(r"\begin{itemize}") == 1 and r"\begin{enumerate}" in tex
    assert "raw_code" in tex and r"\begin{verbatim}" in tex
    book = markdown_book_to_latex("T", "S", [("One", "text")])
    assert book.startswith(r"\documentclass") and book.rstrip().endswith(r"\end{document}")
    assert r"\tableofcontents" in book


async def test_time_budget_always_ships_a_book():
    """Chapters still running when the budget ends are cut, the rest still compile."""

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("Plan a short book"):
            content = json.dumps({"title": "T", "chapters": [
                {"title": f"C{i}", "key_points": []} for i in range(1, 5)]})
        else:
            if "chapter 3" in prompt or "chapter 4" in prompt:
                await asyncio.sleep(30)  # far beyond the budget
            content = CHAPTER_MD
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    s = settings(time_budget_s=10)
    client = EuriClient(s, transport=httpx.MockTransport(handler))
    import time as _time

    start = _time.monotonic()
    async with client:
        result = await generate_book(BookRequest(topic="tea"), s, client=client)
    assert _time.monotonic() - start < 9
    statuses = [c.status for c in result.chapters]
    assert statuses[:2] == ["verified", "verified"] and statuses[2:] == ["failed", "failed"]
    assert "ran out of time" in result.markdown
