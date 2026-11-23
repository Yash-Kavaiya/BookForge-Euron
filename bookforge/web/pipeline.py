"""Topic-to-book multi-agent pipeline on the Euron EURI API.

Same agent roles as the ADK YouTube pipeline (intake -> plan -> write ->
verify -> critique -> refine -> compile), sized to finish inside a single
serverless request. Every stage reports progress through ``emit`` so the web
UI can stream it.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, field_validator

from bookforge.web.euri_client import EuriClient, EuriError
from bookforge.web.latex_export import markdown_book_to_latex
from bookforge.web.settings import WebSettings

logger = logging.getLogger(__name__)

Mode = Literal["quick", "standard", "deep"]
Emit = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class ModeProfile:
    chapters: int
    words: int
    critic: bool
    deep_writer: bool
    max_tokens: int


MODES: dict[str, ModeProfile] = {
    # quick: fits comfortably inside a 300 s serverless function.
    "quick": ModeProfile(chapters=4, words=450, critic=False, deep_writer=False, max_tokens=1800),
    "standard": ModeProfile(chapters=6, words=700, critic=True, deep_writer=False, max_tokens=2600),
    # deep: the reasoning-heavy writer is slow, so fewer, longer chapters; the
    # time budget below still guarantees a book inside the function limit.
    "deep": ModeProfile(chapters=6, words=1000, critic=True, deep_writer=True, max_tokens=6000),
}


class BookRequest(BaseModel):
    """What the user asks for in the web form."""

    topic: str = Field(min_length=3, max_length=300)
    brief: str = Field(default="", max_length=2000)
    audience: str = Field(default="curious beginners", max_length=200)
    tone: str = Field(default="clear, friendly and practical", max_length=200)
    language: str = Field(default="English", max_length=40)
    mode: Mode = "quick"
    chapters: int | None = Field(default=None, ge=2, le=12)
    cover: bool = False

    @field_validator("topic", "brief", "audience", "tone", "language")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class ChapterPlan(BaseModel):
    title: str
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)


class BookPlan(BaseModel):
    title: str
    subtitle: str = ""
    description: str = ""
    chapters: list[ChapterPlan]


@dataclass
class Chapter:
    number: int
    plan: ChapterPlan
    markdown: str = ""
    status: str = "pending"  # pending | written | revised | verified
    issues: list[str] = field(default_factory=list)


@dataclass
class BookResult:
    request: BookRequest
    plan: BookPlan
    chapters: list[Chapter]
    markdown: str
    latex: str
    cover_url: str | None
    seconds: float
    usage: dict[str, int]
    models: dict[str, str]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_EDITOR = (
    "You are BookForge, a senior non-fiction editor and author. You write accurate, "
    "well-structured, engaging books. Never mention that you are an AI."
)


def _plan_messages(req: BookRequest, n: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_EDITOR},
        {
            "role": "user",
            "content": (
                f"Plan a short book.\nTopic: {req.topic}\n"
                f"Brief from the author: {req.brief or '(none)'}\n"
                f"Audience: {req.audience}\nTone: {req.tone}\nLanguage: {req.language}\n\n"
                f"Return ONLY JSON with this shape:\n"
                '{"title": str, "subtitle": str, "description": str (2 sentences), '
                '"chapters": [{"title": str, "summary": str (1-2 sentences), '
                '"key_points": [3-5 short strings]}]}\n'
                f"Exactly {n} chapters, ordered so each builds on the previous one. "
                "Chapter titles must not include the word 'Chapter' or numbers."
            ),
        },
    ]


def _writer_messages(
    req: BookRequest, plan: BookPlan, ch: Chapter, words: int
) -> list[dict[str, str]]:
    toc = "\n".join(f"{i}. {c.title}" for i, c in enumerate(plan.chapters, 1))
    points = "\n".join(f"- {p}" for p in ch.plan.key_points) or "- (use your judgement)"
    return [
        {"role": "system", "content": SYSTEM_EDITOR},
        {
            "role": "user",
            "content": (
                f'Write chapter {ch.number} of the book "{plan.title}".\n'
                f"Audience: {req.audience}. Tone: {req.tone}. Language: {req.language}.\n"
                f"Book outline:\n{toc}\n\n"
                f"This chapter: {ch.plan.title}\nSummary: {ch.plan.summary}\n"
                f"Must cover:\n{points}\n\n"
                f"Rules:\n- About {words} words of Markdown.\n"
                "- Do NOT repeat the chapter title as a heading; start with an opening paragraph.\n"
                "- Use 2-4 '## ' section headings, short paragraphs, and at least one "
                "bulleted or numbered list.\n"
                "- End with a '## Key Takeaways' section of 3-5 bullets.\n"
                "- Concrete examples; no filler; do not invent precise statistics or quotes."
            ),
        },
    ]


def _critic_messages(plan: BookPlan, ch: Chapter) -> list[dict[str, str]]:
    points = "; ".join(ch.plan.key_points)
    return [
        {"role": "system", "content": "You are a strict book editor. Reply with JSON only."},
        {
            "role": "user",
            "content": (
                f'Review chapter "{ch.plan.title}" of "{plan.title}". It must cover: {points}.\n'
                "Check accuracy, coverage of the key points, structure and clarity.\n"
                'Return JSON: {"verdict": "pass" | "revise", "defects": [short, specific, '
                "actionable fix strings, max 5]}\n\nCHAPTER:\n" + ch.markdown[:12000]
            ),
        },
    ]


def _refiner_messages(
    req: BookRequest, plan: BookPlan, ch: Chapter, words: int
) -> list[dict[str, str]]:
    fixes = "\n".join(f"{i}. {d}" for i, d in enumerate(ch.issues, 1))
    return [
        {"role": "system", "content": SYSTEM_EDITOR},
        {
            "role": "user",
            "content": (
                f'Revise this chapter ("{ch.plan.title}") of "{plan.title}" for {req.audience}. '
                f"Fix every numbered defect, keep the same Markdown structure, about {words} "
                "words, keep the '## Key Takeaways' section, and return ONLY the revised "
                f"chapter.\n\nDEFECTS:\n{fixes}\n\nCHAPTER:\n{ch.markdown}"
            ),
        },
    ]


# --------------------------------------------------------------------------- #
# Deterministic checks (the "verifier" agent)
# --------------------------------------------------------------------------- #

_META_PATTERNS = re.compile(
    r"(as an ai|language model|i cannot|i can't help|here is the (revised )?chapter)", re.I
)


def clean_chapter_markdown(text: str, title: str) -> str:
    """Strip fences, a repeated chapter title, and assistant chatter."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    lines = text.splitlines()
    while lines and (
        not lines[0].strip()
        or re.match(r"^#\s+", lines[0])
        or re.match(r"^\*\*?chapter\b", lines[0].strip(), re.I)
        or lines[0].strip().lower().startswith(("here is", "here's", "sure"))
    ):
        lines.pop(0)
    return "\n".join(lines).strip()


def verify_chapter(markdown: str, plan: ChapterPlan, target_words: int) -> list[str]:
    """Machine checks; returns a list of defects (empty means pass)."""
    issues: list[str] = []
    words = len(re.findall(r"\w+", markdown))
    if words < target_words * 0.5:
        issues.append(f"Chapter is too short ({words} words); expand it to about {target_words}.")
    if not re.search(r"^##\s+", markdown, re.M):
        issues.append("Add '## ' section headings to structure the chapter.")
    if "key takeaways" not in markdown.lower():
        issues.append("End with a '## Key Takeaways' section of 3-5 bullets.")
    if _META_PATTERNS.search(markdown):
        issues.append("Remove assistant/meta commentary; write only book prose.")
    return issues


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


async def _noop_emit(_event: dict[str, Any]) -> None:
    return None


def _fallback_plan(req: BookRequest, n: int) -> BookPlan:
    return BookPlan(
        title=req.topic.title(),
        subtitle="A BookForge guide",
        description=req.brief or f"A short book about {req.topic}.",
        chapters=[
            ChapterPlan(title=f"{req.topic.title()}: Part {i}", summary="", key_points=[])
            for i in range(1, n + 1)
        ],
    )


async def plan_book(client: EuriClient, req: BookRequest, n: int, model: str) -> BookPlan:
    data = await client.chat_json(_plan_messages(req, n), model=model, temperature=0.5, max_tokens=2500)
    if isinstance(data, list):
        data = {"title": req.topic, "chapters": data}
    plan = BookPlan.model_validate(data)
    plan.chapters = [c for c in plan.chapters if c.title.strip()][:n]
    if not plan.chapters:
        raise ValueError("planner returned no chapters")
    return plan


def assemble_markdown(plan: BookPlan, chapters: list[Chapter], cover_url: str | None) -> str:
    parts = [f"# {plan.title}"]
    if plan.subtitle:
        parts.append(f"_{plan.subtitle}_")
    if cover_url and not cover_url.startswith("data:"):
        parts.append(f"![Cover]({cover_url})")
    if plan.description:
        parts.append(plan.description)
    parts.append("## Contents\n\n" + "\n".join(
        f"{c.number}. {c.plan.title}" for c in chapters
    ))
    for c in chapters:
        parts.append(f"# Chapter {c.number}: {c.plan.title}\n\n{c.markdown}")
    parts.append("---\n_Generated with BookForge on the Euron EURI API._")
    return "\n\n".join(parts) + "\n"


async def generate_book(
    req: BookRequest,
    settings: WebSettings,
    emit: Emit = _noop_emit,
    *,
    client: EuriClient | None = None,
) -> BookResult:
    """Run intake -> plan -> write -> verify/critique/refine -> compile."""
    started = time.monotonic()
    profile = MODES[req.mode]
    n_chapters = req.chapters or profile.chapters
    deadline = started + settings.time_budget_s

    def remaining() -> float:
        return deadline - time.monotonic()

    writer_model = settings.deep_writer_model if profile.deep_writer else settings.writer_model
    models = {
        "planner": settings.planner_model,
        "writer": writer_model,
        "critic": settings.critic_model if profile.critic else "",
        "cover": settings.image_model if req.cover else "",
    }
    own_client = client is None
    client = client or EuriClient(settings)
    try:
        await emit({"type": "stage", "stage": "intake", "message": f"Planning a {n_chapters}-chapter book ({req.mode} mode)"})

        cover_task: asyncio.Task[str | None] | None = None
        if req.cover:
            async def _cover() -> str | None:
                try:
                    return await client.image(
                        f"Elegant minimal book cover illustration for a book about {req.topic}. "
                        "No text, no letters.",
                        model=settings.image_model,
                    )
                except EuriError as exc:
                    logger.warning("cover generation failed: %s", exc)
                    return None
            cover_task = asyncio.create_task(_cover())

        try:
            plan = await plan_book(client, req, n_chapters, settings.planner_model)
        except (ValueError, EuriError) as exc:
            if isinstance(exc, EuriError) and "API key" in str(exc):
                raise
            logger.warning("planner failed (%s); using fallback outline", exc)
            plan = _fallback_plan(req, n_chapters)
        chapters = [Chapter(number=i, plan=p) for i, p in enumerate(plan.chapters, 1)]
        await emit({
            "type": "outline",
            "title": plan.title,
            "subtitle": plan.subtitle,
            "chapters": [c.plan.title for c in chapters],
        })

        # Chapters take turns in slots sized like the EURI concurrency cap, so the
        # "writing" status the UI shows matches what is really in flight.
        slots = asyncio.Semaphore(settings.max_concurrency)

        async def write(ch: Chapter) -> None:
            try:
                async with slots:
                    await _write(ch)
            except EuriError as exc:
                logger.warning("chapter %s failed: %s", ch.number, exc)
                ch.status = "failed"
                ch.markdown = "_This chapter could not be generated. Please try again._"
                await emit({"type": "chapter", "number": ch.number, "status": "failed", "title": ch.plan.title})

        async def _write(ch: Chapter) -> None:
            model = writer_model
            if profile.deep_writer and remaining() < 120:
                model = settings.writer_model  # fall back to the fast model near the deadline
            await emit({"type": "chapter", "number": ch.number, "status": "writing", "title": ch.plan.title})
            raw = await client.chat(
                _writer_messages(req, plan, ch, profile.words),
                model=model, temperature=0.6, max_tokens=profile.max_tokens,
            )
            ch.markdown = clean_chapter_markdown(raw, ch.plan.title)
            ch.issues = verify_chapter(ch.markdown, ch.plan, profile.words)
            ch.status = "written"

            if profile.critic and remaining() > 60:
                await emit({"type": "chapter", "number": ch.number, "status": "reviewing", "title": ch.plan.title})
                try:
                    verdict = await client.chat_json(
                        _critic_messages(plan, ch), model=settings.critic_model,
                        temperature=0.0, max_tokens=800,
                    )
                    if isinstance(verdict, dict) and verdict.get("verdict") == "revise":
                        ch.issues += [str(d) for d in (verdict.get("defects") or [])][:5]
                except (ValueError, EuriError) as exc:
                    logger.warning("critic failed for chapter %s: %s", ch.number, exc)

            if ch.issues and remaining() > 45:
                await emit({"type": "chapter", "number": ch.number, "status": "revising", "title": ch.plan.title})
                revised = await client.chat(
                    _refiner_messages(req, plan, ch, profile.words),
                    model=settings.writer_model, temperature=0.4, max_tokens=profile.max_tokens,
                )
                revised = clean_chapter_markdown(revised, ch.plan.title)
                if len(revised) > len(ch.markdown) * 0.5:
                    ch.markdown = revised
                    ch.status = "revised"
                ch.issues = verify_chapter(ch.markdown, ch.plan, profile.words)
            if not ch.issues:
                ch.status = "verified"
            await emit({
                "type": "chapter", "number": ch.number, "status": ch.status,
                "title": ch.plan.title, "words": len(re.findall(r"\w+", ch.markdown)),
            })

        # The client semaphore caps EURI concurrency at 2. Chapters still running
        # when the time budget ends are cancelled so the book always ships.
        tasks = [asyncio.create_task(write(c)) for c in chapters]
        _, pending = await asyncio.wait(tasks, timeout=max(remaining() - 8, 1))
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            for ch in chapters:
                if ch.status in ("pending", "written") and not ch.markdown:
                    ch.status = "failed"
                    ch.markdown = "_This chapter ran out of time. Try Quick mode or fewer chapters._"
                    await emit({"type": "chapter", "number": ch.number, "status": "failed", "title": ch.plan.title})
                elif ch.status == "written":
                    await emit({"type": "chapter", "number": ch.number, "status": "written",
                                "title": ch.plan.title, "words": len(re.findall(r"\w+", ch.markdown))})
        if all(c.status == "failed" for c in chapters):
            raise EuriError("Every chapter failed to generate; EURI may be unavailable.")

        cover_url = None
        if cover_task:
            try:
                cover_url = await asyncio.wait_for(cover_task, timeout=max(remaining() - 5, 1))
            except asyncio.TimeoutError:
                logger.warning("cover image timed out; continuing without it")
        await emit({"type": "stage", "stage": "compile", "message": "Compiling Markdown and LaTeX"})
        markdown = assemble_markdown(plan, chapters, cover_url)
        latex = markdown_book_to_latex(plan.title, plan.subtitle, [(c.plan.title, c.markdown) for c in chapters])
        return BookResult(
            request=req, plan=plan, chapters=chapters, markdown=markdown, latex=latex,
            cover_url=cover_url, seconds=round(time.monotonic() - started, 1),
            usage=client.usage.as_dict(), models=models,
        )
    finally:
        if own_client:
            await client.aclose()
