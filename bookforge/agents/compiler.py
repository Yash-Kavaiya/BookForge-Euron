"""BookCompilerAgent: assemble main.tex from verified chapters -> book.pdf."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from bookforge.config import get_settings
from bookforge.tools import branding as brand
from bookforge.tools import latex as latex_tools
from bookforge.tools import visuals
from bookforge.tools.workspace import Workspace

logger = logging.getLogger(__name__)


def cover_prompt(title: str, subtitle: str, chapter_titles: list[str]) -> str:
    topics = "; ".join(t for t in chapter_titles[:6] if t)
    return (
        f"Elegant minimalist book cover artwork for a book titled '{title}'"
        + (f" ({subtitle})" if subtitle else "")
        + (f". Themes: {topics}" if topics else "")
        + ". Editorial flat illustration, rich but calm colours, strong central "
        "motif, generous negative space, no text, no letters, no logos, no watermark."
    )


class BookCompilerAgent(BaseAgent):
    name: str = "book_compiler"
    description: str = (
        "Assembles the book preamble, title page, TOC and every completed "
        "chapter into main.tex, compiles with pdflatex and publishes book.pdf."
    )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        settings = get_settings()
        state = ctx.session.state
        ws = Workspace(settings.workspace_root_abs, state["channel_slug"])
        manifest = ws.load_manifest()

        completed = [
            v
            for v in sorted(manifest.videos, key=lambda x: x.chapter_number)
            if v.status in ("verified", "written")
        ]
        failed = [v for v in manifest.videos if v.status == "failed"]
        if not completed:
            raise RuntimeError("No completed chapters — nothing to compile.")

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=f"Compiling book from {len(completed)} completed chapters "
                        f"({len(failed)} failed excluded). This can take a few minutes."
                    )
                ],
            ),
        )

        chapter_rel_dirs = []
        chapter_links = []
        for record in completed:
            chapter_dir = ws.chapter_dir(record)
            ch_tex_path = chapter_dir / "chapter.tex"
            if ch_tex_path.exists():
                raw_tex = Workspace.read_text(ch_tex_path)
                clean_tex = latex_tools.sanitize_chapter_tex(raw_tex)
                Workspace.write_text(ch_tex_path, clean_tex)
                chapter_rel_dirs.append(chapter_dir.relative_to(ws.root).as_posix())
                # Per-chapter footer link: prefer the clean youtu.be short form
                # (no & or ? beyond the id) so hyperref never chokes on it.
                if record.video_id:
                    chapter_links.append(
                        {
                            "url": f"https://youtu.be/{record.video_id}",
                            "link_text": f"youtu.be/{record.video_id}",
                        }
                    )
                else:
                    chapter_links.append({"url": record.url, "link_text": record.url})
            else:
                logger.warning("chapter.tex missing for %s — skipped", record.video_id)

        channel_name = manifest.channel_title or "YouTube Channel"
        book_title = settings.book_title.strip() or f"{channel_name} — The Complete Book"
        credit = "Compiled by BookForge-Euron (Google ADK agents on the Euron EURI API)"
        author_line = settings.book_author.strip() or credit
        subtitle = settings.book_subtitle.strip()

        branding = await asyncio.to_thread(brand.resolve_branding, settings, ws.root)
        cover_rel = await self._cover(settings, ws, book_title, subtitle, completed)
        title_page = brand.title_page_tex(
            book_title,
            subtitle,
            author_line,
            logo_rel=branding.logos.get("title", ""),
            credit=credit if author_line != credit else "",
        )
        cover_page = (
            brand.cover_page_tex(cover_rel, book_title, subtitle, author_line if author_line != credit else "")
            if cover_rel
            else ""
        )
        part_titles = [r.part_title for r in completed if (ws.chapter_dir(r) / "chapter.tex").exists()]

        Workspace.write_text(ws.root / "preamble.tex", latex_tools.PREAMBLE)
        main_tex = Workspace.write_text(
            ws.root / "main.tex",
            latex_tools.assemble_main_tex(
                book_title,
                author_line,
                chapter_rel_dirs,
                channel_name=channel_name,
                chapter_links=chapter_links,
                branding_tex=brand.header_footer_tex(branding),
                title_page=title_page,
                cover_page=cover_page,
                list_of_figures=settings.list_of_figures,
                part_titles=part_titles,
            ),
        )

        try:
            await asyncio.to_thread(
                brand.write_source_zip,
                ws.root,
                ws.book_dir / "latex_source.zip",
                chapter_rel_dirs,
            )
        except Exception:  # noqa: BLE001 - the zip is a convenience
            logger.exception("could not write latex_source.zip")

        compile_ok, log_tail = False, "compile disabled"
        # A book.pdf from an earlier run no longer matches main.tex: never
        # report it as this run's output.
        (ws.book_dir / "book.pdf").unlink(missing_ok=True)
        if settings.compile_latex:
            compile_ok, log_tail = latex_tools.compile_tex(
                main_tex, ws.book_dir, passes=2, cwd=ws.root
            )
            produced = ws.book_dir / "main.pdf"
            if not compile_ok and (cover_page or branding.customized):
                # Branding must never cost the book: retry with the plain front matter.
                logger.error("branded compile failed; retrying without cover/branding:\n%s", log_tail)
                Workspace.write_text(
                    main_tex,
                    latex_tools.assemble_main_tex(
                        book_title,
                        author_line,
                        chapter_rel_dirs,
                        channel_name=channel_name,
                        chapter_links=chapter_links,
                        list_of_figures=settings.list_of_figures,
                        part_titles=part_titles,
                    ),
                )
                compile_ok, log_tail = latex_tools.compile_tex(
                    main_tex, ws.book_dir, passes=2, cwd=ws.root
                )
            if compile_ok and produced.exists():
                produced.replace(ws.book_dir / "book.pdf")

        book_meta = {
            "title": book_title,
            "channel_url": manifest.channel_url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "chapters_included": len(chapter_rel_dirs),
            "chapters_failed": [v.video_id for v in failed],
            "pdf": "book/book.pdf" if (ws.book_dir / "book.pdf").exists() else None,
            "cover": "book/cover.png" if (ws.book_dir / "cover.png").exists() else None,
            "latex_source": "book/latex_source.zip"
            if (ws.book_dir / "latex_source.zip").exists()
            else None,
            "logos": branding.logos,
            "parts": sorted({p for p in part_titles if p}),
        }
        Workspace.write_text(
            ws.book_dir / "book_manifest.json", json.dumps(book_meta, indent=2)
        )

        if settings.compile_latex and not compile_ok:
            logger.error("book compile failed:\n%s", log_tail)
            raise RuntimeError(f"pdflatex failed on the book: {log_tail[:800]}")

        pdf_path = book_meta["pdf"] or "(compile disabled)"
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=f"Book compiled: {len(chapter_rel_dirs)} chapters "
                        f"({len(failed)} videos failed and were excluded). "
                        f"PDF: {pdf_path}"
                    )
                ],
            ),
            actions=EventActions(
                state_delta={
                    "book_pdf": book_meta["pdf"],
                    "chapters_included": len(chapter_rel_dirs),
                    "chapters_failed": len(failed),
                }
            ),
        )


    async def _cover(self, settings, ws: Workspace, title: str, subtitle: str, records) -> str:
        """Workspace-relative cover image path ('' when there is none).

        Uses BOOKFORGE_COVER_IMAGE when given, else generates one with the
        image model (cached in branding/cover.png). Also copies it to
        book/cover.png. Never raises.
        """
        dest = ws.root / "branding" / "cover.png"
        try:
            if settings.cover_image:
                src = Path(settings.cover_image).expanduser()
                if src.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    from PIL import Image

                    with Image.open(src) as img:
                        img.convert("RGB").save(dest, format="PNG")
                else:
                    logger.warning("cover image %s not found", src)
            elif settings.cover_generate and not dest.exists():
                prompt = settings.cover_prompt or cover_prompt(
                    title, subtitle, [r.title for r in records]
                )
                await visuals.generate_image(
                    prompt,
                    dest,
                    api_key=settings.euri_api_key,
                    api_base=settings.euri_api_base,
                    model=settings.image_model,
                    max_concurrency=settings.llm_max_concurrency,
                )
            if dest.exists():
                ws.book_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dest, ws.book_dir / "cover.png")
                return dest.relative_to(ws.root).as_posix()
        except Exception:  # noqa: BLE001 - no cover is fine, no book is not
            logger.exception("cover step failed; continuing without a cover")
        return ""
