"""Book compiler: never reports a stale PDF from an earlier run."""

from __future__ import annotations

from types import SimpleNamespace

from bookforge.agents import compiler as compiler_mod
from bookforge.config import Settings
from bookforge.schemas import VideoRecord
from bookforge.tools.workspace import Workspace


async def test_stale_pdf_is_not_reported_when_compile_is_skipped(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, workspace_root=str(tmp_path), compile_latex=False, book_title="My Title"
    )
    monkeypatch.setattr(compiler_mod, "get_settings", lambda: settings)
    ws = Workspace(settings.workspace_root_abs, "chan")
    manifest = ws.init_manifest("https://youtube.com/playlist?list=x", "Chan", "chan")
    rec = VideoRecord(
        video_id="abc", title="T", url="https://youtu.be/abc", chapter_number=1, status="verified"
    )
    manifest.videos.append(rec)
    ws.save_manifest(manifest)
    (ws.chapter_dir(rec) / "chapter.tex").write_text("\\chapter{T}\nBody.", encoding="utf-8")
    ws.book_dir.mkdir(parents=True, exist_ok=True)
    (ws.book_dir / "book.pdf").write_bytes(b"%PDF stale")

    ctx = SimpleNamespace(session=SimpleNamespace(state={"channel_slug": "chan"}))
    events = [e async for e in compiler_mod.BookCompilerAgent()._run_async_impl(ctx)]

    assert not (ws.book_dir / "book.pdf").exists()
    assert events[-1].actions.state_delta["book_pdf"] is None
    assert "\\title{My Title}" in (ws.root / "main.tex").read_text(encoding="utf-8")
