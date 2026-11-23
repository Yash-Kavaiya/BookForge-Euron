"""End-to-end chapter production through the real ADK runner with a fake LLM.

Exercises the wiring that unit tests cannot: callback-based skipping inside a
LoopAgent, escalate exiting the loop, state deltas flowing between stages,
the resume paths (media/analysis caches, written-chapter reuse), the final
machine verification and the metrics files. No network, no pdflatex.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest
from google.adk.agents import SequentialAgent
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import bookforge.agents.analyst as analyst_mod
import bookforge.agents.critic as critic_mod
import bookforge.agents.orchestrator as orchestrator_mod
import bookforge.agents.writer as writer_mod
from bookforge.agents.orchestrator import BookProductionAgent, build_chapter_pipeline
from bookforge.config import get_settings
from bookforge.schemas import MediaBundle, VideoRecord
from bookforge.tools.workspace import Workspace

TRANSCRIPT = (
    "[00:00:05] welcome to the test video about caching\n"
    "[00:00:20] a cache stores results so repeated work is skipped\n"
    "[00:00:40] hit rates above 90 percent are common for hot keys\n"
)

ANALYSIS = {
    "chapter_title": "Caching Basics",
    "summary": "A cache stores results so repeated work is skipped.",
    "learning_objectives": ["The reader can define a cache", "The reader can explain hit rate"],
    "concepts": [
        {
            "heading": "What a Cache Does",
            "explanation": "A cache stores results so repeated work is skipped.",
            "key_points": ["store results", "skip repeated work"],
            "equations": [],
            "note": "",
            "evidence_timestamps": ["00:00:20"],
        }
    ],
    "tables": [
        {
            "title": "Hit rates",
            "caption": "Typical hit rates",
            "headers": ["Keys", "Hit rate"],
            "rows": [["hot", "90 percent"]],
        }
    ],
    "charts": [],
    "diagrams": [],
    "glossary": [{"term": "hit rate", "definition": "share of lookups served from the cache"}],
    "exercises": [{"question": "What does a cache store?", "answer": "Results of earlier work."}],
    "frame_captions": [],
}

FILLER = " ".join(["Caches store results so repeated work is skipped reliably."] * 70)


def chapter(table_path: str, extra_sentence: str = "") -> str:
    return (
        "\\chapter{Caching Basics}\\label{ch:01-caching}\n"
        "\\section*{Overview}\nA cache stores results so repeated work is skipped. " + FILLER + "\n"
        "\\section*{Learning Objectives}\n\\begin{itemize}\\item The reader can define a cache"
        "\\item The reader can explain hit rate\\end{itemize}\n"
        "\\section{What a Cache Does}\nA cache stores results so repeated work is skipped. "
        + extra_sentence
        + "\n\\section*{Key Data and Comparisons}\nTable~\\ref{tab:ch01-1} lists typical hit rates.\n"
        f"\\input{{{table_path}}}\n"
        "\\section*{Key Takeaways}\n\\begin{itemize}\\item Skip repeated work.\\end{itemize}\n"
        "\\section*{Glossary}\n\\begin{description}\\item[hit rate] share of lookups served\\end{description}\n"
        "\\section*{Exercises}\n\\begin{enumerate}\\item What does a cache store?\\end{enumerate}\n"
        "\\section*{Solutions}\n\\begin{enumerate}\\item Results of earlier work.\\end{enumerate}\n"
    )


class ScriptedLlm(BaseLlm):
    """Dispatches on the agent's system instruction; records every call."""

    model: str = "scripted"
    calls: list[str] = []
    table_path: str = ""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        system = str(llm_request.config.system_instruction or "")
        if "Transcript Analyst" in system:
            kind, text = "analyst", json.dumps(ANALYSIS)
        elif "Chapter Writer" in system:
            # first draft smuggles in a statistic that is in neither the
            # analysis nor the transcript -> the verifier must catch it
            kind, text = "writer", chapter(self.table_path, "Hit rates reach 47.3 percent on 12,500 keys.")
        elif "Chapter Refiner" in system:
            assert "unsupported_number" in system, "refiner must receive the machine defect list"
            assert "47.3" in system
            kind, text = "refiner", chapter(self.table_path)
        elif "Chapter Critic" in system:
            assert "machine checks" in system.lower() or "passed" in system.lower()
            kind, text = "critic", json.dumps({"verdict": "pass", "summary": "Faithful.", "defects": []})
        else:  # pragma: no cover
            raise AssertionError(f"unexpected prompt: {system[:120]}")
        self.calls.append(kind)
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1000, candidates_token_count=200
            ),
        )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOKFORGE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("BOOKFORGE_COMPILE_LATEX", "false")
    monkeypatch.setenv("BOOKFORGE_QA_LLM_CRITIC_MODE", "once")
    monkeypatch.setenv("BOOKFORGE_QA_MAX_ITERATIONS", "3")
    monkeypatch.setenv("BOOKFORGE_OPENAI_API_KEY", "")
    monkeypatch.setenv("BOOKFORGE_DISABLE_MLFLOW_TRACING", "1")
    # Fully offline: never touch an MLflow server even if another test enabled it.
    monkeypatch.setattr(orchestrator_mod, "log_chapter_metrics", lambda *a, **k: False)
    monkeypatch.setattr(orchestrator_mod, "log_run_metrics", lambda *a, **k: False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _seed_workspace(root, status: str = "pending") -> tuple[Workspace, VideoRecord]:
    ws = Workspace(root, "demo")
    record = VideoRecord(
        video_id="v1", title="Test Video", url="https://youtu.be/v1", chapter_number=1, status=status
    )
    if not ws.manifest_exists():
        manifest = ws.init_manifest("https://youtube.com/@demo", "Demo", "demo")
        manifest.videos.append(record)
        ws.save_manifest(manifest)
    else:
        ws.update_video("v1", status=status)
    vdir = ws.video_dir("v1")
    (vdir / "transcript.txt").write_text(TRANSCRIPT, encoding="utf-8")
    bundle = MediaBundle(
        video_id="v1",
        video_path=str(vdir / "video.mp4"),
        transcript_path=str(vdir / "transcript.txt"),
        transcript_source="captions",
        frames=[],
    )
    (vdir / "media.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return ws, record


def _build(monkeypatch, llm: ScriptedLlm) -> SequentialAgent:
    for mod in (analyst_mod, writer_mod, critic_mod):
        monkeypatch.setattr(mod, "resolve_model", lambda name, settings: llm)
    settings = get_settings()
    return SequentialAgent(
        name="production_only",
        sub_agents=[BookProductionAgent(pipeline=build_chapter_pipeline(settings))],
    )


async def _run(root: SequentialAgent) -> list:
    session_service = InMemorySessionService()
    runner = Runner(agent=root, app_name="bf-test", session_service=session_service)
    session = await session_service.create_session(
        app_name="bf-test", user_id="u", session_id="s1", state={"channel_slug": "demo"}
    )
    msg = types.Content(role="user", parts=[types.Part(text="produce chapters")])
    events = []
    async for event in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
        events.append(event)
    return events


def _texts(events, author: str) -> list[str]:
    out = []
    for e in events:
        if e.author == author and e.content and e.content.parts:
            out.append("".join(p.text or "" for p in e.content.parts))
    return out


@pytest.mark.asyncio
async def test_chapter_flows_verifier_refiner_critic_to_verified(env, monkeypatch):
    ws, record = _seed_workspace(env)
    table_path = f"chapters/{ws.chapter_dir(record).name}/tables/table_1.tex"
    llm = ScriptedLlm(table_path=table_path, calls=[])
    root = _build(monkeypatch, llm)

    events = await _run(root)

    # LLM budget: analyst, first draft, one refine, one critic judgement
    assert llm.calls == ["analyst", "writer", "refiner", "critic"]

    verifier_msgs = _texts(events, "chapter_verifier")
    assert len(verifier_msgs) == 2
    assert "blocking" in verifier_msgs[0] and "LLM critic skipped" in verifier_msgs[0]
    assert "machine checks passed" in verifier_msgs[1] and "LLM critic" in verifier_msgs[1]
    # iteration 1: the critic emitted a defect-less skip verdict, not an LLM judgement
    critic_msgs = _texts(events, "chapter_critic")
    assert len(critic_msgs) == 2
    assert json.loads(critic_msgs[0])["summary"].startswith("LLM critic skipped")
    assert json.loads(critic_msgs[1])["verdict"] == "pass"
    # loop exit: the refiner gate re-emits the unchanged chapter and escalates
    refiner_msgs = _texts(events, "chapter_refiner")
    assert refiner_msgs[-1].startswith("\\chapter{Caching Basics}")
    finish = [m for m in _texts(events, "book_production") if "Finished" in m][0]
    assert "(verified)" in finish and "approved by the LLM critic" in finish
    assert "Cache hits: media" in finish

    manifest = ws.load_manifest()
    assert manifest.by_id("v1").status == "verified"
    assert ws.pending_videos() == []

    chapter_tex = (ws.chapter_dir(record) / "chapter.tex").read_text(encoding="utf-8")
    assert "47.3" not in chapter_tex
    assert (ws.chapter_dir(record) / "qa_report_1.json").exists()
    assert (ws.chapter_dir(record) / "qa_report_2.json").exists()
    assert (ws.chapter_dir(record) / "analysis.json").exists()

    metrics = json.loads((ws.chapter_dir(record) / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "verified"
    assert metrics["llm_calls"] == 4
    assert metrics["prompt_tokens"] == 4000 and metrics["completion_tokens"] == 800
    assert metrics["qa_iterations"] == 2
    assert metrics["qa_llm_critic_calls"] == 1
    assert metrics["qa_defects_final"] == 0
    assert "media" in metrics["cache_hits"]
    for stage in ("media_acquisition", "transcript_analyst", "visual_assets", "chapter_writer",
                  "chapter_verifier", "chapter_critic", "chapter_refiner"):
        assert stage in metrics["stage_sec"], stage

    run_metrics = json.loads((ws.root / "run_metrics.json").read_text(encoding="utf-8"))
    assert run_metrics["chapters_verified"] == 1
    assert run_metrics["llm_calls"] == 4
    summary = _texts(events, "book_production")[-1]
    assert "LLM calls 4" in summary and "critic calls 1" in summary


@pytest.mark.asyncio
async def test_rerun_reuses_analysis_and_skips_analyst(env, monkeypatch):
    ws, record = _seed_workspace(env)
    table_path = f"chapters/{ws.chapter_dir(record).name}/tables/table_1.tex"
    llm = ScriptedLlm(table_path=table_path, calls=[])
    await _run(_build(monkeypatch, llm))
    assert llm.calls[0] == "analyst"

    # Force the chapter to be produced again; the transcript is unchanged so
    # the cached analysis must be reused and the analyst never called.
    ws.update_video("v1", status="pending")
    llm2 = ScriptedLlm(table_path=table_path, calls=[])
    events = await _run(_build(monkeypatch, llm2))
    assert "analyst" not in llm2.calls
    assert llm2.calls[0] == "writer"
    # the analyst's skip event carries the cached analysis itself
    analyst_msgs = _texts(events, "transcript_analyst")
    assert json.loads(analyst_msgs[0])["chapter_title"] == "Caching Basics"
    finish = [m for m in _texts(events, "book_production") if "Finished" in m][0]
    assert "analysis_json" in finish
    metrics = json.loads((ws.chapter_dir(record) / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["cache_hits"]) >= {"media", "analysis_json"}
    assert ws.load_manifest().by_id("v1").status == "verified"


@pytest.mark.asyncio
async def test_written_chapter_resumes_in_qa_loop_without_writer(env, monkeypatch):
    ws, record = _seed_workspace(env, status="written")
    table_path = f"chapters/{ws.chapter_dir(record).name}/tables/table_1.tex"
    # a previous run left a clean chapter on disk but never got it verified
    Workspace.write_text(ws.chapter_dir(record) / "chapter.tex", chapter(table_path))
    llm = ScriptedLlm(table_path=table_path, calls=[])
    events = await _run(_build(monkeypatch, llm))

    assert "writer" not in llm.calls
    assert "refiner" not in llm.calls
    assert llm.calls == ["analyst", "critic"]
    # the writer's skip event is the cached chapter (ADK stores it under chapter_tex)
    assert _texts(events, "chapter_writer")[0].startswith("\\chapter{Caching Basics}")
    finish = [m for m in _texts(events, "book_production") if "Finished" in m][0]
    assert "Resuming previously written chapter" in " ".join(_texts(events, "book_production"))
    assert "chapter_tex" in finish
    assert ws.load_manifest().by_id("v1").status == "verified"
    metrics = json.loads((ws.chapter_dir(record) / "metrics.json").read_text(encoding="utf-8"))
    assert "chapter_tex" in metrics["cache_hits"]
    assert metrics["qa_iterations"] == 1


@pytest.mark.asyncio
async def test_machine_only_mode_never_calls_critic(env, monkeypatch):
    monkeypatch.setenv("BOOKFORGE_QA_LLM_CRITIC_MODE", "never")
    get_settings.cache_clear()
    ws, record = _seed_workspace(env)
    table_path = f"chapters/{ws.chapter_dir(record).name}/tables/table_1.tex"
    llm = ScriptedLlm(table_path=table_path, calls=[])
    events = await _run(_build(monkeypatch, llm))
    assert llm.calls == ["analyst", "writer", "refiner"]
    assert any("chapter approved" in m for m in _texts(events, "chapter_verifier"))
    assert ws.load_manifest().by_id("v1").status == "verified"
