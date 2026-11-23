"""QA loop control flow without an LLM: verifier decisions, critic gating,
refiner escalation, analyst/writer cache callbacks, instrumentation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.adk.events import EventActions

import bookforge.agents.analyst as analyst_mod
import bookforge.agents.critic as critic_mod
import bookforge.agents.verifier as verifier_mod
from bookforge import instrumentation
from bookforge.agents.analyst import (
    analysis_cache_key,
    load_cached_analysis,
    reuse_cached_analysis,
    store_cached_analysis,
)
from bookforge.agents.critic import skip_critic_unless_machine_ok
from bookforge.agents.verifier import (
    CRITIQUE_REPORT_KEY,
    QA_CRITIC_RUNS_KEY,
    QA_MACHINE_OK_KEY,
    QA_REPORT_KEY,
    ChapterVerifierAgent,
    llm_critic_needed,
)
from bookforge.agents.writer import (
    QA_NOTE_KEY,
    gate_refiner_on_critique,
    reuse_written_chapter,
)
from bookforge.config import Settings
from bookforge.schemas import VideoRecord, coerce_critique
from bookforge.tools.workspace import Workspace
from tests.test_qa_checks import TRANSCRIPT, analysis, good_chapter, manifest


class FakeCallbackContext:
    def __init__(self, state: dict, agent_name: str = "agent", invocation_id: str = "inv-1"):
        self.state = state
        self.actions = EventActions()
        self.agent_name = agent_name
        self.invocation_id = invocation_id


@pytest.fixture
def ws(tmp_path) -> Workspace:
    ws = Workspace(tmp_path, "demo")
    ch = ws.root / "chapters" / "01_optimizers"
    (ch / "figures").mkdir(parents=True)
    (ch / "tables").mkdir(parents=True)
    (ch / "figures" / "chart_1.pdf").write_bytes(b"%PDF-1.4")
    (ch / "figures" / "frame_01_5s.jpg").write_bytes(b"\xff\xd8\xff")
    (ch / "tables" / "table_1.tex").write_text(
        r"\begin{table}\caption{x}\label{tab:ch01-1}\end{table}", encoding="utf-8"
    )
    return ws


def _record() -> VideoRecord:
    return VideoRecord(video_id="v1", title="Optimizers", url="u", chapter_number=1)


def _settings(tmp_root, **kw) -> Settings:
    return Settings(_env_file=None, compile_latex=False, workspace_root=tmp_root, **kw)


def _state(ws: Workspace, tex: str) -> dict:
    return {
        "current_video": _record().model_dump(),
        "channel_slug": "demo",
        "chapter_tex": tex,
        "analysis_json": analysis().model_dump(mode="json"),
        "assets_manifest": manifest().model_dump(mode="json"),
        "transcript": TRANSCRIPT,
        "critique": "",
        CRITIQUE_REPORT_KEY: "old-verdict-must-be-cleared",
    }


async def _run_verifier(state: dict) -> list:
    agent = ChapterVerifierAgent()
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))
    return [e async for e in agent._run_async_impl(ctx)]


# --- llm_critic_needed ------------------------------------------------------


def test_llm_critic_mode_semantics():
    assert llm_critic_needed("never", 0) is False
    assert llm_critic_needed("always", 5) is True
    assert llm_critic_needed("once", 0) is True
    assert llm_critic_needed("once", 1) is False


# --- verifier ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_approves_clean_chapter_without_llm(ws, monkeypatch):
    settings = _settings(ws.root.parent, qa_llm_critic_mode="never")
    monkeypatch.setattr(verifier_mod, "get_settings", lambda: settings)
    state = _state(ws, good_chapter())
    events = await _run_verifier(state)
    assert len(events) == 1
    delta = events[0].actions.state_delta
    assert events[0].actions.escalate is True
    assert delta["qa_verdict"] == "pass"
    assert delta[QA_MACHINE_OK_KEY] is True
    assert delta[CRITIQUE_REPORT_KEY] == ""  # stale verdict cleared
    assert delta[QA_REPORT_KEY]["passed"] is True
    assert (ws.chapter_dir(_record()) / "qa_report_1.json").exists()
    assert (ws.chapter_dir(_record()) / "chapter.tex").exists()


@pytest.mark.asyncio
async def test_verifier_hands_clean_chapter_to_llm_critic_once(ws, monkeypatch):
    settings = _settings(ws.root.parent, qa_llm_critic_mode="once")
    monkeypatch.setattr(verifier_mod, "get_settings", lambda: settings)
    state = _state(ws, good_chapter())
    events = await _run_verifier(state)
    delta = events[0].actions.state_delta
    assert not events[0].actions.escalate
    assert delta[QA_MACHINE_OK_KEY] is True
    assert "qa_verdict" not in delta  # critic decides
    # second machine pass after the critic already ran -> accepted without LLM
    state[QA_CRITIC_RUNS_KEY] = 1
    events = await _run_verifier(state)
    assert events[0].actions.escalate is True
    assert events[0].actions.state_delta["qa_verdict"] == "pass"


@pytest.mark.asyncio
async def test_verifier_sends_exact_fixes_to_refiner_and_skips_critic(ws, monkeypatch):
    settings = _settings(ws.root.parent)
    monkeypatch.setattr(verifier_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(critic_mod, "get_settings", lambda: settings)
    broken = good_chapter().replace(
        "chapters/01_optimizers/figures/frame_01_5s.jpg", "chapters/01_optimizers/figures/nope.jpg"
    )
    state = _state(ws, broken)
    events = await _run_verifier(state)
    delta = events[0].actions.state_delta
    assert not events[0].actions.escalate
    assert delta["qa_verdict"] == "fail"
    assert delta[QA_MACHINE_OK_KEY] is False
    assert "CRITICAL missing_file" in delta["critique"]
    assert "nope.jpg" in delta["critique"]
    assert "LLM critic skipped" in events[0].content.parts[0].text

    # the critic's gate honours the machine verdict: no LLM call. ADK stores
    # the skip text under output_key, so it must be a valid CritiqueReport.
    state.update(delta)
    ctx = FakeCallbackContext(state, agent_name="chapter_critic")
    skipped = skip_critic_unless_machine_ok(ctx)
    assert skipped is not None
    report = coerce_critique(skipped.parts[0].text)
    assert report.verdict == "fail" and report.defects == [] and "skipped" in report.summary
    assert state.get(QA_CRITIC_RUNS_KEY, 0) == 0

    # refiner runs against the machine defect list; the defect-less skip
    # verdict must not overwrite it
    state[CRITIQUE_REPORT_KEY] = report.model_dump(mode="json")
    machine_critique = state["critique"]
    assert gate_refiner_on_critique(FakeCallbackContext(state, "chapter_refiner")) is None
    assert state["critique"] == machine_critique
    assert not ctx.actions.escalate
# --- critic gate ------------------------------------------------------------


def test_critic_runs_once_then_budget_is_spent(monkeypatch):
    settings = Settings(_env_file=None, qa_llm_critic_mode="once")
    monkeypatch.setattr(critic_mod, "get_settings", lambda: settings)
    state = {QA_MACHINE_OK_KEY: True, QA_CRITIC_RUNS_KEY: 0}
    assert skip_critic_unless_machine_ok(FakeCallbackContext(state)) is None
    assert state[QA_CRITIC_RUNS_KEY] == 1
    spent = skip_critic_unless_machine_ok(FakeCallbackContext(state))
    assert spent is not None
    assert coerce_critique(spent.parts[0].text).verdict == "pass"  # machine ok -> accept
    settings.qa_llm_critic_mode = "always"
    assert skip_critic_unless_machine_ok(FakeCallbackContext(state)) is None
    assert state[QA_CRITIC_RUNS_KEY] == 2


# --- refiner gate -----------------------------------------------------------


def test_refiner_escalates_on_llm_pass_and_formats_fail():
    passing = {"verdict": "pass", "summary": "Faithful and complete.", "defects": []}
    state = {
        QA_MACHINE_OK_KEY: True,
        CRITIQUE_REPORT_KEY: passing,
        "critique": "x",
        "chapter_tex": r"\chapter{Keep me}",
    }
    ctx = FakeCallbackContext(state, "chapter_refiner")
    content = gate_refiner_on_critique(ctx)
    # ADK stores the returned text under output_key=chapter_tex: it must be
    # the unchanged chapter, never a status message.
    assert content is not None and content.parts[0].text == r"\chapter{Keep me}"
    assert ctx.actions.escalate is True
    assert state["qa_verdict"] == "pass" and state["critique"] == ""
    assert "approved" in state[QA_NOTE_KEY] and "Faithful and complete." in state[QA_NOTE_KEY]

    failing = {
        "verdict": "fail",
        "summary": "Two claims are unsupported.",
        "defects": [
            {"severity": "critical", "code": "unsupported_claim", "message": "says X", "fix": "remove X", "location": "Adam Optimizer"},
            {"severity": "minor", "code": "style", "message": "long sentence"},
        ],
    }
    state = {QA_MACHINE_OK_KEY: True, CRITIQUE_REPORT_KEY: json.dumps(failing)}
    ctx = FakeCallbackContext(state, "chapter_refiner")
    assert gate_refiner_on_critique(ctx) is None
    assert not ctx.actions.escalate
    assert state["critique"].startswith("LLM critic verdict: FAIL.")
    assert "1. CRITICAL unsupported_claim [Adam Optimizer]: says X. FIX: remove X" in state["critique"]
    assert "style" not in state["critique"]  # minors don't block; only blocking defects are sent


def test_refiner_treats_unparsable_verdict_as_fail():
    state = {QA_MACHINE_OK_KEY: True, CRITIQUE_REPORT_KEY: "not json at all"}
    ctx = FakeCallbackContext(state, "chapter_refiner")
    assert gate_refiner_on_critique(ctx) is None
    assert not ctx.actions.escalate
    assert "could not be parsed" in state["critique"]


def test_refiner_never_escalates_on_pass_without_machine_ok():
    passing = {"verdict": "pass", "defects": []}
    state = {QA_MACHINE_OK_KEY: False, CRITIQUE_REPORT_KEY: passing}
    ctx = FakeCallbackContext(state, "chapter_refiner")
    assert gate_refiner_on_critique(ctx) is None
    assert not ctx.actions.escalate
# --- caches -----------------------------------------------------------------


def test_analysis_cache_roundtrip_and_key_sensitivity(tmp_path):
    s1 = Settings(_env_file=None, analyst_model="m1")
    s2 = Settings(_env_file=None, analyst_model="m2")
    key = analysis_cache_key(TRANSCRIPT, s1)
    assert key != analysis_cache_key(TRANSCRIPT + " more", s1)
    assert key != analysis_cache_key(TRANSCRIPT, s2)
    payload = analysis().model_dump(mode="json")
    store_cached_analysis(tmp_path, key, payload)
    assert load_cached_analysis(tmp_path, key) == payload
    assert load_cached_analysis(tmp_path, "other-key") is None
    (tmp_path / "analysis.json").write_text('{"cache_key": "%s", "analysis": {"bad": 1}}' % key)
    assert load_cached_analysis(tmp_path, key) is None  # invalid cache is a miss


def test_reuse_cached_analysis_callback(ws, monkeypatch):
    settings = _settings(ws.root.parent)
    monkeypatch.setattr(analyst_mod, "get_settings", lambda: settings)
    state = _state(ws, "")
    del state["analysis_json"]
    ctx = FakeCallbackContext(state, "transcript_analyst")
    assert reuse_cached_analysis(ctx) is None  # miss -> LLM runs

    key = analysis_cache_key(TRANSCRIPT, settings)
    store_cached_analysis(ws.chapter_dir(_record()), key, analysis().model_dump(mode="json"))
    content = reuse_cached_analysis(ctx)
    assert content is not None
    # ADK validates the skip text against output_schema and stores it under
    # analysis_json, so the content must be the analysis itself.
    assert json.loads(content.parts[0].text) == analysis().model_dump(mode="json")
    assert state["analysis_json"]["chapter_title"] == "Optimizers Compared"
    assert state["cache_hits"] == ["analysis_json"]
    assert state["analysis_cache_key"] == key

    settings.reuse_cached_analysis = False
    assert reuse_cached_analysis(FakeCallbackContext(_state(ws, ""))) is None


def test_reuse_written_chapter_callback():
    state = {"chapter_tex_cached": "", "chapter_tex": ""}
    assert reuse_written_chapter(FakeCallbackContext(state)) is None
    state = {"chapter_tex_cached": r"\chapter{X}", "chapter_tex": ""}
    content = reuse_written_chapter(FakeCallbackContext(state))
    assert content is not None
    assert content.parts[0].text == r"\chapter{X}"  # saved under output_key by ADK
    assert state["chapter_tex"] == r"\chapter{X}"
    assert state["chapter_tex_cached"] == ""
    assert state["cache_hits"] == ["chapter_tex"]
    assert "Writer skipped" in state[QA_NOTE_KEY]


# --- instrumentation --------------------------------------------------------


def test_stage_timer_and_token_usage_accumulate():
    state: dict = {}
    ctx = FakeCallbackContext(state, agent_name="chapter_writer", invocation_id="inv-9")
    instrumentation.before_agent_timer(ctx)
    instrumentation.before_model_usage(ctx, llm_request=None)
    resp = SimpleNamespace(
        partial=False,
        usage_metadata=SimpleNamespace(prompt_token_count=1200, candidates_token_count=340),
    )
    instrumentation.after_model_usage(ctx, resp)
    instrumentation.after_model_usage(ctx, SimpleNamespace(partial=True, usage_metadata=None))
    instrumentation.after_agent_timer(ctx)

    usage = state[instrumentation.LLM_USAGE_KEY]["chapter_writer"]
    assert usage["calls"] == 1
    assert usage["prompt_tokens"] == 1200 and usage["completion_tokens"] == 340
    assert usage["latency_sec"] >= 0
    timings = state[instrumentation.STAGE_TIMINGS_KEY]["chapter_writer"]
    assert timings["runs"] == 1 and timings["seconds"] >= 0

    assert instrumentation.summarize_usage(state[instrumentation.LLM_USAGE_KEY]) == (1, 1200, 340)
    assert instrumentation.stage_seconds(state[instrumentation.STAGE_TIMINGS_KEY]) == {
        "chapter_writer": round(timings["seconds"], 2)
    }
    # an after-timer without a matching before is a no-op, not a crash
    instrumentation.after_agent_timer(FakeCallbackContext({}, "ghost", "inv-x"))
