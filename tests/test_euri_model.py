"""The ADK pipeline routes every LLM call to the Euron EURI API."""

from __future__ import annotations

import asyncio

import pytest
from google.adk.labs.openai import OpenAILlm
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from bookforge.agents import model_resolver
from bookforge.agents.model_resolver import EuriLlm, RobustNvidiaLlm, resolve_model
from bookforge.config import Settings

EURI_BASE = "https://api.euron.one/api/v1/euri"


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_settings_default_to_euri(monkeypatch):
    monkeypatch.delenv("EURI_API_KEY", raising=False)
    monkeypatch.delenv("BOOKFORGE_EURI_API_KEY", raising=False)
    s = _settings()
    assert s.euri_api_base == EURI_BASE
    assert s.analyst_model == "euri-chat-fast"
    assert s.writer_model == "euri-chat-smart"
    assert s.critic_model == "euri-chat-fast"
    assert s.llm_max_output_tokens <= 8192  # EURI output cap
    assert s.llm_max_concurrency == 2
    assert s.euri_configured is False


def test_key_read_from_euri_api_key(monkeypatch):
    monkeypatch.setenv("EURI_API_KEY", "k-test")
    s = _settings()
    assert s.euri_api_key == "k-test" and s.euri_configured
    monkeypatch.setenv("BOOKFORGE_EURI_API_BASE", "https://example.test/v1")
    assert _settings().euri_api_base == "https://example.test/v1"


def test_resolve_model_returns_euri_llm():
    m = resolve_model("euri-chat-fast", _settings(euri_api_key="k-test"))
    assert isinstance(m, EuriLlm) and RobustNvidiaLlm is EuriLlm
    assert m.model == "euri-chat-fast"
    assert str(m.client.base_url).rstrip("/") == EURI_BASE
    assert m.client.api_key == "k-test"
    assert m.max_tokens == 8192 and m.max_concurrency == 2


async def test_missing_key_fails_on_first_call_not_at_build(monkeypatch):
    monkeypatch.delenv("EURI_API_KEY", raising=False)
    m = resolve_model("euri-chat-fast", _settings(euri_api_key=""))  # building is fine
    with pytest.raises(RuntimeError, match="EURI_API_KEY"):
        async for _ in m.generate_content_async(LlmRequest()):
            pass


async def test_semaphore_caps_concurrency_at_two(monkeypatch):
    state = {"now": 0, "peak": 0}

    async def fake_generate(self, llm_request, stream=False):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.02)
        state["now"] -= 1
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="ok")]))

    monkeypatch.setattr(OpenAILlm, "generate_content_async", fake_generate)
    model_resolver._SEMAPHORES.clear()
    models = [resolve_model("euri-chat-fast", _settings(euri_api_key="k")) for _ in range(3)]

    async def call(m):
        return [r async for r in m.generate_content_async(LlmRequest())]

    results = await asyncio.gather(*(call(models[i % 3]) for i in range(6)))
    assert all(r[0].content.parts[0].text == "ok" for r in results)
    assert state["peak"] == 2  # shared across all model instances


async def test_structured_output_is_json_cleaned(monkeypatch):
    async def fake_generate(self, llm_request, stream=False):
        text = 'Sure!\n```json\n{"chapter_title": "X", "sections": []}\n```'
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))

    monkeypatch.setattr(OpenAILlm, "generate_content_async", fake_generate)
    m = resolve_model("euri-chat-fast", _settings(euri_api_key="k"))
    req = LlmRequest(config=types.GenerateContentConfig(response_schema={"type": "OBJECT"}))
    out = [r async for r in m.generate_content_async(req)]
    assert out[0].content.parts[0].text == '{"chapter_title": "X", "sections": []}'


async def test_invalid_structured_output_is_retried(monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        chapter_title: str
        summary: str

    answers = iter(
        [
            # truncated answer: only an inner object survives JSON cleaning
            '{"chapter_title": "X", "concepts": [{"heading": "h", "explanation": "e"}',
            '{"chapter_title": "X", "summary": "ok"}',
        ]
    )
    calls = {"n": 0}

    async def fake_generate(self, llm_request, stream=False):
        calls["n"] += 1
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=next(answers))]))

    monkeypatch.setattr(OpenAILlm, "generate_content_async", fake_generate)
    m = resolve_model("euri-chat-fast", _settings(euri_api_key="k"))
    req = LlmRequest(config=types.GenerateContentConfig(response_schema=Out))
    out = [r async for r in m.generate_content_async(req)]
    assert calls["n"] == 2
    assert len(out) == 1
    assert out[0].content.parts[0].text == '{"chapter_title": "X", "summary": "ok"}'


async def test_structured_retries_are_bounded(monkeypatch):
    calls = {"n": 0}

    async def fake_generate(self, llm_request, stream=False):
        calls["n"] += 1
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="not json")]))

    monkeypatch.setattr(OpenAILlm, "generate_content_async", fake_generate)
    m = resolve_model("euri-chat-fast", _settings(euri_api_key="k"))
    req = LlmRequest(config=types.GenerateContentConfig(response_schema={"type": "OBJECT"}))
    out = [r async for r in m.generate_content_async(req)]
    assert calls["n"] == 3 and out[0].content.parts[0].text == "not json"
