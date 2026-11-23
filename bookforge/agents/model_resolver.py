"""Model resolution for BookForge's Google ADK agents: every call goes to Euron EURI.

EURI (https://euron.one/euri/docs) is OpenAI-compatible, so the agents use ADK's
``OpenAILlm`` pointed at the EURI base URL with ``EURI_API_KEY`` as the Bearer
token. ``EuriLlm`` adds two things on top:

* JSON sanitising for structured-output calls (fences, chatter, duplicate braces).
* A process-wide semaphore so at most ``llm_max_concurrency`` (default 2, EURI's
  per-account limit) requests are in flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from google.adk.labs.openai import OpenAILlm
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from openai import AsyncOpenAI

from bookforge.config import Settings

logger = logging.getLogger(__name__)

MISSING_KEY_MESSAGE = (
    "EURI_API_KEY is not set. Create a key at https://euron.one/euri and export "
    "EURI_API_KEY before running BookForge."
)


def extract_clean_json(text: str) -> str:
    """Extract valid JSON from raw LLM output which might contain markdown, duplicate braces, or thinking artifacts."""
    text = text.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "chapter_title" in data:
            return text
    except json.JSONDecodeError:
        pass

    # Find balanced JSON object candidates; prefer the one with chapter_title.
    starts = [i for i, c in enumerate(text) if c == "{"]
    ends = [i for i, c in enumerate(text) if c == "}"]

    valid_candidates: list[tuple[int, str]] = []
    for start in starts:
        for end in reversed(ends):
            if end > start:
                candidate = text[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "chapter_title" in parsed:
                        return candidate
                    if isinstance(parsed, dict):
                        valid_candidates.append((len(candidate), candidate))
                except json.JSONDecodeError:
                    continue

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        return valid_candidates[0][1]

    return text


# One semaphore per (event loop, limit): EURI's concurrency cap is per account,
# so every agent in the process shares it.
_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}


def euri_semaphore(limit: int) -> asyncio.Semaphore:
    """Return the shared semaphore for the running event loop."""
    key = (id(asyncio.get_running_loop()), max(1, limit))
    sem = _SEMAPHORES.get(key)
    if sem is None:
        if len(_SEMAPHORES) > 32:
            _SEMAPHORES.clear()
        sem = _SEMAPHORES[key] = asyncio.Semaphore(key[1])
    return sem


class EuriLlm(OpenAILlm):
    """ADK model that talks to the Euron EURI API with JSON cleaning and a concurrency cap."""

    max_concurrency: int = 2
    has_api_key: bool = True

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        if not self.has_api_key:
            raise RuntimeError(MISSING_KEY_MESSAGE)
        async with euri_semaphore(self.max_concurrency):
            async for resp in super().generate_content_async(llm_request, stream=stream):
                if not resp.partial and resp.content and resp.content.parts:
                    for part in resp.content.parts:
                        if part.text and llm_request.config and llm_request.config.response_schema:
                            part.text = extract_clean_json(part.text)
                yield resp


# Backward-compatible name from the NVIDIA NIM era.
RobustNvidiaLlm = EuriLlm


def resolve_model(model_name: str, settings: Settings) -> str | BaseLlm:
    """Resolve a stage's model name to an ``EuriLlm`` bound to the EURI endpoint.

    Construction never fails without a key (so the agent graph can be built and
    inspected offline); the first actual LLM call raises a clear error instead.
    """
    api_key = settings.euri_api_key
    return EuriLlm(
        model=model_name,
        max_tokens=settings.llm_max_output_tokens,
        max_concurrency=settings.llm_max_concurrency,
        has_api_key=bool(api_key),
        client=AsyncOpenAI(
            base_url=settings.euri_api_base,
            api_key=api_key or "missing-euri-api-key",
        ),
    )


__all__: list[Any] = ["EuriLlm", "RobustNvidiaLlm", "extract_clean_json", "resolve_model", "euri_semaphore"]
