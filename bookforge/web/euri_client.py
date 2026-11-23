"""Minimal async client for the Euron EURI API (OpenAI-compatible).

* Bearer auth from ``EURI_API_KEY`` (never logged or echoed).
* A shared semaphore caps in-flight requests (EURI allows 2 concurrent).
* Retries with backoff on 429 / 5xx / network errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from bookforge.web.settings import WebSettings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# EURI's concurrency limit is per account, so every client in this process
# shares one semaphore (per event loop) instead of each job getting its own.
_SEMAPHORES: dict[tuple[int, int], asyncio.Semaphore] = {}


def _shared_semaphore(limit: int) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), limit)
    sem = _SEMAPHORES.get(key)
    if sem is None:
        if len(_SEMAPHORES) > 32:  # drop semaphores of loops that are gone
            _SEMAPHORES.clear()
        sem = _SEMAPHORES[key] = asyncio.Semaphore(limit)
    return sem


class EuriError(RuntimeError):
    """Raised when the EURI API cannot fulfil a request."""


class MissingApiKeyError(EuriError):
    """Raised when EURI_API_KEY is not configured."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, usage: dict[str, Any] | None) -> None:
        self.calls += 1
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply that may include fences or chatter."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("model reply did not contain valid JSON")


class EuriClient:
    """Async EURI client with a concurrency cap and retries."""

    def __init__(
        self,
        settings: WebSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        if not settings.api_key:
            raise MissingApiKeyError(
                "EURI_API_KEY is not set. Get a key at https://euron.one/euri and "
                "add it as an environment variable."
            )
        self.settings = settings
        self.usage = Usage()
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=settings.api_base,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=httpx.Timeout(settings.request_timeout_s, connect=15),
            transport=transport,
        )

    async def __aenter__(self) -> "EuriClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self._max_retries + 1):
            async with _shared_semaphore(self.settings.max_concurrency):
                try:
                    resp = await self._client.post(path, json=payload)
                except httpx.HTTPError as exc:  # network / timeout
                    last_error = f"{type(exc).__name__}"
                    resp = None
            if resp is not None:
                if resp.status_code < 400:
                    return resp.json()
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code in (401, 403):
                    raise EuriError(
                        "EURI rejected the API key (HTTP %d). Check EURI_API_KEY."
                        % resp.status_code
                    )
                if resp.status_code not in RETRYABLE_STATUS:
                    raise EuriError(last_error)
            if attempt < self._max_retries:
                delay = min(2**attempt * 2, 20)
                logger.warning("EURI %s failed (%s); retry in %ss", path, last_error, delay)
                await asyncio.sleep(delay)
        raise EuriError(f"EURI request to {path} failed: {last_error}")

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 4000,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await self._post("/chat/completions", payload)
        self.usage.add(data.get("usage"))
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise EuriError("EURI returned an unexpected response shape") from exc
        return content.strip()

    async def chat_json(
        self, messages: list[dict[str, str]], *, model: str, **kwargs: Any
    ) -> Any:
        """Chat call that must return JSON; retries once with a nudge on bad JSON."""
        reply = await self.chat(messages, model=model, json_mode=True, **kwargs)
        try:
            return extract_json(reply)
        except ValueError:
            nudged = messages + [
                {"role": "assistant", "content": reply[:2000]},
                {"role": "user", "content": "Reply again with ONLY valid JSON, no prose."},
            ]
            return extract_json(await self.chat(nudged, model=model, json_mode=True, **kwargs))

    async def image(self, prompt: str, *, model: str, size: str = "1024x1024") -> str | None:
        """Generate one image and return its (temporary) URL, or None."""
        data = await self._post(
            "/images/generations", {"model": model, "prompt": prompt, "n": 1, "size": size}
        )
        items = data.get("data") or []
        if not items:
            return None
        first = items[0]
        if first.get("url"):
            return first["url"]
        if first.get("b64_json"):
            return "data:image/png;base64," + first["b64_json"]
        return None
