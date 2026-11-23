"""Timing + token instrumentation for BookForge agents (ADK callbacks).

Every agent in the chapter pipeline gets a before/after agent callback pair
that accumulates wall-clock seconds per stage into session state under
`stage_timings`, and every LlmAgent gets an after-model callback that adds
prompt/completion tokens and call counts under `llm_usage`. The orchestrator
resets both keys per chapter, snapshots them into `metrics.json`, and feeds
them to the ProgressTracker so ETA and the final summary reflect real costs.

All writes go through the callback context so ADK records them as normal
state deltas — no concurrent mutation of session.state.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse

logger = logging.getLogger(__name__)

STAGE_TIMINGS_KEY = "stage_timings"
LLM_USAGE_KEY = "llm_usage"

# (invocation_id, agent_name) -> monotonic start. Agents run sequentially
# inside one invocation, so a plain dict is safe; entries are popped on exit.
_agent_started: dict[tuple[str, str], float] = {}
_model_started: dict[tuple[str, str], float] = {}


def _merge(state: Any, key: str, name: str, **increments: float) -> None:
    current = dict(state.get(key) or {})
    entry = dict(current.get(name) or {})
    for field, value in increments.items():
        entry[field] = float(entry.get(field, 0.0)) + float(value)
    current[name] = entry
    state[key] = current  # reassign so the delta is recorded


def before_agent_timer(callback_context: CallbackContext) -> None:
    key = (callback_context.invocation_id, callback_context.agent_name)
    _agent_started[key] = time.monotonic()
    return None


def after_agent_timer(callback_context: CallbackContext) -> None:
    key = (callback_context.invocation_id, callback_context.agent_name)
    started = _agent_started.pop(key, None)
    if started is None:
        return None
    elapsed = time.monotonic() - started
    _merge(
        callback_context.state,
        STAGE_TIMINGS_KEY,
        callback_context.agent_name,
        seconds=elapsed,
        runs=1,
    )
    return None


def before_model_usage(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    key = (callback_context.invocation_id, callback_context.agent_name)
    _model_started[key] = time.monotonic()
    return None


def after_model_usage(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    if getattr(llm_response, "partial", False):
        return None
    key = (callback_context.invocation_id, callback_context.agent_name)
    started = _model_started.pop(key, None)
    latency = time.monotonic() - started if started is not None else 0.0
    usage = getattr(llm_response, "usage_metadata", None)
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion = int(getattr(usage, "candidates_token_count", 0) or 0)
    _merge(
        callback_context.state,
        LLM_USAGE_KEY,
        callback_context.agent_name,
        calls=1,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_sec=latency,
    )
    return None


def agent_callbacks() -> dict[str, Any]:
    """kwargs for any BaseAgent/LlmAgent constructor (stage timing)."""
    return {
        "before_agent_callback": [before_agent_timer],
        "after_agent_callback": [after_agent_timer],
    }


def llm_callbacks() -> dict[str, Any]:
    """kwargs for an LlmAgent constructor (stage timing + token usage)."""
    return {
        **agent_callbacks(),
        "before_model_callback": [before_model_usage],
        "after_model_callback": [after_model_usage],
    }


def summarize_usage(llm_usage: dict[str, Any] | None) -> tuple[int, int, int]:
    """(calls, prompt_tokens, completion_tokens) across all agents."""
    calls = prompt = completion = 0
    for entry in (llm_usage or {}).values():
        calls += int(entry.get("calls", 0))
        prompt += int(entry.get("prompt_tokens", 0))
        completion += int(entry.get("completion_tokens", 0))
    return calls, prompt, completion


def stage_seconds(stage_timings: dict[str, Any] | None) -> dict[str, float]:
    return {
        name: round(float(entry.get("seconds", 0.0)), 2)
        for name, entry in (stage_timings or {}).items()
    }
