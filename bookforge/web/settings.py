"""Environment-driven settings for the web app (no secrets are ever logged)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_EURI_BASE = "https://api.euron.one/api/v1/euri"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class WebSettings:
    """Runtime configuration for the web pipeline, read from env vars."""

    api_key: str = field(repr=False, default="")
    api_base: str = DEFAULT_EURI_BASE
    planner_model: str = "euri-chat-fast"
    writer_model: str = "euri-chat-fast"
    critic_model: str = "euri-chat-fast"
    deep_writer_model: str = "euri-chat-smart"
    image_model: str = "euri-image-fast"
    max_concurrency: int = 2
    request_timeout_s: int = 120
    time_budget_s: int = 270
    output_root: str = "/tmp/bookforge"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "WebSettings":
        return cls(
            api_key=os.environ.get("EURI_API_KEY", "").strip(),
            api_base=_env("BOOKFORGE_EURI_API_BASE", DEFAULT_EURI_BASE).rstrip("/"),
            planner_model=_env("BOOKFORGE_WEB_PLANNER_MODEL", "euri-chat-fast"),
            writer_model=_env("BOOKFORGE_WEB_WRITER_MODEL", "euri-chat-fast"),
            critic_model=_env("BOOKFORGE_WEB_CRITIC_MODEL", "euri-chat-fast"),
            deep_writer_model=_env("BOOKFORGE_WEB_DEEP_WRITER_MODEL", "euri-chat-smart"),
            image_model=_env("BOOKFORGE_WEB_IMAGE_MODEL", "euri-image-fast"),
            max_concurrency=max(1, _env_int("BOOKFORGE_LLM_MAX_CONCURRENCY", 2)),
            request_timeout_s=_env_int("BOOKFORGE_WEB_REQUEST_TIMEOUT", 120),
            time_budget_s=_env_int("BOOKFORGE_WEB_TIME_BUDGET", 270),
            output_root=_env("BOOKFORGE_WEB_OUTPUT_DIR", "/tmp/bookforge"),
        )
