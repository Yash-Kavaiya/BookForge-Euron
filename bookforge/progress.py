"""Human-readable pipeline progress, ETA and run-level metrics.

`ProgressTracker` keeps the rolling per-chapter ETA and, fed with each
chapter's `ChapterMetrics`, accumulates stage timings, LLM calls/tokens, QA
iterations and cache hits so a run can be compared before/after a change.
"""

from __future__ import annotations

import time
from typing import Any

from bookforge.schemas import ChapterMetrics


def format_duration(seconds: float) -> str:
    """Format seconds as 3m12s / 1h05m."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class ProgressTracker:
    """Rolling per-chapter ETA for BookProductionAgent events."""

    def __init__(self, total: int, default_chapter_sec: float = 8 * 60) -> None:
        self.total = max(0, total)
        self.done = 0
        self.failed = 0
        self.started_at = time.monotonic()
        self.durations: list[float] = []
        self.default_chapter_sec = default_chapter_sec
        self._chapter_started_at: float | None = None
        self.chapters: list[ChapterMetrics] = []
        self.stage_totals: dict[str, float] = {}
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.qa_iterations = 0
        self.qa_llm_critic_calls = 0
        self.cache_hits: dict[str, int] = {}

    # -- metrics ------------------------------------------------------------

    @property
    def chapter_elapsed_sec(self) -> float:
        if self._chapter_started_at is None:
            return 0.0
        return time.monotonic() - self._chapter_started_at

    def record_chapter(self, metrics: ChapterMetrics) -> None:
        self.chapters.append(metrics)
        for stage, sec in metrics.stage_sec.items():
            self.stage_totals[stage] = self.stage_totals.get(stage, 0.0) + float(sec)
        self.llm_calls += metrics.llm_calls
        self.prompt_tokens += metrics.prompt_tokens
        self.completion_tokens += metrics.completion_tokens
        self.qa_iterations += metrics.qa_iterations
        self.qa_llm_critic_calls += metrics.qa_llm_critic_calls
        for hit in metrics.cache_hits:
            self.cache_hits[hit] = self.cache_hits.get(hit, 0) + 1

    def run_metrics(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        verified = sum(1 for c in self.chapters if c.status == "verified")
        return {
            "chapters_total": self.total,
            "chapters_done": self.done,
            "chapters_failed": self.failed,
            "chapters_verified": verified,
            "wall_sec": round(elapsed, 1),
            "avg_chapter_sec": round(self.average_chapter_sec(), 1),
            "stage_sec": {k: round(v, 1) for k, v in sorted(self.stage_totals.items())},
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "qa_iterations": self.qa_iterations,
            "qa_llm_critic_calls": self.qa_llm_critic_calls,
            "cache_hits": dict(sorted(self.cache_hits.items())),
            "chapters": [c.model_dump(mode="json") for c in self.chapters],
        }

    def metrics_line(self) -> str:
        parts = [
            f"LLM calls {self.llm_calls}",
            f"tokens {self.prompt_tokens}+{self.completion_tokens}",
            f"QA iterations {self.qa_iterations}",
            f"critic calls {self.qa_llm_critic_calls}",
        ]
        if self.cache_hits:
            hits = ", ".join(f"{k}x{v}" for k, v in sorted(self.cache_hits.items()))
            parts.append(f"cache hits {hits}")
        if self.stage_totals:
            top = sorted(self.stage_totals.items(), key=lambda kv: -kv[1])[:4]
            parts.append(
                "slowest stages "
                + ", ".join(f"{k} {format_duration(v)}" for k, v in top)
            )
        return "; ".join(parts) + "."

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    def average_chapter_sec(self) -> float:
        if self.durations:
            return sum(self.durations) / len(self.durations)
        return self.default_chapter_sec

    def eta_sec(self, remaining: int | None = None) -> float:
        left = self.remaining if remaining is None else remaining
        return self.average_chapter_sec() * left

    def start_chapter(self, index: int, title: str, extra: str = "") -> str:
        self._chapter_started_at = time.monotonic()
        elapsed = time.monotonic() - self.started_at
        left = max(0, self.total - index)
        suffix = f" {extra}" if extra else ""
        return (
            f"[{index + 1}/{self.total}] Starting: {title}.{suffix} "
            f"Elapsed {format_duration(elapsed)}. "
            f"ETA ~{format_duration(self.eta_sec(left))} "
            f"({left} left, ~{format_duration(self.average_chapter_sec())}/chapter)."
        )

    def finish_chapter(self, index: int, title: str, status: str) -> str:
        now = time.monotonic()
        if self._chapter_started_at is not None:
            elapsed_ch = now - self._chapter_started_at
            # Ignore sub-second timings (tests / cache hits) so ETA stays useful.
            if elapsed_ch >= 1.0:
                self.durations.append(elapsed_ch)
        self._chapter_started_at = None
        self.done += 1
        if status in {"failed", "error"}:
            self.failed += 1
        elapsed = now - self.started_at
        left = max(0, self.total - self.done)
        return (
            f"[{index + 1}/{self.total}] Finished: {title} ({status}). "
            f"Elapsed {format_duration(elapsed)}. "
            f"ETA ~{format_duration(self.eta_sec(left))} "
            f"({left} left, {self.failed} failed)."
        )

    def summary(self) -> str:
        elapsed = time.monotonic() - self.started_at
        base = (
            f"Production done: {self.done}/{self.total} chapters "
            f"({self.failed} failed) in {format_duration(elapsed)}."
        )
        if self.chapters:
            return f"{base} {self.metrics_line()}"
        return base
