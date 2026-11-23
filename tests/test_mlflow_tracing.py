"""Regression: Python ADK must keep its TracerProvider and persist sessions."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from bookforge.integrations.mlflow_integration import (
    ADK_WEB_USER_ID,
    apply_otel_env,
    session_db_path,
    setup_mlflow_tracing,
)


def test_session_db_is_the_adk_web_store():
    path = session_db_path()
    assert path.name == "session.db"
    assert path.parent.name == ".adk"
    assert path.parent.parent.name == "bookforge"


def test_cli_user_matches_adk_web():
    assert ADK_WEB_USER_ID == "user"


def test_apply_otel_env_matches_official_spec(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    apply_otel_env("http://127.0.0.1:5000", "2")
    import os

    assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:5000"
    assert (
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
        == "http://127.0.0.1:5000/v1/traces"
    )
    assert os.environ["OTEL_EXPORTER_OTLP_HEADERS"] == "x-mlflow-experiment-id=2"
    assert os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"


def test_setup_does_not_replace_existing_provider(monkeypatch):
    import bookforge.integrations.mlflow_integration as mod

    existing = TracerProvider()
    trace.set_tracer_provider(existing)

    fake_mlflow = SimpleNamespace(
        google_adk=None,
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: SimpleNamespace(experiment_id="2"),
    )
    monkeypatch.setenv("BOOKFORGE_DISABLE_MLFLOW_TRACING", "0")
    monkeypatch.setattr(mod, "_tracking_reachable", lambda uri, timeout=2.0: True)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    # Never attach a real OTLP exporter to the process-wide provider: with a
    # SimpleSpanProcessor every later test's spans would block on HTTP retries.
    attached: list = []
    monkeypatch.setattr(
        mod, "_attach_otlp_exporter", lambda provider, uri, exp_id: attached.append(provider)
    )
    monkeypatch.setattr(mod, "_STATE", dict(mod._STATE))  # restored after the test

    assert setup_mlflow_tracing(
        tracking_uri="http://127.0.0.1:5000", experiment_name="my-experiment"
    )
    assert trace.get_tracer_provider() is existing
    assert attached == [existing]  # exporter added to the existing provider, not a new one
    assert mod.mlflow_enabled() is True


def test_setup_disabled_by_env(monkeypatch):
    monkeypatch.setenv("BOOKFORGE_DISABLE_MLFLOW_TRACING", "1")
    assert setup_mlflow_tracing() is False


def test_metric_logging_is_noop_when_disabled(monkeypatch):
    import bookforge.integrations.mlflow_integration as mod

    monkeypatch.setattr(mod, "_STATE", {**mod._STATE, "enabled": False})
    assert mod.log_chapter_metrics({"video_id": "v", "wall_sec": 1.0}) is False
    assert mod.log_run_metrics({"wall_sec": 1.0}) is False


def test_metric_logging_flattens_and_logs_when_enabled(monkeypatch):
    import bookforge.integrations.mlflow_integration as mod

    logged: dict = {"params": {}, "metrics": {}, "dicts": [], "runs": []}

    class _Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_mlflow = SimpleNamespace(
        start_run=lambda run_name=None: logged["runs"].append(run_name) or _Run(),
        log_params=lambda p: logged["params"].update(p),
        log_metrics=lambda m: logged["metrics"].update(m),
        log_dict=lambda d, name: logged["dicts"].append(name),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(mod, "_STATE", {**mod._STATE, "enabled": True})

    metrics = {
        "video_id": "abc",
        "chapter_number": 3,
        "status": "verified",
        "wall_sec": 120.5,
        "stage_sec": {"chapter_writer": 60.0},
        "llm_calls": 4,
        "cache_hits": ["media"],
    }
    assert mod.log_chapter_metrics(metrics) is True
    assert logged["runs"] == ["chapter-03-abc"]
    assert logged["params"] == {"video_id": "abc", "chapter_number": "3", "status": "verified"}
    assert logged["metrics"]["chapter.wall_sec"] == 120.5
    assert logged["metrics"]["chapter.stage_sec.chapter_writer"] == 60.0
    assert logged["metrics"]["chapter.llm_calls"] == 4.0
    assert "chapter.cache_hits" not in logged["metrics"]  # lists are not metrics

    assert mod.log_run_metrics({"wall_sec": 9.0, "chapters": [{}]}, run_name="r") is True
    assert logged["metrics"]["run.wall_sec"] == 9.0
    assert logged["dicts"] == ["run_metrics.json"]


def test_cli_no_longer_uses_in_memory_sessions():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "bookforge" / "main.py"
    text = src.read_text(encoding="utf-8")
    assert "InMemorySessionService" not in text
    assert "create_session_service" in text
    assert "ADK_WEB_USER_ID" in text
    assert "flush_mlflow_tracing" in text
