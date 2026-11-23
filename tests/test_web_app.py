"""API tests for the FastAPI web app (EURI mocked, no network)."""

from __future__ import annotations

import base64
import io
import json
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient

import bookforge.web.pipeline as pipeline_mod
from bookforge.web.euri_client import EuriClient
from tests.test_web_pipeline import fake_euri


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("BOOKFORGE_WEB_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("BOOKFORGE_ACCESS_CODE", raising=False)
    from app import app

    return TestClient(app)


def test_index_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "BookForge" in r.text


def test_health_reports_key_state_without_value(client, monkeypatch):
    monkeypatch.delenv("EURI_API_KEY", raising=False)
    assert client.get("/api/health").json()["euri_configured"] is False
    monkeypatch.setenv("EURI_API_KEY", "super-secret-value")
    r = client.get("/api/health")
    assert r.json()["euri_configured"] is True
    assert "super-secret-value" not in r.text


def test_generate_without_key_gives_clear_error(client, monkeypatch):
    monkeypatch.delenv("EURI_API_KEY", raising=False)
    r = client.post("/api/generate", json={"topic": "tea"})
    assert r.status_code == 503
    assert "EURI_API_KEY" in r.json()["error"]


def test_generate_validates_input(client, monkeypatch):
    monkeypatch.setenv("EURI_API_KEY", "test-key")
    r = client.post("/api/generate", json={"topic": "x"})
    assert r.status_code == 422


def test_access_code_gate(client, monkeypatch):
    monkeypatch.setenv("EURI_API_KEY", "test-key")
    monkeypatch.setenv("BOOKFORGE_ACCESS_CODE", "letmein")
    assert client.post("/api/generate", json={"topic": "tea time"}).status_code == 401


def test_generate_streams_events_and_files(client, monkeypatch, tmp_path):
    monkeypatch.setenv("EURI_API_KEY", "test-key")
    real_init = EuriClient.__init__

    def mocked_init(self, settings, **kw):
        kw["transport"] = fake_euri()
        real_init(self, settings, **kw)

    monkeypatch.setattr(pipeline_mod.EuriClient, "__init__", mocked_init)
    with client.stream("POST", "/api/generate", json={"topic": "history of tea", "mode": "quick"}) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line.strip()]
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "done" and "outline" in types
    done = events[-1]
    assert done["markdown"].startswith("# Tea & Time")
    assert r"\begin{document}" in done["latex"]
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(done["zip_base64"]))).namelist()
    assert {n.split("/")[-1] for n in names} == {"book.md", "book.tex", "book.json", "README.txt"}
    assert (tmp_path / done["job_id"] / "book.md").exists()  # written under the /tmp-style root
