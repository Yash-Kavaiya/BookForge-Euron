"""Per-stage LLM routing and speed/QA settings."""

from bookforge.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_refiner_falls_back_to_writer_model():
    s = _settings(writer_model="strong", refiner_model="")
    assert s.effective_refiner_model == "strong"
    assert s.model_for_stage("refiner") == "strong"
    s = _settings(writer_model="strong", refiner_model="fast")
    assert s.model_for_stage("refiner") == "fast"


def test_model_and_temperature_per_stage():
    s = _settings(
        analyst_model="a", writer_model="w", critic_model="c",
        analyst_temperature=0.0, writer_temperature=0.5, critic_temperature=0.1,
    )
    assert [s.model_for_stage(x) for x in ("analyst", "writer", "refiner", "critic")] == ["a", "w", "w", "c"]
    assert s.temperature_for_stage("analyst") == 0.0
    assert s.temperature_for_stage("refiner") == 0.5  # refiner shares writer temperature
    assert s.temperature_for_stage("critic") == 0.1


def test_env_prefix_routing(monkeypatch):
    monkeypatch.setenv("BOOKFORGE_CRITIC_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("BOOKFORGE_REFINER_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("BOOKFORGE_QA_LLM_CRITIC_MODE", "never")
    monkeypatch.setenv("BOOKFORGE_FRAME_HASH_WORKERS", "8")
    s = _settings()
    assert s.critic_model == "gemini-2.5-flash-lite"
    assert s.effective_refiner_model == "gemini-2.5-flash"
    assert s.qa_llm_critic_mode == "never"
    assert s.frame_hash_workers == 8


def test_speed_and_qa_defaults_are_sane():
    s = _settings()
    assert s.qa_llm_critic_mode == "once"
    assert s.qa_strict_numbers is True
    assert s.reuse_cached_analysis is True
    assert s.reuse_written_chapters is True
    assert s.asset_render_workers >= 1
    assert s.frame_hash_workers >= 1
    assert s.media_prefetch_concurrency >= 1
    assert 0.0 <= s.critic_temperature <= s.writer_temperature <= 1.0
