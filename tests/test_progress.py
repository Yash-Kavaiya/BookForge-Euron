"""Progress / ETA formatting."""

from bookforge.progress import ProgressTracker, format_duration
from bookforge.schemas import ChapterMetrics


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(9) == "9s"
    assert format_duration(75) == "1m15s"
    assert format_duration(3661) == "1h01m"


def test_eta_uses_default_then_rolling_average():
    tracker = ProgressTracker(total=4, default_chapter_sec=120)
    assert tracker.eta_sec() == 480
    msg = tracker.start_chapter(0, "Intro")
    assert "[1/4] Starting: Intro" in msg
    assert "ETA ~" in msg
    done = tracker.finish_chapter(0, "Intro", "written")
    assert "[1/4] Finished: Intro (written)" in done
    assert tracker.done == 1
    assert tracker.remaining == 3
    assert tracker.eta_sec() == 360  # still using default avg after a sub-second finish


def test_failed_chapters_counted():
    tracker = ProgressTracker(total=2, default_chapter_sec=10)
    tracker.start_chapter(0, "A")
    tracker.finish_chapter(0, "A", "failed")
    assert tracker.failed == 1
    assert "1 failed" in tracker.summary()


def test_chapter_metrics_aggregate_into_run_metrics():
    tracker = ProgressTracker(total=2, default_chapter_sec=10)
    tracker.start_chapter(0, "A")
    tracker.finish_chapter(0, "A", "verified")
    tracker.record_chapter(
        ChapterMetrics(
            video_id="a",
            chapter_number=1,
            status="verified",
            wall_sec=100.0,
            stage_sec={"chapter_writer": 60.0, "media_acquisition": 30.0},
            llm_calls=3,
            prompt_tokens=10_000,
            completion_tokens=4_000,
            qa_iterations=1,
            qa_llm_critic_calls=1,
            cache_hits=["media"],
        )
    )
    tracker.start_chapter(1, "B")
    tracker.finish_chapter(1, "B", "written")
    tracker.record_chapter(
        ChapterMetrics(
            video_id="b",
            chapter_number=2,
            status="written",
            wall_sec=200.0,
            stage_sec={"chapter_writer": 90.0, "chapter_refiner": 80.0},
            llm_calls=5,
            prompt_tokens=20_000,
            completion_tokens=6_000,
            qa_iterations=3,
            qa_llm_critic_calls=1,
            cache_hits=["media", "analysis_json"],
        )
    )
    run = tracker.run_metrics()
    assert run["chapters_done"] == 2 and run["chapters_verified"] == 1
    assert run["llm_calls"] == 8
    assert run["prompt_tokens"] == 30_000 and run["completion_tokens"] == 10_000
    assert run["qa_iterations"] == 4 and run["qa_llm_critic_calls"] == 2
    assert run["stage_sec"] == {
        "chapter_refiner": 80.0,
        "chapter_writer": 150.0,
        "media_acquisition": 30.0,
    }
    assert run["cache_hits"] == {"analysis_json": 1, "media": 2}
    assert len(run["chapters"]) == 2
    summary = tracker.summary()
    assert "LLM calls 8" in summary
    assert "tokens 30000+10000" in summary
    assert "cache hits analysis_jsonx1, mediax2" in summary
    assert "slowest stages chapter_writer" in summary
