"""Video ordering helper used when creating a new manifest."""

from bookforge.agents.intake import order_videos
from bookforge.config import Settings


def _sample_videos() -> list[dict]:
    return [
        {"video_id": "c", "title": "Part 3", "upload_date": "20250301", "duration_sec": 600, "url": "x"},
        {"video_id": "a", "title": "Part 1", "upload_date": "20250101", "duration_sec": 600, "url": "x"},
        {"video_id": "b", "title": "Part 2", "upload_date": "20250201", "duration_sec": 600, "url": "x"},
    ]


# -- mode: playlist --


def test_playlist_keeps_given_order():
    videos = _sample_videos()
    result = order_videos(videos, "playlist")
    assert [v["video_id"] for v in result] == ["c", "a", "b"]


# -- mode: reverse --


def test_reverse_flips_order():
    videos = _sample_videos()
    result = order_videos(videos, "reverse")
    assert [v["video_id"] for v in result] == ["b", "a", "c"]


# -- mode: auto with all dates present --


def test_auto_sorts_by_upload_date_when_all_present():
    videos = _sample_videos()
    result = order_videos(videos, "auto")
    assert [v["video_id"] for v in result] == ["a", "b", "c"]


def test_auto_with_missing_dates_keeps_given_order():
    videos = [
        {"video_id": "x", "title": "A", "upload_date": "", "duration_sec": 100, "url": "x"},
        {"video_id": "y", "title": "B", "upload_date": "20250101", "duration_sec": 100, "url": "x"},
    ]
    result = order_videos(videos, "auto")
    assert [v["video_id"] for v in result] == ["x", "y"]


def test_auto_with_all_empty_dates_keeps_given_order():
    videos = [
        {"video_id": "x", "title": "A", "upload_date": "", "duration_sec": 100, "url": "x"},
        {"video_id": "y", "title": "B", "upload_date": "", "duration_sec": 100, "url": "x"},
    ]
    result = order_videos(videos, "auto")
    assert [v["video_id"] for v in result] == ["x", "y"]


# -- no mutation of input list --


def test_playlist_does_not_mutate_input():
    videos = _sample_videos()
    original = list(videos)
    order_videos(videos, "playlist")
    assert videos == original


def test_reverse_does_not_mutate_input():
    videos = _sample_videos()
    original = list(videos)
    order_videos(videos, "reverse")
    assert videos == original


def test_auto_does_not_mutate_input():
    videos = _sample_videos()
    original = list(videos)
    order_videos(videos, "auto")
    assert videos == original


# -- invalid mode --


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown video_order"):
        order_videos([], "bad_mode")


# -- Settings reads BOOKFORGE_VIDEO_ORDER from env --


def test_settings_video_order_from_env(monkeypatch):
    monkeypatch.setenv("BOOKFORGE_VIDEO_ORDER", "reverse")
    s = Settings(_env_file=None)
    assert s.video_order == "reverse"


def test_settings_video_order_default():
    s = Settings(_env_file=None)
    assert s.video_order == "auto"
