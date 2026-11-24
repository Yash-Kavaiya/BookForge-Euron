"""Source listing: classify, group, and list_channel_videos (all mocked, no network)."""

from unittest.mock import patch
import pytest

from bookforge.tools.youtube import (
    classify_source_url,
    channel_videos_url,
    channel_playlists_url,
    group_videos_by_part,
    list_channel_videos,
)


# ---------------------------------------------------------------------------
# classify_source_url
# ---------------------------------------------------------------------------

class TestClassifySourceUrl:
    def test_youtu_be(self):
        assert classify_source_url("https://youtu.be/dQw4w9WgXcQ") == "video"

    def test_watch(self):
        assert classify_source_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "video"

    def test_watch_with_list_is_playlist(self):
        assert classify_source_url(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc"
        ) == "playlist"

    def test_shorts(self):
        assert classify_source_url("https://www.youtube.com/shorts/abc123") == "video"

    def test_live(self):
        assert classify_source_url("https://www.youtube.com/live/abc123") == "video"

    def test_handle(self):
        assert classify_source_url("https://www.youtube.com/@LinusTechTips") == "channel"

    def test_handle_with_videos(self):
        assert classify_source_url("https://www.youtube.com/@LinusTechTips/videos") == "channel"

    def test_handle_with_featured(self):
        assert classify_source_url("https://www.youtube.com/@LinusTechTips/featured") == "channel"

    def test_handle_with_streams(self):
        assert classify_source_url("https://www.youtube.com/@LinusTechTips/streams") == "channel"

    def test_channel_id(self):
        assert classify_source_url("https://www.youtube.com/channel/UCabcdef") == "channel"

    def test_c_name(self):
        assert classify_source_url("https://www.youtube.com/c/SomeChannel") == "channel"

    def test_user_name(self):
        assert classify_source_url("https://www.youtube.com/user/OldStyleChannel") == "channel"

    def test_playlist_path(self):
        assert classify_source_url(
            "https://www.youtube.com/playlist?list=PLabcdef"
        ) == "playlist"

    def test_list_without_v(self):
        """URL with list= but no v= is a playlist."""
        assert classify_source_url(
            "https://www.youtube.com/?list=PLabcdef"
        ) == "playlist"

    def test_unknown_defaults_to_playlist(self):
        assert classify_source_url("https://www.youtube.com/some/random/page") == "playlist"


# ---------------------------------------------------------------------------
# channel_videos_url / channel_playlists_url
# ---------------------------------------------------------------------------

class TestChannelUrlHelpers:
    def test_videos_url_from_handle(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_strips_featured(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/featured") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_strips_streams(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/streams") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_strips_shorts(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/shorts") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_strips_about(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/about") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_strips_playlists(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/playlists") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_already_videos(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/videos") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_strips_query(self):
        assert channel_videos_url("https://www.youtube.com/@LinusTechTips/featured?foo=1") == (
            "https://www.youtube.com/@LinusTechTips/videos"
        )

    def test_videos_url_channel_id(self):
        assert channel_videos_url("https://www.youtube.com/channel/UCabcdef") == (
            "https://www.youtube.com/channel/UCabcdef/videos"
        )

    def test_playlists_url_from_handle(self):
        assert channel_playlists_url("https://www.youtube.com/@LinusTechTips") == (
            "https://www.youtube.com/@LinusTechTips/playlists"
        )

    def test_playlists_url_strips_videos(self):
        assert channel_playlists_url("https://www.youtube.com/@LinusTechTips/videos") == (
            "https://www.youtube.com/@LinusTechTips/playlists"
        )


# ---------------------------------------------------------------------------
# group_videos_by_part
# ---------------------------------------------------------------------------

class TestGroupVideosByPart:
    def test_basic_grouping(self):
        videos = [
            {"video_id": "a", "part_title": ""},
            {"video_id": "b", "part_title": ""},
            {"video_id": "c", "part_title": ""},
        ]
        membership = {"a": "Part 1", "c": "Part 2"}
        result = group_videos_by_part(videos, membership, "More from the channel")
        # Assignment
        ids = [v["video_id"] for v in result]
        assert ids == ["a", "c", "b"]  # parts by first video; the fallback part is always last
        assert result[0]["part_title"] == "Part 1"
        assert result[1]["part_title"] == "Part 2"
        assert result[2]["part_title"] == "More from the channel"

    def test_empty_membership(self):
        videos = [{"video_id": "x", "part_title": ""}]
        result = group_videos_by_part(videos, {}, "Unsorted")
        assert result[0]["part_title"] == "Unsorted"

    def test_stable_order(self):
        """Videos within a part keep their original relative order."""
        videos = [
            {"video_id": "a", "part_title": ""},
            {"video_id": "b", "part_title": ""},
            {"video_id": "c", "part_title": ""},
        ]
        membership = {"a": "P1", "b": "P1"}
        result = group_videos_by_part(videos, membership, "Other")
        assert [v["video_id"] for v in result] == ["a", "b", "c"]

    def test_parts_ordered_by_first_video(self):
        """Part order is determined by the position of its first video in the original list."""
        videos = [
            {"video_id": "a", "part_title": ""},
            {"video_id": "b", "part_title": ""},
            {"video_id": "c", "part_title": ""},
        ]
        # b appears in P2 first (pos 1), a in P1 (pos 0) -> P1 comes first
        membership = {"a": "P1", "b": "P2"}
        result = group_videos_by_part(videos, membership, "Other")
        assert [v["part_title"] for v in result] == ["P1", "P2", "Other"]

    def test_custom_fallback(self):
        videos = [{"video_id": "x", "part_title": ""}]
        result = group_videos_by_part(videos, {}, "Custom")
        assert result[0]["part_title"] == "Custom"


# ---------------------------------------------------------------------------
# list_channel_videos: video kind
# ---------------------------------------------------------------------------

class TestListVideo:
    def _make_ydl(self, info_map):
        class _YDL:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                return info_map.get(url, {})
        return _YDL

    def test_single_video(self, monkeypatch):
        info = {
            "id": "abc",
            "title": "My Video",
            "url": "https://www.youtube.com/watch?v=abc",
            "duration": 120,
            "upload_date": "20240101",
            "view_count": 5000,
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl({"https://www.youtube.com/watch?v=abc": info}),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/watch?v=abc")
        assert title == "My Video"
        assert len(vids) == 1
        v = vids[0]
        assert v["video_id"] == "abc"
        assert v["title"] == "My Video"
        assert v["duration_sec"] == 120
        assert v["upload_date"] == "20240101"
        assert v["part_title"] == ""
        assert v["view_count"] == 5000

    def test_single_video_youtu_be(self, monkeypatch):
        info = {
            "id": "xyz",
            "title": "Short",
            "url": "https://youtu.be/xyz",
            "duration": None,
            "upload_date": None,
            "view_count": None,
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl({"https://youtu.be/xyz": info}),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://youtu.be/xyz")
        assert len(vids) == 1
        assert vids[0]["duration_sec"] == 0
        assert vids[0]["view_count"] == 0

    def test_single_video_duration_filter_not_applied(self, monkeypatch):
        """Duration filtering does NOT apply to a single explicitly requested video."""
        info = {
            "id": "abc",
            "title": "Short video",
            "url": "https://www.youtube.com/watch?v=abc",
            "duration": 1,
            "upload_date": "20240101",
            "view_count": 100,
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl({"https://www.youtube.com/watch?v=abc": info}),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        # Even with min_duration_sec=600, the video is returned
        title, vids = lcv("https://www.youtube.com/watch?v=abc", min_duration_sec=600)
        assert len(vids) == 1


# ---------------------------------------------------------------------------
# list_channel_videos: channel newest
# ---------------------------------------------------------------------------

class TestListChannelNewest:
    def _make_ydl(self, info_map):
        class _YDL:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                return info_map.get(url, {"entries": []})
        return _YDL

    def test_channel_newest_order(self, monkeypatch):
        """Newest sort: YouTube order is newest first; after max_videos selection, reverse to oldest-first."""
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": "v3", "title": "Newest", "duration": 60, "upload_date": "20240301", "view_count": 100},
                    {"id": "v2", "title": "Middle", "duration": 120, "upload_date": "20240201", "view_count": 200},
                    {"id": "v1", "title": "Oldest", "duration": 180, "upload_date": "20240101", "view_count": 300},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/@TestChannel", max_videos=3, sort="newest")
        assert title == "TestChannel"
        assert [v["video_id"] for v in vids] == ["v1", "v2", "v3"]  # oldest first

    def test_channel_newest_with_max(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": f"v{i}", "title": f"Video {i}", "duration": 60, "upload_date": "20240101", "view_count": 100}
                    for i in range(10)
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/@TestChannel", max_videos=3, sort="newest")
        # Take first 3 from YT order (newest first), then reverse
        assert [v["video_id"] for v in vids] == ["v2", "v1", "v0"]

    def test_channel_fallback_to_uploader(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "uploader": "Falling Back",
                "entries": [{"id": "v1", "title": "V1", "duration": 60, "upload_date": "20240101", "view_count": 10}],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/@TestChannel", max_videos=1, sort="newest")
        assert title == "Falling Back"

    def test_channel_fallback_to_title_strips_videos(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "title": "Some Channel - Videos",
                "entries": [{"id": "v1", "title": "V1", "duration": 60, "upload_date": "20240101", "view_count": 10}],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/@TestChannel", max_videos=1, sort="newest")
        assert title == "Some Channel"

    def test_channel_duration_filter(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": "v1", "title": "Long", "duration": 600, "upload_date": "20240101", "view_count": 100},
                    {"id": "v2", "title": "Short", "duration": 10, "upload_date": "20240102", "view_count": 50},
                    {"id": "v3", "title": "Long 2", "duration": 300, "upload_date": "20240103", "view_count": 200},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv(
            "https://www.youtube.com/@TestChannel", max_videos=10,
            min_duration_sec=300, max_duration_sec=900, sort="newest"
        )
        # newest-first on YouTube -> the book reads oldest to newest
        assert [v["video_id"] for v in vids] == ["v3", "v1"]

    def test_channel_view_count_default_zero(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": "v1", "title": "V1", "duration": 60, "upload_date": "20240101"},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/@TestChannel", max_videos=1, sort="newest")
        assert vids[0]["view_count"] == 0


# ---------------------------------------------------------------------------
# list_channel_videos: channel popular
# ---------------------------------------------------------------------------

class TestListChannelPopular:
    def _make_ydl(self, info_map):
        class _YDL:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                return info_map.get(url, {"entries": []})
        return _YDL

    def test_channel_popular_order(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": "v1", "title": "Low views", "duration": 60, "upload_date": "20240101", "view_count": 100},
                    {"id": "v2", "title": "High views", "duration": 120, "upload_date": "20240201", "view_count": 10000},
                    {"id": "v3", "title": "Med views", "duration": 180, "upload_date": "20240301", "view_count": 5000},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/@TestChannel", max_videos=3, sort="popular")
        assert [v["video_id"] for v in vids] == ["v2", "v3", "v1"]  # descending view_count


# ---------------------------------------------------------------------------
# list_channel_videos: group_by_playlists
# ---------------------------------------------------------------------------

class TestListGroupByPlaylists:
    def _make_ydl(self, info_map):
        class _YDL:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                return info_map.get(url, {"entries": []})
        return _YDL

    def test_group_by_playlists(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        playlists_url = "https://www.youtube.com/@TestChannel/playlists"
        pl1_url = "https://www.youtube.com/playlist?list=PL1"
        pl2_url = "https://www.youtube.com/playlist?list=PL2"

        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": "v3", "title": "Newest", "duration": 60, "upload_date": "20240301", "view_count": 100},
                    {"id": "v2", "title": "Middle", "duration": 120, "upload_date": "20240201", "view_count": 200},
                    {"id": "v1", "title": "Oldest", "duration": 180, "upload_date": "20240101", "view_count": 300},
                ],
            },
            playlists_url: {
                "entries": [
                    {"id": "PL1", "title": "Tutorial Series"},
                    {"id": "PL2", "title": "Q&A"},
                ],
            },
            pl1_url: {
                "entries": [
                    {"id": "v1"},
                    {"id": "v2"},
                ],
            },
            pl2_url: {
                "entries": [
                    {"id": "v3"},
                ],
            },
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv(
            "https://www.youtube.com/@TestChannel", max_videos=3, sort="newest",
            group_by_playlists=True,
        )
        # After newest: selection is [v3, v2, v1], reversed to [v1, v2, v3]
        # v1 in Tutorial Series (first in original order), v2 in Tutorial Series, v3 in Q&A
        # Grouped: Tutorial Series group first (v1's part), then Q&A (v3's part), then ungrouped
        assert [v["video_id"] for v in vids] == ["v1", "v2", "v3"]
        assert vids[0]["part_title"] == "Tutorial Series"
        assert vids[1]["part_title"] == "Tutorial Series"
        assert vids[2]["part_title"] == "Q&A"

    def test_playlist_fetch_failure_logged(self, monkeypatch, caplog):
        """If fetching a playlist fails, part_title is left "" on every video (never fail intake)."""
        import logging
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        playlists_url = "https://www.youtube.com/@TestChannel/playlists"

        class _FailingYDL:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                if url == channel_url:
                    return {
                        "channel": "TestChannel",
                        "entries": [
                            {"id": "v1", "title": "V1", "duration": 60, "upload_date": "20240101", "view_count": 100},
                        ],
                    }
                elif url == playlists_url:
                    return {
                        "entries": [{"id": "PL1", "title": "Series"}],
                    }
                else:
                    raise ConnectionError("network error")
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            _FailingYDL,
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        with caplog.at_level(logging.WARNING, logger="bookforge.tools.youtube"):
            title, vids = lcv(
                "https://www.youtube.com/@TestChannel", max_videos=1, sort="newest",
                group_by_playlists=True,
            )
        assert any("PL1" in r.getMessage() for r in caplog.records)
        assert title == "TestChannel"
        assert len(vids) == 1
        assert vids[0]["part_title"] == ""

    def test_skip_playlist_entries_without_id(self, monkeypatch):
        channel_url = "https://www.youtube.com/@TestChannel/videos"
        playlists_url = "https://www.youtube.com/@TestChannel/playlists"

        info_map = {
            channel_url: {
                "channel": "TestChannel",
                "entries": [
                    {"id": "v1", "title": "V1", "duration": 60, "upload_date": "20240101", "view_count": 100},
                ],
            },
            playlists_url: {
                "entries": [
                    {"id": "PL1", "title": "Good"},
                    {"title": "No ID"},  # should be skipped
                ],
            },
            "https://www.youtube.com/playlist?list=PL1": {
                "entries": [{"id": "v1"}],
            },
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv(
            "https://www.youtube.com/@TestChannel", max_videos=1, sort="newest",
            group_by_playlists=True,
        )
        assert len(vids) == 1
        assert vids[0]["part_title"] == "Good"


# ---------------------------------------------------------------------------
# list_channel_videos: backwards-compatible playlist (existing behaviour)
# ---------------------------------------------------------------------------

class TestListPlaylist:
    def _make_ydl(self, info_map):
        class _YDL:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                return info_map.get(url, {"entries": []})
        return _YDL

    def test_playlist_backwards_compat(self, monkeypatch):
        """Existing playlist behaviour still works (including part_title='' and view_count)."""
        info_map = {
            "https://www.youtube.com/playlist?list=PLx": {
                "title": "My Playlist",
                "entries": [
                    {"id": "aaa", "title": "Real talk", "duration": 60, "upload_date": "20240101", "view_count": 500},
                    {"id": "bbb", "title": None, "duration": None},
                    {"id": "ccc", "title": "[Private video]", "duration": None},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/playlist?list=PLx")
        assert title == "My Playlist"
        assert len(vids) == 1
        assert vids[0]["video_id"] == "aaa"
        assert vids[0]["part_title"] == ""
        assert vids[0]["view_count"] == 500

    def test_playlist_duration_filter(self, monkeypatch):
        info_map = {
            "https://www.youtube.com/playlist?list=PLx": {
                "title": "My Playlist",
                "entries": [
                    {"id": "a", "title": "Long", "duration": 600, "upload_date": "20240101", "view_count": 100},
                    {"id": "b", "title": "Short", "duration": 10, "upload_date": "20240102", "view_count": 50},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv(
            "https://www.youtube.com/playlist?list=PLx",
            min_duration_sec=120, max_duration_sec=900,
        )
        assert [v["video_id"] for v in vids] == ["a"]

    def test_playlist_overfetch(self, monkeypatch):
        """Playlist uses playlistend = max_videos * 3."""
        captured_opts = {}
        class _YDL:
            def __init__(self, opts):
                captured_opts.update(opts)
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def extract_info(self, url, download=False):
                return {"title": "PL", "entries": []}
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            _YDL,
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        lcv("https://www.youtube.com/playlist?list=PLx", max_videos=10)
        assert captured_opts.get("playlistend") == 30

    def test_playlist_duplicate_filter(self, monkeypatch):
        info_map = {
            "https://www.youtube.com/playlist?list=PLx": {
                "title": "Dupes",
                "entries": [
                    {"id": "a", "title": "First", "duration": 60, "upload_date": "20240101", "view_count": 100},
                    {"id": "a", "title": "First again", "duration": 60, "upload_date": "20240101", "view_count": 100},
                ],
            }
        }
        monkeypatch.setattr(
            __import__("bookforge.tools.youtube", fromlist=["YoutubeDL"]),
            "YoutubeDL",
            self._make_ydl(info_map),
        )
        from bookforge.tools.youtube import list_channel_videos as lcv
        title, vids = lcv("https://www.youtube.com/playlist?list=PLx")
        assert len(vids) == 1
        assert vids[0]["title"] == "First"
