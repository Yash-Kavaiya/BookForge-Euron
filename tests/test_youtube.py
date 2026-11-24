"""VTT -> timestamped text conversion (pure parsing, no network)."""

from pathlib import Path

from bookforge.tools import youtube as yt
from bookforge.tools.youtube import load_transcript_text, vtt_to_text

SAMPLE_VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
welcome to this tutorial

00:00:03.000 --> 00:00:05.000
welcome to this tutorial

00:00:05.500 --> 00:00:08.000
today we cover <c>neural</c> networks
"""


def test_vtt_to_text_strips_and_dedupes(tmp_path):
    vtt = tmp_path / "captions.en.vtt"
    vtt.write_text(SAMPLE_VTT, encoding="utf-8")
    out = vtt_to_text(vtt, tmp_path)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[00:00:01] welcome to this tutorial"
    # consecutive duplicate cue collapsed
    assert sum("welcome" in line for line in lines) == 1
    assert "[00:00:05] today we cover neural networks" in lines
    # idempotent: second call returns existing file unchanged
    assert vtt_to_text(vtt, tmp_path) == out


def test_load_transcript_text_samples_whole_length(tmp_path):
    p = tmp_path / "transcript.txt"
    lines = [f"[00:{i // 60:02d}:{i % 60:02d}] line {i}" for i in range(3000)]
    p.write_text("\n".join(lines), encoding="utf-8")
    text = load_transcript_text(p, max_chars=3000, windows=5)
    assert len(text) <= 3000
    assert "transcript excerpt gap" in text
    assert text.startswith("[00:00:00] line 0")
    assert text.endswith("line 2999")
    # the middle of the video is represented, not just head and tail
    assert "line 1500" in text or any(f"line {n}" in text for n in range(1300, 1700))
    # every kept line is whole (starts with a timestamp)
    for chunk_line in text.splitlines():
        assert chunk_line.startswith("[") or "excerpt gap" in chunk_line or not chunk_line


def test_load_transcript_text_short_is_untouched(tmp_path):
    p = tmp_path / "transcript.txt"
    p.write_text("[00:00:00] hi", encoding="utf-8")
    assert load_transcript_text(p, max_chars=400) == "[00:00:00] hi"


def test_original_caption_langs_prefers_spoken_language():
    info = {
        "language": "hi",
        "automatic_captions": {"en": [], "hi": [], "hi-orig": [], "fr": []},
    }
    assert yt.original_caption_langs(info) == ["hi-orig", "hi"]
    assert yt.original_caption_langs({"automatic_captions": {"ta-orig": []}}) == ["ta-orig"]
    assert yt.original_caption_langs({}) == []


class _FakeYDL:
    calls: list = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        langs = self.opts["subtitleslangs"]
        _FakeYDL.calls.append(langs)
        if langs == ["hi-orig"]:
            dest = Path(self.opts["outtmpl"]).parent
            (dest / "captions.hi-orig.vtt").write_text("WEBVTT\n", encoding="utf-8")
        # English request: yt-dlp swallows the HTTP 429 and writes nothing

    def extract_info(self, url, download=False):
        return {"language": "hi", "automatic_captions": {"en": [], "hi-orig": []}}


def test_fetch_captions_falls_back_to_original_language(tmp_path, monkeypatch):
    _FakeYDL.calls = []
    monkeypatch.setattr(yt, "YoutubeDL", _FakeYDL)
    got = yt.fetch_captions("https://youtu.be/x", tmp_path)
    assert got is not None and got.name == "captions.hi-orig.vtt"
    assert _FakeYDL.calls == [["en.*", "en"], ["hi-orig"]]
    # cached on rerun: no new downloads
    assert yt.fetch_captions("https://youtu.be/x", tmp_path) == got
    assert len(_FakeYDL.calls) == 2


def test_list_channel_videos_skips_unavailable(monkeypatch):
    class _ListYDL(_FakeYDL):
        def extract_info(self, url, download=False):
            return {
                "entries": [
                    {"id": "aaa", "title": "Real talk", "duration": 60},
                    {"id": "bbb", "title": None, "duration": None},
                    {"id": "ccc", "title": "[Private video]", "duration": None},
                ]
            }

    monkeypatch.setattr(yt, "YoutubeDL", _ListYDL)
    _title, vids = yt.list_channel_videos("https://www.youtube.com/playlist?list=PLx")
    assert [v["video_id"] for v in vids] == ["aaa"]


def test_download_video_ignores_partial_files(tmp_path, monkeypatch):
    (tmp_path / "video.mp4.part").write_bytes(b"x")
    (tmp_path / "video.mp4.ytdl").write_bytes(b"x")

    class _DL(_FakeYDL):
        def download(self, urls):
            assert self.opts["retries"] >= 3 and self.opts["continuedl"]
            (tmp_path / "video.mp4").write_bytes(b"done")

    monkeypatch.setattr(yt, "YoutubeDL", _DL)
    assert yt.download_video("https://youtu.be/x", tmp_path).name == "video.mp4"
