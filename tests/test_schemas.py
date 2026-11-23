"""Schema validation: round-trip, constraints, coercion helpers."""

import json

import pytest
from pydantic import ValidationError

from bookforge.schemas import (
    AssetsManifest,
    ChannelManifest,
    ChapterAnalysis,
    ChapterMetrics,
    FigureAsset,
    QADefect,
    QAReport,
    TableAsset,
    VideoRecord,
    coerce_analysis,
    coerce_assets_manifest,
    coerce_critique,
    coerce_qa_report,
)


def _minimal_analysis() -> dict:
    return {
        "chapter_title": "Introduction to Neural Networks",
        "summary": "This chapter covers the basics of neural networks.",
        "learning_objectives": [
            "Understand perceptrons",
            "Implement a basic neural network",
        ],
        "concepts": [
            {
                "heading": "Perceptrons",
                "explanation": "A perceptron is the simplest form of a neural network. "
                "It takes inputs and produces an output.",
                "key_points": ["Linear classifier", "Activation function"],
            }
        ],
        "tables": [],
        "charts": [],
        "diagrams": [],
        "glossary": [
            {"term": "perceptron", "definition": "Simplest neural network unit"}
        ],
        "exercises": [
            {
                "question": "What is a perceptron?",
                "answer": "A single-layer neural network.",
            }
        ],
    }


def test_chapter_analysis_roundtrip():
    data = _minimal_analysis()
    analysis = ChapterAnalysis(**data)
    dumped = json.loads(analysis.model_dump_json())
    assert dumped["chapter_title"] == "Introduction to Neural Networks"
    assert len(dumped["concepts"]) == 1


def test_coerce_analysis_from_dict():
    analysis = coerce_analysis(_minimal_analysis())
    assert isinstance(analysis, ChapterAnalysis)


def test_coerce_analysis_from_json_string():
    json_str = json.dumps(_minimal_analysis())
    analysis = coerce_analysis(json_str)
    assert isinstance(analysis, ChapterAnalysis)


def test_coerce_analysis_passthrough():
    original = ChapterAnalysis(**_minimal_analysis())
    assert coerce_analysis(original) is original


def test_video_record_defaults():
    record = VideoRecord(video_id="x", title="X", url="u")
    assert record.status == "pending"
    assert record.chapter_number == 0


def test_channel_manifest_by_id():
    manifest = ChannelManifest(
        channel_url="u",
        channel_slug="s",
        videos=[VideoRecord(video_id="a", title="A", url="u")],
    )
    assert manifest.by_id("a") is not None
    assert manifest.by_id("z") is None


def test_assets_manifest_roundtrip():
    m = AssetsManifest(
        chapter_slug="01_intro",
        chapter_dir="chapters/01_intro",
        figures=[FigureAsset(filename="f.jpg", kind="frame", caption="c", label="l")],
        tables=[TableAsset(tex_file="t.tex", caption="c", label="l")],
    )
    assert coerce_assets_manifest(m.model_dump_json()).chapter_slug == "01_intro"


def test_analysis_min_objectives_enforced():
    data = _minimal_analysis()
    data["learning_objectives"] = ["Only one"]
    with pytest.raises(ValidationError):
        ChapterAnalysis(**data)


def test_concept_evidence_timestamps_optional_and_bounded():
    data = _minimal_analysis()
    assert ChapterAnalysis(**data).concepts[0].evidence_timestamps == []
    data["concepts"][0]["evidence_timestamps"] = ["00:01:00"] * 7
    with pytest.raises(ValidationError):
        ChapterAnalysis(**data)


def test_qa_report_roundtrip_and_blocking():
    report = QAReport(
        passed=False,
        compile_ok=None,
        defects=[
            QADefect(severity="critical", code="missing_file", message="m"),
            QADefect(severity="minor", code="style", message="s"),
        ],
        checks_run=["structure"],
        metrics={"prose_words": 900.0},
        iteration=2,
    )
    again = coerce_qa_report(report.model_dump(mode="json"))
    assert [d.code for d in again.blocking] == ["missing_file"]
    assert coerce_qa_report(report.model_dump_json()).iteration == 2
    assert coerce_qa_report(report) is report


def test_critique_report_is_strict():
    ok = coerce_critique({"verdict": "pass", "defects": []})
    assert ok.verdict == "pass" and ok.blocking == []
    fail = coerce_critique(
        json.dumps(
            {
                "verdict": "fail",
                "summary": "x",
                "defects": [
                    {"severity": "major", "code": "wrong_answer", "message": "m", "fix": "f"},
                    {"severity": "minor", "code": "tone", "message": "m"},
                ],
            }
        )
    )
    assert [d.code for d in fail.blocking] == ["wrong_answer"]
    with pytest.raises(ValidationError):
        coerce_critique({"verdict": "maybe", "defects": []})
    with pytest.raises(ValidationError):
        coerce_critique({"verdict": "fail", "defects": [{"severity": "huge", "code": "c", "message": "m"}]})


def test_chapter_metrics_defaults():
    m = ChapterMetrics(video_id="v")
    assert m.llm_calls == 0 and m.stage_sec == {} and m.cache_hits == []
    dumped = json.loads(m.model_dump_json())
    assert dumped["video_id"] == "v"
