"""Deterministic chapter verification: the machine gate must catch every
mechanical defect class and pass a well-formed chapter."""

from __future__ import annotations

import pytest

from bookforge.config import Settings
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    FigureAsset,
    TableAsset,
    VideoRecord,
)
from bookforge.tools import qa_checks
from bookforge.tools.workspace import Workspace

TRANSCRIPT = """[00:00:05] welcome, today we compare Adam and SGD optimizers
[00:01:10] Adam reached 92.5 percent accuracy after 30 epochs
[00:02:40] SGD got 88 percent, and the learning rate was 0.001
[00:04:00] the update rule is theta = theta - eta * gradient
"""


def analysis() -> ChapterAnalysis:
    return ChapterAnalysis(
        chapter_title="Optimizers Compared",
        summary="Adam versus SGD on a small benchmark.",
        learning_objectives=["The reader can explain Adam", "The reader can tune SGD"],
        concepts=[
            {
                "heading": "Adam Optimizer",
                "explanation": "Adam adapts the learning rate per parameter.",
                "key_points": ["adaptive", "momentum"],
                "equations": [r"\theta = \theta - \eta \nabla L"],
                "evidence_timestamps": ["00:01:10"],
            },
            {
                "heading": "Stochastic Gradient Descent",
                "explanation": "SGD uses a fixed learning rate.",
                "key_points": ["simple"],
                "evidence_timestamps": ["00:02:40"],
            },
        ],
        tables=[
            {
                "title": "t",
                "caption": "Accuracy by optimizer",
                "headers": ["Optimizer", "Accuracy"],
                "rows": [["Adam", "92.5%"], ["SGD", "88%"]],
            }
        ],
        charts=[
            {
                "title": "Accuracy",
                "caption": "Accuracy chart",
                "categories": ["Adam", "SGD"],
                "values": [92.5, 88.0],
            }
        ],
        diagrams=[
            {
                "title": "Training loop",
                "caption": "Loop",
                "elements": ["Forward", "Loss", "Update"],
                "relationships": ["Forward -> Loss", "Loss -> Update"],
            }
        ],
        glossary=[{"term": "epoch", "definition": "one pass over the data"}],
        exercises=[{"question": "What is Adam?", "answer": "An adaptive optimizer."}],
    )


def manifest() -> AssetsManifest:
    return AssetsManifest(
        chapter_slug="01_optimizers",
        chapter_dir="chapters/01_optimizers",
        figures=[
            FigureAsset(
                filename="chapters/01_optimizers/figures/chart_1.pdf",
                kind="chart",
                caption="Accuracy chart",
                label="fig:ch01-chart1",
            ),
            FigureAsset(
                filename="chapters/01_optimizers/figures/frame_01_5s.jpg",
                kind="frame",
                caption="frame",
                label="fig:ch01-frame1",
            ),
        ],
        tables=[
            TableAsset(
                tex_file="chapters/01_optimizers/tables/table_1.tex",
                caption="Accuracy by optimizer",
                label="tab:ch01-1",
            )
        ],
    )


def _filler(n: int) -> str:
    return " ".join(["optimizers adapt gradients carefully during training"] * n)


def good_chapter() -> str:
    return "\n".join(
        [
            r"\chapter{Optimizers Compared}\label{ch:01-optimizers}",
            r"\section*{Overview}",
            "Adam versus SGD on a small benchmark. " + _filler(30),
            r"\section*{Learning Objectives}",
            r"\begin{itemize}\item The reader can explain Adam \item The reader can tune SGD\end{itemize}",
            r"\section{Adam Optimizer}",
            "Adam reached 92.5 percent accuracy after 30 epochs. " + _filler(30),
            r"\begin{equation}\theta = \theta - \eta \nabla L\end{equation}",
            r"\begin{figure}[!htb]\centering",
            r"\includegraphics[width=0.85\linewidth]{chapters/01_optimizers/figures/frame_01_5s.jpg}",
            r"\caption{Frame}\label{fig:ch01-frame1}\end{figure}",
            r"\section{Stochastic Gradient Descent}",
            "SGD got 88 percent with a learning rate of 0.001. " + _filler(30),
            r"\section*{Key Data and Comparisons}",
            r"Table~\ref{tab:ch01-1} and Figure~\ref{fig:ch01-chart1} compare the optimizers.",
            r"\input{chapters/01_optimizers/tables/table_1.tex}",
            r"\begin{figure}[!htb]\centering",
            r"\includegraphics[width=0.85\linewidth]{chapters/01_optimizers/figures/chart_1.pdf}",
            r"\caption{Accuracy chart}\label{fig:ch01-chart1}\end{figure}",
            r"\section*{Conceptual Models}",
            r"\begin{figure}[!htb]\centering\begin{tikzpicture}",
            r"\node[modernbox] (a) {Forward}; \node[modernbox, right=1.8cm of a] (b) {Loss};",
            r"\draw[flowarrow] (a) -- (b);",
            r"\end{tikzpicture}\caption{Loop}\label{fig:ch01-diag1}\end{figure}",
            r"\section*{Key Takeaways}",
            r"\begin{itemize}\item Adam adapts.\end{itemize}",
            r"\section*{Glossary}",
            r"\begin{description}\item[epoch] one pass over the data\end{description}",
            r"\section*{Exercises}",
            r"\begin{enumerate}\item What is Adam?\end{enumerate}",
            r"\section*{Solutions}",
            r"\begin{enumerate}\item An adaptive optimizer.\end{enumerate}",
        ]
    )


@pytest.fixture
def ws(tmp_path) -> Workspace:
    ws = Workspace(tmp_path, "demo")
    ch = ws.root / "chapters" / "01_optimizers"
    (ch / "figures").mkdir(parents=True)
    (ch / "tables").mkdir(parents=True)
    (ch / "figures" / "chart_1.pdf").write_bytes(b"%PDF-1.4")
    (ch / "figures" / "frame_01_5s.jpg").write_bytes(b"\xff\xd8\xff")
    (ch / "tables" / "table_1.tex").write_text(
        r"\begin{table}\caption{x}\label{tab:ch01-1}\end{table}", encoding="utf-8"
    )
    return ws


def codes(defects) -> set[str]:
    return {d.code for d in defects}


def test_good_chapter_passes(ws):
    defects, metrics = qa_checks.run_static_checks(
        good_chapter(), analysis(), manifest(), TRANSCRIPT, ws.root
    )
    assert [d for d in defects if d.severity != "minor"] == [], [d.message for d in defects]
    assert metrics["prose_words"] > qa_checks.MIN_PROSE_WORDS
    assert metrics["unsupported_numbers"] == 0
    assert metrics["tables_used"] == 1 and metrics["figures_used"] == 2


def test_missing_sections_and_chapter_command(ws):
    tex = good_chapter().replace(r"\section*{Glossary}", r"\section*{Terms}")
    tex = tex.replace(r"\chapter{Optimizers Compared}\label{ch:01-optimizers}", "")
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert "missing_chapter_command" in codes(defects)
    assert any(d.code == "missing_section" and d.location == "Glossary" for d in defects)


def test_unbalanced_environment_is_critical(ws):
    tex = good_chapter().replace(r"\end{itemize}", "", 1)
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    bad = [d for d in defects if d.code == "unbalanced_latex"]
    assert bad and bad[0].severity == "critical"
    assert "itemize" in bad[0].message


def test_missing_and_unknown_files_are_critical(ws):
    tex = good_chapter().replace(
        "chapters/01_optimizers/figures/frame_01_5s.jpg",
        "chapters/01_optimizers/figures/invented.jpg",
    )
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert {"missing_file", "unknown_asset"} <= codes(defects)
    assert all(d.severity == "critical" for d in defects if d.code in ("missing_file", "unknown_asset"))


def test_unused_table_and_chart_are_major(ws):
    tex = good_chapter()
    tex = tex.replace(r"\input{chapters/01_optimizers/tables/table_1.tex}", "")
    tex = tex.replace(
        r"\includegraphics[width=0.85\linewidth]{chapters/01_optimizers/figures/chart_1.pdf}", ""
    )
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert {"unused_table", "unused_chart"} <= codes(defects)


def test_undefined_and_duplicate_labels(ws):
    tex = good_chapter().replace(r"Figure~\ref{fig:ch01-chart1}", r"Figure~\ref{fig:nope}")
    tex += "\n" + r"\label{tab:ch01-1}"  # duplicates the fragment's label
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert any(d.code == "undefined_ref" and d.location == "fig:nope" for d in defects)
    assert any(d.code == "duplicate_label" and d.location == "tab:ch01-1" for d in defects)


def test_table_labels_from_fragments_count_as_defined(ws):
    # \ref{tab:ch01-1} is only defined inside the fragment -> not undefined.
    defects, _ = qa_checks.run_static_checks(
        good_chapter(), analysis(), manifest(), TRANSCRIPT, ws.root
    )
    assert "undefined_ref" not in codes(defects)


def test_unsupported_numbers_are_flagged_and_supported_ones_are_not(ws):
    tex = good_chapter().replace(
        "SGD got 88 percent", "SGD got 88 percent on 4,096 samples and 73.4 percent later"
    )
    defects, metrics = qa_checks.run_static_checks(
        tex, analysis(), manifest(), TRANSCRIPT, ws.root
    )
    bad = [d for d in defects if d.code == "unsupported_number"]
    assert bad and bad[0].severity == "major"
    assert "4,096" in bad[0].message and "73.4" in bad[0].message
    assert "92.5" not in bad[0].message and "88" not in bad[0].message
    assert metrics["unsupported_numbers"] == 2
    # small structural integers are never audited
    _, m2 = qa_checks.run_static_checks(
        good_chapter() + " Step 3 of 5 uses 12 layers.", analysis(), manifest(), TRANSCRIPT, ws.root
    )
    assert m2["unsupported_numbers"] == 0


def test_strict_numbers_toggle_demotes_severity(ws):
    tex = good_chapter().replace("SGD got 88 percent", "SGD got 88 percent on 4,096 samples")
    defects, _ = qa_checks.run_static_checks(
        tex, analysis(), manifest(), TRANSCRIPT, ws.root, strict_numbers=False
    )
    assert [d.severity for d in defects if d.code == "unsupported_number"] == ["minor"]


def test_numbers_inside_paths_labels_math_and_tikz_are_ignored(ws):
    tex = good_chapter() + "\n" + "\n".join(
        [
            r"\includegraphics[width=0.85\linewidth]{chapters/01_optimizers/figures/frame_01_5s.jpg}",
            r"\label{fig:ch01-frame1-9999}",
            r"\begin{equation} x = 31415 \end{equation}",
            r"Inline $y = 27182$ math.",
            r"\begin{tikzpicture}\node at (123456, 0) {Z};\end{tikzpicture}",
        ]
    )
    _, metrics = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert metrics["unsupported_numbers"] == 0


def test_missing_equation_concept_and_diagram(ws):
    tex = good_chapter()
    tex = tex.replace(r"\begin{equation}\theta = \theta - \eta \nabla L\end{equation}", "")
    tex = tex.replace(r"\section{Stochastic Gradient Descent}", r"\section{Another Topic}")
    tex = tex.replace("SGD got 88 percent with a learning rate of 0.001. ", "")
    tex = tex.replace(r"\begin{tikzpicture}", "").replace(r"\end{tikzpicture}", "")
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert {"missing_equation", "missing_concept", "missing_diagram"} <= codes(defects)
    concept = next(d for d in defects if d.code == "missing_concept")
    assert concept.location == "Stochastic Gradient Descent"


def test_equation_match_is_whitespace_insensitive(ws):
    tex = good_chapter().replace(
        r"\begin{equation}\theta = \theta - \eta \nabla L\end{equation}",
        r"\begin{equation}" "\n" r"  \theta=\theta-\eta\,\nabla L" "\n" r"\end{equation}",
    )
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert "missing_equation" not in codes(defects)


def test_bogus_evidence_timestamp_is_minor_only(ws):
    a = analysis()
    a.concepts[0].evidence_timestamps = ["00:59:59"]
    defects, _ = qa_checks.run_static_checks(good_chapter(), a, manifest(), TRANSCRIPT, ws.root)
    ev = [d for d in defects if d.code == "evidence_timestamp_unknown"]
    assert ev and ev[0].severity == "minor"


def test_short_chapter_is_major(ws):
    tex = good_chapter()
    for _ in range(3):
        tex = tex.replace(_filler(30), "short.", 1)
    defects, _ = qa_checks.run_static_checks(tex, analysis(), manifest(), TRANSCRIPT, ws.root)
    assert "chapter_too_short" in codes(defects)


def test_empty_chapter():
    defects, _ = qa_checks.run_static_checks("", None, None, "", None)
    assert codes(defects) == {"empty_chapter"}


def test_run_machine_checks_report_and_early_compile_skip(ws):
    settings = Settings(_env_file=None, compile_latex=True, workspace_root=ws.root.parent)
    record = VideoRecord(video_id="v1", title="Optimizers", url="u", chapter_number=1)
    bad = good_chapter().replace(
        "chapters/01_optimizers/figures/frame_01_5s.jpg", "chapters/01_optimizers/figures/no.jpg"
    )
    report = qa_checks.run_machine_checks(bad, analysis(), manifest(), TRANSCRIPT, ws, record, settings, 2)
    assert report.passed is False
    assert report.compile_ok is False  # critical static defects -> compile not attempted
    assert "pdflatex" not in report.checks_run
    assert report.iteration == 2
    assert report.blocking and all(d.severity in ("critical", "major") for d in report.blocking)

    settings.compile_latex = False
    report = qa_checks.run_machine_checks(
        good_chapter(), analysis(), manifest(), TRANSCRIPT, ws, record, settings, 1
    )
    assert report.passed is True and report.compile_ok is None
    assert report.metrics["blocking_defects"] == 0


@pytest.mark.skipif(not qa_checks.latex_tools.pdflatex_available(), reason="pdflatex missing")
def test_run_machine_checks_compiles_with_pdflatex(ws):
    """The real compile gate: a clean chapter with a real table fragment and
    chart passes; a LaTeX error becomes a quotable CRITICAL compile_error."""
    from bookforge.schemas import ChartSpec, TableSpec

    ch = ws.root / "chapters" / "01_optimizers"
    qa_checks.latex_tools.render_table_fragment(
        TableSpec(title="t", caption="Accuracy by optimizer", headers=["Optimizer", "Accuracy"],
                  rows=[["Adam", "92.5%"], ["SGD", "88%"]]),
        ch / "tables" / "table_1.tex", label="tab:ch01-1",
    )
    (ch / "figures" / "chart_1.pdf").unlink()
    qa_checks.latex_tools.render_chart(
        ChartSpec(title="Accuracy", caption="c", categories=["Adam", "SGD"], values=[92.5, 88.0]),
        ch / "figures" / "chart_1.pdf",
    )
    from PIL import Image

    Image.new("RGB", (64, 36), (40, 60, 200)).save(ch / "figures" / "frame_01_5s.jpg", "JPEG")

    settings = Settings(_env_file=None, compile_latex=True, workspace_root=ws.root.parent)
    record = VideoRecord(video_id="v1", title="Optimizers", url="u", chapter_number=1)
    report = qa_checks.run_machine_checks(
        good_chapter(), analysis(), manifest(), TRANSCRIPT, ws, record, settings, 1
    )
    assert report.compile_ok is True, [d.message for d in report.defects]
    assert report.passed and "pdflatex" in report.checks_run

    broken = good_chapter().replace(r"\section*{Overview}", r"\section*{Overview}\undefinedmacro")
    report = qa_checks.run_machine_checks(
        broken, analysis(), manifest(), TRANSCRIPT, ws, record, settings, 2
    )
    assert report.compile_ok is False and not report.passed
    err = next(d for d in report.defects if d.code == "compile_error")
    assert err.severity == "critical" and "undefinedmacro" in err.message


def test_format_defects_is_numbered_and_actionable():
    report_text = qa_checks.format_defects(
        [
            qa_checks.QADefect(severity="critical", code="missing_file", message="x missing", fix="remove it", location="x"),
            qa_checks.QADefect(severity="minor", code="style", message="meh"),
        ],
        header="Header.",
    )
    lines = report_text.splitlines()
    assert lines[0] == "Header."
    assert lines[1].startswith("1. CRITICAL missing_file [x]: x missing. FIX: remove it")
    assert lines[2].startswith("2. MINOR style: meh.")
    assert qa_checks.format_defects([], header="none") == "none"
