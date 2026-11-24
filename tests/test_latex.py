"""LaTeX tooling: escaping, table fragments, charts, compile smoke."""

import pytest

from bookforge.schemas import ChartSpec, TableSpec
from bookforge.tools import latex
from bookforge.tools.workspace import Workspace


def test_tex_escape():
    assert latex.tex_escape("50% & $5 #a_b") == r"50\% \& \$5 \#a\_b"


def test_render_table_fragment(tmp_path):
    spec = TableSpec(
        title="t",
        caption="Optimizers compared & ranked",
        headers=["Name", "LR"],
        rows=[["Adam", "1e-3"], ["SGD", "1e-2"]],
    )
    frag = latex.render_table_fragment(spec, tmp_path / "table_1.tex", label="tab:ch01-1")
    body = frag.read_text(encoding="utf-8")
    assert r"\toprule" in body and r"\bottomrule" in body
    assert r"Optimizers compared \& ranked" in body
    assert "Adam & 1e-3" in body
    # tabularx keeps wide cells inside the text column; label makes \ref work
    assert r"\begin{tabularx}{\textwidth}" in body
    assert r"\label{tab:ch01-1}" in body
    assert r"\textbf{Name}" in body


def test_render_wide_table_uses_wrapping_columns(tmp_path):
    spec = TableSpec(
        title="t",
        caption="Five columns",
        headers=["A", "B", "C", "D", "E"],
        rows=[["1", "2", "3", "4", "5"]],
        note="Source: transcript",
    )
    frag = latex.render_table_fragment(spec, tmp_path / "table_2.tex")
    body = frag.read_text(encoding="utf-8")
    assert "LLLLL" in body  # all wrapping X-columns
    assert r"\small" in body  # wide tables shrink slightly
    assert "Source: transcript" in body


def test_render_chart_pdf(tmp_path):
    spec = ChartSpec(
        title="Accuracy",
        caption="Accuracy by epoch",
        chart_type="line",
        x_label="epoch",
        y_label="acc",
        categories=["1", "2", "3"],
        values=[0.5, 0.7, 0.9],
    )
    pdf = latex.render_chart(spec, tmp_path / "chart_1.pdf")
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"


def test_sanitize_fixes_htmlish_end_figure():
    messy = r"\begin{figure}[htbp]\centering\includegraphics{f.jpg}\end{figure>" + "\n"
    clean = latex.sanitize_chapter_tex(messy)
    assert r"\end{figure}" in clean
    assert r"\end{figure>" not in clean
    assert r"\textbackslash{}" in latex.sanitize_chapter_tex(r"use \backslash foo")


def test_sanitize_chapter_tex():
    messy = (
        "```latex\n\\documentclass{book}\n\\usepackage{x}\n"
        "\\begin{document}\n\\chapter{Hi}\n\\end{document}\n```"
    )
    clean = latex.sanitize_chapter_tex(messy)
    assert "documentclass" not in clean
    assert "usepackage" not in clean
    assert "begin{document}" not in clean
    assert clean.startswith("\\chapter{Hi}")


def test_render_pie_chart_with_long_labels(tmp_path):
    spec = ChartSpec(
        title="A very long chart title that must wrap instead of overflowing the axes",
        caption="Share",
        chart_type="pie",
        categories=[f"Extremely long category label number {i}" for i in range(7)],
        values=[10.0, 20.0, 15.0, 25.0, 10.0, 12.0, 8.0],
    )
    pdf = latex.render_chart(spec, tmp_path / "chart_pie.pdf")
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"


def test_render_pie_chart_falls_back_to_bar_on_bad_values(tmp_path):
    spec = ChartSpec(
        title="Bad pie",
        caption="c",
        chart_type="pie",
        categories=["a", "b"],
        values=[-1.0, 2.0],
    )
    pdf = latex.render_chart(spec, tmp_path / "chart_bad_pie.pdf")
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"


def test_sanitize_normalizes_float_placement_and_centering():
    messy = r"\begin{figure}[t]\includegraphics[width=0.8\textwidth]{f.jpg}\end{figure}"
    clean = latex.sanitize_chapter_tex(messy)
    assert r"\begin{figure}[!htb]" in clean
    assert clean.count(r"\centering") == 1
    # already-centered figures don't get a second \centering
    tidy = r"\begin{figure}[htbp]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=0.8\textwidth]{f.jpg}\end{figure}"
    assert latex.sanitize_chapter_tex(tidy).count(r"\centering") == 1


def test_sanitize_clamps_overwide_graphics():
    clean = latex.sanitize_chapter_tex(
        r"\includegraphics[width=1.2\textwidth]{f.jpg}"
        r"\includegraphics[width=0.85\linewidth]{g.jpg}"
        r"\includegraphics{h.jpg}"
    )
    assert r"width=1.2" not in clean
    assert r"width=0.9\linewidth" in clean
    assert r"width=0.85\linewidth]{g.jpg}" in clean
    assert r"\includegraphics[width=0.85\linewidth,keepaspectratio]{h.jpg}" in clean


def test_long_chapter_title_gets_truncated_header_mark():
    long_title = (
        "An Exhaustive and Deeply Detailed Exploration of Retrieval "
        "Augmented Generation Systems in Modern Production Environments"
    )
    clean = latex.sanitize_chapter_tex(f"\\chapter{{{long_title}}}\nBody text.")
    assert f"\\chapter{{{long_title}}}" in clean  # TOC keeps the full title
    assert "\\chaptermark{" in clean
    mark = clean.split("\\chaptermark{", 1)[1].split("}", 1)[0]
    assert len(mark) < len(long_title)
    # short titles are left untouched
    assert "\\chaptermark" not in latex.sanitize_chapter_tex("\\chapter{Intro}\nHi.")


def test_assemble_main_tex_footer_metadata():
    tex = latex.assemble_main_tex(
        "My Book",
        "BookForge",
        ["chapters/01_intro", "chapters/02_deep"],
        channel_name="Tech & AI Channel",
        chapter_links=[
            {"url": "https://youtu.be/abc123", "link_text": "youtu.be/abc123"},
            {},
        ],
    )
    assert r"\renewcommand{\BFchannelName}{Tech \& AI Channel}" in tex
    assert r"\href{https://youtu.be/abc123}" in tex
    # a chapter without a link resets the footer instead of inheriting one
    assert tex.count(r"\renewcommand{\BFvideoFooter}{}") == 1
    # backward-compatible: metadata is optional
    plain = latex.assemble_main_tex("B", "A", ["chapters/01_x"])
    assert r"\BFchannelName" not in plain.replace(r"\input{preamble.tex}", "")


def test_preamble_defines_layout_system():
    assert r"\usepackage{fancyhdr}" in latex.PREAMBLE
    assert r"\usepackage{tabularx}" in latex.PREAMBLE
    assert r"\usepackage{lmodern}" in latex.PREAMBLE
    assert r"\usepackage[T1]{fontenc}" in latex.PREAMBLE
    assert r"\usepackage[expansion=false]{microtype}" in latex.PREAMBLE
    assert r"\usepackage{microtype}" not in latex.PREAMBLE
    assert r"\newcommand{\BFchannelName}{}" in latex.PREAMBLE
    assert r"\newcommand{\BFvideoFooter}{}" in latex.PREAMBLE
    for env in ("notebox", "keyinsight", "formulabox"):
        assert f"\\newenvironment{{{env}}}" in latex.PREAMBLE


def test_find_referenced_files():
    body = (
        r"\includegraphics[width=0.8\textwidth]{chapters/01_x/figures/f.jpg}"
        r"\input{chapters/01_x/tables/table_1.tex}"
    )
    refs = latex.find_referenced_files(body)
    assert refs == ["chapters/01_x/figures/f.jpg", "chapters/01_x/tables/table_1.tex"]


def test_find_labels_and_refs():
    body = (
        r"\label{ch:01-x} see Table~\ref{tab:ch01-1}, \eqref{eq:1}, "
        r"\autoref{fig:a} and \ref{fig:a, fig:b}"
    )
    assert latex.find_labels(body) == {"ch:01-x"}
    assert latex.find_refs(body) == ["tab:ch01-1", "eq:1", "fig:a", "fig:a", "fig:b"]


def test_check_environment_balance():
    assert latex.check_environment_balance(r"\begin{itemize}\item a\end{itemize} {x}") == []
    problems = latex.check_environment_balance(r"\begin{figure}\begin{itemize}\item a\end{itemize}")
    assert problems and "figure" in problems[0]
    braces = latex.check_environment_balance(r"\caption{open")
    assert any("unclosed" in p for p in braces)
    stray = latex.check_environment_balance(r"text} more")
    assert any("stray" in p for p in stray)
    # escaped braces, comments and \verb are not counted
    assert latex.check_environment_balance(r"a \{ b % }}}" + "\n" + r"\verb|{|") == []


def test_extract_prose_drops_machine_text():
    body = (
        r"\section{Results 2024}"
        r"\includegraphics[width=0.85\linewidth]{figs/frame_01_5s.jpg}"
        r"\label{fig:99}\ref{tab:ch01-1}"
        r"\begin{equation}x = 31415\end{equation} inline $y=27182$ "
        r"\begin{tikzpicture}\node at (123,456) {Z};\end{tikzpicture}"
        " Accuracy rose to 92.5 percent over 1,000 runs."
    )
    prose = latex.extract_prose(body)
    numbers = latex.extract_numbers(prose)
    assert numbers == ["2024", "92.5", "1,000"]
    assert "31415" not in prose and "27182" not in prose and "123" not in prose


def test_number_normalization():
    assert latex.normalize_number("1,000") == "1000"
    assert latex.normalize_number("92.50") == "92.5"
    assert latex.normalize_number("0.001") == "0.001"
    assert latex.normalize_number("007") == "7"
    assert latex.normalize_number("1000.0") == "1000"
    assert latex.normalize_math(r"\theta = \theta - \eta \, \nabla L") == latex.normalize_math(
        r"\theta=\theta-\eta\nabla L"
    )
    assert latex.normalize_math(r"\left( a+b \right)") == "(a+b)"


def test_render_chart_is_thread_safe(tmp_path):
    """Charts render concurrently from worker threads (OO matplotlib API)."""
    from concurrent.futures import ThreadPoolExecutor

    specs = [
        ChartSpec(
            title=f"Chart {i}",
            caption="c",
            chart_type=["bar", "line", "pie"][i % 3],
            categories=["a", "b", "c"],
            values=[1.0 + i, 2.0, 3.0],
        )
        for i in range(9)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(lambda iv: latex.render_chart(iv[1], tmp_path / f"c{iv[0]}.pdf"), enumerate(specs))
        )
    assert len(paths) == 9
    for p in paths:
        assert p.exists() and p.read_bytes()[:5] == b"%PDF-"
    assert not list(tmp_path.glob("*.tmp.pdf"))  # atomic rename left no temp files


@pytest.mark.skipif(not latex.pdflatex_available(), reason="pdflatex missing")
def test_compile_minimal_book(tmp_path):
    """End-to-end: preamble + chapter with table, chart figure, callout boxes,
    an equation, footer metadata and a long chapter title -> PDF."""
    ws = Workspace(tmp_path, "demo")
    chapter_dir = ws.root / "chapters" / "01_intro"
    (chapter_dir / "tables").mkdir(parents=True)

    spec = TableSpec(title="t", caption="Cap", headers=["A", "B"], rows=[["1", "2"]])
    latex.render_table_fragment(
        spec, chapter_dir / "tables" / "table_1.tex", label="tab:ch01-1"
    )
    chart = ChartSpec(
        title="Throughput", caption="Cap", chart_type="bar",
        categories=["v1", "v2"], values=[10.0, 30.0],
    )
    latex.render_chart(chart, chapter_dir / "figures" / "chart_1.pdf")

    body = latex.sanitize_chapter_tex(
        "\\chapter{An Extremely Long Introductory Chapter Title That Would "
        "Overflow the Running Page Header Without Truncation}\\label{ch:01-intro}\n"
        "\\section{Key Data and Comparisons}\n"
        "See Table~\\ref{tab:ch01-1} and Figure~\\ref{fig:ch01-chart1}.\n"
        "\\input{chapters/01_intro/tables/table_1.tex}\n"
        "\\begin{figure}[htbp]\n"
        "\\includegraphics[width=0.85\\linewidth]{chapters/01_intro/figures/chart_1.pdf}\n"
        "\\caption{Cap}\\label{fig:ch01-chart1}\n"
        "\\end{figure}\n"
        "\\begin{notebox}\nShort supplementary note.\n\\end{notebox}\n"
        "\\begin{keyinsight}[Why it matters]\nThe key point.\n\\end{keyinsight}\n"
        "\\begin{formulabox}\n\\begin{equation}E = mc^2\\end{equation}\n\\end{formulabox}\n"
    )
    Workspace.write_text(chapter_dir / "chapter.tex", body)
    Workspace.write_text(ws.root / "preamble.tex", latex.PREAMBLE)
    main = Workspace.write_text(
        ws.root / "main.tex",
        latex.assemble_main_tex(
            "Demo Book",
            "Tests",
            ["chapters/01_intro"],
            channel_name="Demo Channel",
            chapter_links=[
                {"url": "https://youtu.be/demo1234", "link_text": "youtu.be/demo1234"}
            ],
        ),
    )
    ok, log = latex.compile_tex(main, ws.book_dir, passes=1, cwd=ws.root)
    assert ok, log
    assert (ws.book_dir / "main.pdf").exists()


def test_ensure_short_chapter_marks_is_idempotent():
    from bookforge.tools.latex import ensure_short_chapter_marks

    body = "\\chapter{" + "A very long chapter title about many things " * 3 + "}\nText."
    once = ensure_short_chapter_marks(body)
    twice = ensure_short_chapter_marks(once)
    assert once.count("\\chaptermark") == 1
    assert twice == once
