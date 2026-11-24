"""BookForge Studio: diagrams, grounded charts, illustrations, frames,
branding, cover, parts and CLI flags. Offline: models/HTTP are mocked."""

from __future__ import annotations

import asyncio
import base64
import io
import shutil
import zipfile
from pathlib import Path

import httpx
import pytest
from PIL import Image, ImageDraw

from bookforge.config import Settings
from bookforge.prompts import analyst_instruction
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    ChartSpec,
    ConceptNote,
    DiagramSpec,
    FigureAsset,
    IllustrationSpec,
    VideoRecord,
)
from bookforge.tools import branding as brand
from bookforge.tools import latex
from bookforge.tools import visuals as v
from bookforge.tools.workspace import Workspace

HAS_PDFLATEX = shutil.which("pdflatex") is not None
needs_tex = pytest.mark.skipif(not HAS_PDFLATEX, reason="pdflatex not installed")

TRANSCRIPT = "\n".join(
    [
        "[00:00:05] welcome to the course",
        "[00:03:10] GPT-4 scored 86.4 on MMLU while the small model got 70",
        "[00:03:20] and the open model reached 79",
        "[00:09:00] now let us build the agent loop",
    ]
)


def _png_bytes(size=(64, 48), color="white", draw_text=False) -> bytes:
    img = Image.new("RGB", size, color)
    if draw_text:
        d = ImageDraw.Draw(img)
        for y in range(4, size[1], 6):
            d.line((4, y, size[0] - 4, y), fill="black", width=1)
            for x in range(4, size[0], 9):
                d.rectangle((x, y - 2, x + 3, y + 1), outline="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- diagrams


def test_build_graph_drops_unusable_and_dedupes():
    assert v.build_graph(DiagramSpec(title="t", caption="c", elements=["only one"])) is None
    g = v.build_graph(
        DiagramSpec(title="t", caption="c", elements=["A step", "a step", "B step", "C step"])
    )
    assert g.nodes == ["A step", "B step", "C step"]
    assert g.edges == [(0, 1, ""), (1, 2, "")]  # stated order becomes the flow


def test_process_keeps_order_and_labels_from_partial_relationships():
    g = v.build_graph(
        DiagramSpec(
            title="t", caption="c", diagram_type="process",
            elements=["Query", "Retrieve", "Rerank", "Answer"],
            relationships=["query -> retrieve : embed", "garbage line"],
        )
    )
    assert g.edges == [(0, 1, "embed"), (1, 2, ""), (2, 3, "")]


def test_hierarchy_and_mindmap_attach_orphans_to_root():
    for kind in ("hierarchy", "mindmap"):
        g = v.build_graph(
            DiagramSpec(title="t", caption="c", diagram_type=kind,
                        elements=["Root", "Left", "Right"], relationships=["Root -> Left"])
        )
        assert {(a, b) for a, b, _ in g.edges} == {(0, 1), (0, 2)}


def test_layouts_are_deterministic_and_non_overlapping():
    for kind, n in [("process", 7), ("flowchart", 3), ("mindmap", 6), ("cycle", 5), ("hierarchy", 6)]:
        spec = DiagramSpec(title="t", caption="c", diagram_type=kind,
                           elements=[f"Node {i}" for i in range(n)])
        g = v.build_graph(spec)
        pos = v.layout(g)
        assert pos == v.layout(g)
        assert len(set(pos.values())) == n, kind  # no two nodes on the same spot


def test_long_chain_wraps_into_rows():
    g = v.build_graph(DiagramSpec(title="t", caption="c", diagram_type="process",
                                  elements=[f"Step {i}" for i in range(7)]))
    ys = {y for _, y in v.layout(g).values()}
    assert len(ys) == 3  # 3 + 3 + 1 snake rows instead of one very wide line


def test_tikz_escapes_node_text():
    g = v.build_graph(DiagramSpec(title="t", caption="c", elements=["R&D 50%", "Cost_$"]))
    tikz = v.diagram_to_tikz(g)
    assert r"R\&D 50\%" in tikz and r"Cost\_\$" in tikz
    assert tikz.startswith(r"\begin{tikzpicture}") and tikz.endswith(r"\end{tikzpicture}")


@needs_tex
def test_render_diagram_compiles_every_kind(tmp_path):
    for kind in ("flowchart", "process", "sequence", "hierarchy", "mindmap", "cycle"):
        spec = DiagramSpec(title=kind, caption="c", diagram_type=kind,
                           elements=["Plan & think", "Act", "Observe", "Reflect"],
                           relationships=["Plan & think -> Act : call tool"])
        out = v.render_diagram(spec, tmp_path / f"{kind}.pdf")
        assert out is not None and out.read_bytes()[:4] == b"%PDF"


def test_render_diagram_failure_is_dropped_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "compile_standalone_tikz", lambda *a, **k: False)
    monkeypatch.setattr(v, "compile_dot", lambda *a, **k: False)
    spec = DiagramSpec(title="t", caption="c", elements=["A", "B", "C"])
    assert v.render_diagram(spec, tmp_path / "d.pdf") is None
    def boom(*a, **k):
        raise RuntimeError("renderer exploded")
    monkeypatch.setattr(v, "diagram_to_tikz", boom)
    assert v.render_diagram(spec, tmp_path / "d2.pdf") is None


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz not installed")
def test_graphviz_engine(tmp_path):
    spec = DiagramSpec(title="t", caption="c", elements=['Say "hi"', "B", "C"])
    out = v.render_diagram(spec, tmp_path / "g.pdf", engine="graphviz")
    assert out is not None and out.read_bytes()[:4] == b"%PDF"


# ---------------------------------------------------------------- charts


def test_chart_values_must_be_stated_in_transcript():
    ok_spec = ChartSpec(title="MMLU", caption="Scores", categories=["GPT-4", "small", "open"],
                        values=[86.4, 70, 79], source_timestamp="00:03:10")
    ok, ts, reason = v.verify_chart_against_transcript(ok_spec, TRANSCRIPT)
    assert ok and ts == "00:03:10" and reason == ""

    invented = ok_spec.model_copy(update={"values": [86.4, 70, 81]})
    ok, _, reason = v.verify_chart_against_transcript(invented, TRANSCRIPT)
    assert not ok and "81" in reason

    one = ok_spec.model_copy(update={"values": [86.4], "categories": ["GPT-4"]})
    assert not v.verify_chart_against_transcript(one, TRANSCRIPT)[0]


def test_chart_timestamp_falls_back_to_where_numbers_occur():
    spec = ChartSpec(title="t", caption="c", categories=["a", "b"], values=[70, 79],
                     source_timestamp="00:59:00")
    ok, ts, _ = v.verify_chart_against_transcript(spec, TRANSCRIPT)
    assert ok and ts == "00:03:10"
    assert v.chart_caption(spec, ts) == "c. Source: values as stated in the video at 00:03:10."


# ---------------------------------------------------------------- illustrations


def _run(coro):
    return asyncio.run(coro)


def test_generate_image_b64_saved_as_png(tmp_path):
    payload = base64.b64encode(_png_bytes()).decode()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.read()
        return httpx.Response(200, json={"data": [{"b64_json": payload}]})

    out = _run(v.generate_image("a robot", tmp_path / "i.png", api_key="k", api_base="https://x/api",
                                transport=httpx.MockTransport(handler)))
    assert out is not None and Image.open(out).format == "PNG"
    assert seen["auth"] == "Bearer k" and b"euri-image-fast" in seen["body"]


def test_generate_image_url_download_never_sends_the_key(tmp_path):
    png = _png_bytes()
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("authorization")))
        if request.url.host == "api.example":
            return httpx.Response(200, json={"data": [{"url": "https://cdn.example/img.png"}]})
        return httpx.Response(200, content=png)

    out = _run(v.generate_image("p", tmp_path / "u.png", api_key="secret", api_base="https://api.example/v1",
                                transport=httpx.MockTransport(handler)))
    assert out is not None
    assert seen == [("api.example", "Bearer secret"), ("cdn.example", None)]


def test_generate_image_retries_when_busy(tmp_path):
    payload = base64.b64encode(_png_bytes()).decode()
    replies = iter([{"error": {"code": "euri_no_capacity"}}, {"data": [{"b64_json": payload}]}])
    t = httpx.MockTransport(lambda r: httpx.Response(200, json=next(replies)))
    out = _run(v.generate_image("p", tmp_path / "r.png", api_key="k", api_base="https://x",
                                transport=t, retry_delays=(0.0,)))
    assert out is not None


def test_generate_image_failures_return_none(tmp_path):
    bad = httpx.MockTransport(lambda r: httpx.Response(500, json={}))
    assert _run(v.generate_image("p", tmp_path / "a.png", api_key="k", api_base="https://x",
                                 transport=bad, retry_delays=(0.0,))) is None
    denied = httpx.MockTransport(lambda r: httpx.Response(400, json={}))
    assert _run(v.generate_image("p", tmp_path / "a2.png", api_key="k", api_base="https://x",
                                 transport=denied)) is None
    junk = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"nope").decode()}]})
    )
    assert _run(v.generate_image("p", tmp_path / "b.png", api_key="k", api_base="https://x",
                                 transport=junk)) is None
    assert _run(v.generate_image("p", tmp_path / "c.png", api_key="", api_base="https://x")) is None
    assert not list(tmp_path.glob("*.png"))


def test_illustration_prompt_forbids_text():
    p = v.illustration_prompt(IllustrationSpec(section="s", prompt="A librarian robot", caption="c"))
    assert p.startswith("A librarian robot.") and "no text" in p


# ---------------------------------------------------------------- frames


def test_informative_frames_beat_flat_ones(tmp_path):
    flat = tmp_path / "flat.png"
    flat.write_bytes(_png_bytes((320, 180), "#777777"))
    slide = tmp_path / "slide.png"
    slide.write_bytes(_png_bytes((320, 180), "white", draw_text=True))
    assert v.frame_information_score(slide) > v.frame_information_score(flat)
    assert v.frame_information_score(tmp_path / "missing.png") == 0.0


def test_pick_informative_one_per_bucket_in_time_order():
    items = [(t, score, f"f{t}") for t, score in [(0, 1), (5, 9), (10, 2), (15, 3), (20, 8), (25, 1)]]
    assert v.pick_informative(items, 3) == ["f5", "f15", "f20"]
    assert v.pick_informative(items, 10) == [f"f{t}" for t in (0, 5, 10, 15, 20, 25)]
    assert v.pick_informative(items, 0) == []


# ---------------------------------------------------------------- settings / prompt


def test_visual_caps_by_level_and_overrides():
    assert Settings(_env_file=None, visuals="light").visual_caps["illustrations"] == 0
    rich = Settings(_env_file=None, visuals="rich")
    assert rich.visual_caps["diagrams"] == 4 and rich.visual_caps["illustrations"] == 2
    assert Settings(_env_file=None, visuals="rich", max_frames_per_chapter=9).visual_caps["frames"] == 9
    assert Settings(_env_file=None, visuals="rich", illustrations_enabled=False).visual_caps["illustrations"] == 0


def test_visuals_env_var(monkeypatch):
    monkeypatch.setenv("BOOKFORGE_VISUALS", "rich")
    monkeypatch.setenv("BOOKFORGE_HEADER_LEFT", "My Header")
    s = Settings(_env_file=None)
    assert s.visuals == "rich" and s.header_left == "My Header"


def test_analyst_instruction_fills_visual_budget():
    text = analyst_instruction({"charts": 6, "diagrams": 4, "illustrations": 2})
    assert "@@" not in text
    assert "up to 4 visual explanations" in text and "up to 2 AI illustrations" in text


# ---------------------------------------------------------------- render_assets


def _analysis(**kw) -> ChapterAnalysis:
    base = dict(
        chapter_title="Agents",
        summary="s",
        learning_objectives=["a", "b"],
        concepts=[ConceptNote(heading="Agent Loop Basics", explanation="e")],
    )
    base.update(kw)
    return ChapterAnalysis(**base)


@needs_tex
def test_render_assets_drops_invented_chart_keeps_grounded_and_diagram(tmp_path):
    settings = Settings(_env_file=None, workspace_root=tmp_path, visuals="rich")
    ws = Workspace(tmp_path, "demo")
    rec = VideoRecord(video_id="vid1", title="T", url="u", chapter_number=1)
    analysis = _analysis(
        charts=[
            ChartSpec(title="Grounded", caption="Scores", categories=["a", "b"], values=[70, 79]),
            ChartSpec(title="Invented", caption="Fake", categories=["a", "b"], values=[12345, 999]),
        ],
        diagrams=[DiagramSpec(title="Agent loop basics", caption="The loop",
                              diagram_type="cycle", elements=["Think", "Act", "Observe"])],
    )
    from bookforge.agents.assets import render_assets

    manifest, n_frames = render_assets(analysis, rec, ws, settings, TRANSCRIPT)
    kinds = sorted(f.kind for f in manifest.figures)
    assert kinds == ["chart", "diagram"] and n_frames == 0
    chart = next(f for f in manifest.figures if f.kind == "chart")
    assert "Source: values as stated in the video at 00:03:10" in chart.caption
    diagram = next(f for f in manifest.figures if f.kind == "diagram")
    assert diagram.section == "Agent Loop Basics"
    assert (ws.root / diagram.filename).exists()


# ---------------------------------------------------------------- figure insertion


CHAPTER = "\n".join([
    r"\chapter{Agents}\label{ch:01-agents}",
    r"\section*{Overview}", "Intro.",
    r"\section{Agent Loop Basics}", "Body text.",
    r"\section*{Key Takeaways}", r"\begin{itemize}\item x\end{itemize}",
    r"\section*{Glossary}", "g",
])


def _manifest(*figs) -> AssetsManifest:
    return AssetsManifest(chapter_slug="01", chapter_dir="chapters/01", figures=list(figs))


def test_ensure_figures_included_places_missing_visuals():
    m = _manifest(
        FigureAsset(filename="chapters/01/figures/diagram_1.pdf", kind="diagram", caption="Loop & more", label="fig:d1"),
        FigureAsset(filename="chapters/01/figures/illustration_1.png", kind="illustration",
                    caption="Robot", label="fig:i1", section="Agent Loop Basics"),
        FigureAsset(filename="chapters/01/figures/frame_01_5s.jpg", kind="frame", caption="f", label="fig:f1"),
    )
    tex, added = latex.ensure_figures_included(CHAPTER, m)
    assert added == ["fig:d1", "fig:i1"]
    assert "frame_01_5s.jpg" not in tex  # frames stay the writer's choice
    assert tex.index(r"\section*{Conceptual Models}") < tex.index(r"\section*{Key Takeaways}")
    assert tex.index("diagram_1.pdf") > tex.index(r"\section*{Conceptual Models}")
    assert tex.index(r"\section{Agent Loop Basics}") < tex.index("illustration_1.png") < tex.index(r"\section*{Key Takeaways}")
    assert r"Loop \& more" in tex
    again, added2 = latex.ensure_figures_included(tex, m)
    assert again == tex and added2 == []
    assert latex.check_environment_balance(tex) == []


# ---------------------------------------------------------------- branding


def test_prepare_logo_png_pdf_bad_and_missing(tmp_path):
    png = tmp_path / "logo.png"
    png.write_bytes(_png_bytes())
    assert brand.prepare_logo(png, tmp_path / "b").name == "logo_logo.png"
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert brand.prepare_logo(bad, tmp_path / "b") is None
    assert brand.prepare_logo(tmp_path / "nope.png", tmp_path / "b") is None
    assert brand.prepare_logo(tmp_path / "x.gif", tmp_path / "b") is None


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="rsvg-convert not installed")
def test_prepare_logo_svg_converts_to_pdf(tmp_path):
    svg = tmp_path / "l.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
                   '<rect width="40" height="20" fill="#2563eb"/></svg>')
    out = brand.prepare_logo(svg, tmp_path / "b")
    assert out.suffix == ".pdf" and out.read_bytes()[:4] == b"%PDF"


def test_resolve_branding_positions(tmp_path):
    png = tmp_path / "logo.png"
    png.write_bytes(_png_bytes())
    s = Settings(_env_file=None, logo=str(png), logo_positions="title,footer", header_left="H & L")
    b = brand.resolve_branding(s, tmp_path)
    assert set(b.logos) == {"title", "footer"}
    tex = brand.header_footer_tex(b)
    assert r"H \& L" in tex and b.logos["footer"] in tex and r"\thepage" in tex
    assert brand.header_footer_tex(brand.Branding()) == ""


def test_assemble_main_tex_cover_title_lof_and_parts():
    tex = latex.assemble_main_tex(
        "Book", "Me", ["chapters/01_a", "chapters/02_b", "chapters/03_c"],
        branding_tex="%BRANDING\n",
        title_page=brand.title_page_tex("Book", "Sub", "Me", logo_rel="branding/logo.png"),
        cover_page=brand.cover_page_tex("branding/cover.png", "Book", "Sub"),
        list_of_figures=True,
        part_titles=["Basics", "Basics", "Advanced"],
    )
    assert tex.index("branding/cover.png") < tex.index(r"\begin{titlepage}\n\centering".replace("\\n", "\n"))
    assert r"\maketitle" not in tex and r"\listoffigures" in tex and "%BRANDING" in tex
    assert tex.count(r"\part{") == 2
    assert tex.index(r"\part{Basics}") < tex.index("01_a") < tex.index(r"\part{Advanced}") < tex.index("03_c")
    single = latex.assemble_main_tex("B", "A", ["c/1", "c/2"], part_titles=["Only", "Only"])
    assert r"\part{" not in single and r"\maketitle" in single


def test_write_source_zip(tmp_path):
    (tmp_path / "main.tex").write_text("m")
    (tmp_path / "preamble.tex").write_text("p")
    ch = tmp_path / "chapters" / "01_a"
    (ch / "figures").mkdir(parents=True)
    (ch / "chapter.tex").write_text("c")
    (ch / "figures" / "d.pdf").write_bytes(b"%PDF")
    (ch / "metrics.json").write_text("{}")
    (tmp_path / "branding").mkdir()
    (tmp_path / "branding" / "logo.png").write_bytes(b"x")
    out = brand.write_source_zip(tmp_path, tmp_path / "book" / "latex_source.zip", ["chapters/01_a"])
    names = set(zipfile.ZipFile(out).namelist())
    assert {"main.tex", "preamble.tex", "chapters/01_a/chapter.tex",
            "chapters/01_a/figures/d.pdf", "branding/logo.png"} <= names
    assert "chapters/01_a/metrics.json" not in names


@needs_tex
def test_branded_book_compiles_end_to_end(tmp_path):
    """Cover + logo title page + custom header/footer + LoF + parts + a
    rendered diagram all compile together with pdflatex."""
    root = tmp_path
    (root / "branding").mkdir()
    Image.new("RGB", (300, 300), "#336699").save(root / "branding" / "cover.png")
    Image.new("RGB", (240, 60), "#2563eb").save(root / "branding" / "logo.png")
    ch = root / "chapters" / "01_a"
    spec = DiagramSpec(title="t", caption="c", diagram_type="process", elements=["A", "B", "C"])
    diag = v.render_diagram(spec, ch / "figures" / "diagram_1.pdf")
    assert diag is not None
    m = _manifest(FigureAsset(filename="chapters/01_a/figures/diagram_1.pdf", kind="diagram",
                              caption="Process", label="fig:ch01-diagram1"))
    body, _ = latex.ensure_figures_included(CHAPTER, m)
    (ch / "chapter.tex").write_text(body)
    b = brand.Branding(header_left="Studio Header", header_right="Right",
                       footer_text="Footer & Co", logos={"header": "branding/logo.png",
                                                         "title": "branding/logo.png",
                                                         "footer": "branding/logo.png"})
    (root / "preamble.tex").write_text(latex.PREAMBLE)
    main = root / "main.tex"
    main.write_text(latex.assemble_main_tex(
        "Book", "Me", ["chapters/01_a"], channel_name="Chan",
        branding_tex=brand.header_footer_tex(b),
        title_page=brand.title_page_tex("Book", "Sub", "Me", logo_rel="branding/logo.png"),
        cover_page=brand.cover_page_tex("branding/cover.png", "Book", "Sub", "Me"),
        list_of_figures=True, part_titles=["P1"]))
    ok, log = latex.compile_tex(main, root / "book", passes=2, cwd=root)
    assert ok, log
    assert (root / "book" / "main.pdf").stat().st_size > 1000


# ---------------------------------------------------------------- CLI


def test_cli_flags_override_settings(tmp_path):
    from bookforge.main import apply_cli_overrides, parse_args

    args = parse_args([
        "https://youtu.be/abc", "--visuals", "rich", "--title", "T", "--header-left", "HL",
        "--footer", "F", "--logo", "l.png", "--no-cover", "--channel-sort", "popular",
        "--no-group-by-playlists", "--workspace", str(tmp_path),
    ])
    s = apply_cli_overrides(Settings(_env_file=None), args)
    assert (s.visuals, s.book_title, s.header_left, s.footer_text, s.logo) == ("rich", "T", "HL", "F", "l.png")
    assert s.cover_generate is False and s.channel_sort == "popular"
    assert s.channel_group_by_playlists is False and s.workspace_root == tmp_path


def test_compiler_cover_uses_given_image(tmp_path):
    from bookforge.agents.compiler import BookCompilerAgent

    src = tmp_path / "art.jpg"
    Image.new("RGB", (50, 80), "red").save(src)
    s = Settings(_env_file=None, cover_image=str(src))
    ws = Workspace(tmp_path, "demo")
    rel = _run(BookCompilerAgent()._cover(s, ws, "T", "", []))
    assert rel == "branding/cover.png" and (ws.book_dir / "cover.png").exists()
    missing = Settings(_env_file=None, cover_image=str(tmp_path / "nope.png"), cover_generate=False)
    ws2 = Workspace(tmp_path, "demo2")
    assert _run(BookCompilerAgent()._cover(missing, ws2, "T", "", [])) == ""


def test_sanitize_maps_invented_visual_environments_to_figure():
    tex = "\\begin{illustration}[h]\\centering x\\end{illustration} \\begin{diagram}y\\end{diagram}"
    out = latex.sanitize_chapter_tex(tex)
    assert "illustration}" not in out and "diagram}" not in out
    assert out.count("\\begin{figure}") == 2 and out.count("\\end{figure}") == 2


def test_visual_sizes_are_normalized_by_kind():
    tex = (r"\includegraphics[width=\textwidth]{chapters/01/figures/diagram_1.pdf} "
           r"\includegraphics{chapters/01/figures/illustration_2.png} "
           r"\includegraphics[width=0.5\linewidth]{chapters/01/figures/other.png}")
    out = latex.normalize_visual_sizes(tex)
    assert r"[width=0.85\linewidth,max height=0.42\textheight,keepaspectratio]{chapters/01/figures/diagram_1.pdf}" in out
    assert r"[width=0.55\linewidth,max height=0.36\textheight,keepaspectratio]{chapters/01/figures/illustration_2.png}" in out
    assert r"[width=0.5\linewidth]{chapters/01/figures/other.png}" in out
    assert latex.find_referenced_files(out) == latex.find_referenced_files(tex)


def test_cover_prompt_never_quotes_the_title():
    from bookforge.agents.compiler import cover_prompt

    p = cover_prompt("Secret Title 123", "Sub", ["Chapter A"])
    assert "Secret Title" not in p and "Chapter A" in p and "no text" in p
