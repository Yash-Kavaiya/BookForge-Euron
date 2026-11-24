"""Safe, deterministic renderers for chapter visuals.

Every renderer here either produces a file that is known to work in the book
(a PDF that pdflatex compiled, or an image Pillow could open) or returns
``None`` and logs why. Nothing in this module may raise into the pipeline: a
visual that fails is dropped, the chapter still builds.

* Diagrams (concept maps, process flows, hierarchies, cycles) are laid out in
  Python and emitted as TikZ, compiled standalone with pdflatex. Graphviz
  (``dot``) is an optional alternative engine.
* Charts are only drawn from numbers the transcript actually states; see
  :func:`verify_chart_against_transcript`.
* Illustrations come from the EURI image model (euri-image-fast) and share
  the process-wide EURI concurrency cap.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from bookforge.schemas import ChartSpec, DiagramSpec, IllustrationSpec
from bookforge.tools.latex import extract_numbers, normalize_number, tex_escape

logger = logging.getLogger(__name__)

MAX_DIAGRAM_NODES = 9
_ROW_WRAP = 3  # nodes per row in snake (boustrophedon) chains
_LAYER_WRAP = 4  # max nodes side by side in a layered layout

# ---------------------------------------------------------------------------
# Diagram graph model
# ---------------------------------------------------------------------------


@dataclass
class DiagramGraph:
    kind: str
    nodes: list[str]
    edges: list[tuple[int, int, str]] = field(default_factory=list)


_EDGE_RE = re.compile(r"^\s*(.+?)\s*(?:-+>|→|=>)\s*(.+?)\s*(?::\s*(.*))?$")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _match_node(name: str, nodes: list[str]) -> int | None:
    key = _norm(name)
    if not key:
        return None
    normed = [_norm(n) for n in nodes]
    if key in normed:
        return normed.index(key)
    for i, n in enumerate(normed):
        if n and (n.startswith(key) or key.startswith(n)):
            return i
    for i, n in enumerate(normed):
        if n and (key in n or n in key):
            return i
    return None


def build_graph(spec: DiagramSpec) -> DiagramGraph | None:
    """Normalise an LLM diagram spec into nodes + edges, or None if unusable."""
    nodes: list[str] = []
    for el in spec.elements:
        el = re.sub(r"\s+", " ", str(el)).strip()
        if el and _norm(el) not in {_norm(n) for n in nodes}:
            nodes.append(el[:80])
    nodes = nodes[:MAX_DIAGRAM_NODES]
    if len(nodes) < 2:
        return None
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for rel in spec.relationships:
        m = _EDGE_RE.match(str(rel))
        if not m:
            continue
        a, b = _match_node(m.group(1), nodes), _match_node(m.group(2), nodes)
        if a is None or b is None or a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        edges.append((a, b, (m.group(3) or "").strip()[:40]))
    kind = spec.diagram_type
    n = len(nodes)
    labels = {(a, b): lab for a, b, lab in edges}
    if kind in ("process", "sequence", "cycle"):
        # Ordered kinds: the element order IS the flow; keep stated labels.
        edges = [(i, i + 1, labels.get((i, i + 1), "")) for i in range(n - 1)]
        if kind == "cycle":
            edges.append((n - 1, 0, labels.get((n - 1, 0), "")))
    else:
        touched = {a for a, _, _ in edges} | {b for _, b, _ in edges}
        for i in range(n):
            if i in touched:
                continue
            if kind in ("hierarchy", "mindmap"):
                edge = (0, i, "") if i else None
            else:
                edge = (i - 1, i, "") if i > 0 else (0, 1, "")
            if edge and (edge[0], edge[1]) not in {(a, b) for a, b, _ in edges}:
                edges.append(edge)
                touched.update(edge[:2])
    return DiagramGraph(kind=kind, nodes=nodes, edges=edges)


def _layers(graph: DiagramGraph) -> list[int]:
    """Longest-path layering; back edges (cycles) are ignored."""
    n = len(graph.nodes)
    succ: dict[int, list[int]] = {i: [] for i in range(n)}
    state = [0] * n  # 0 new, 1 on stack, 2 done
    forward: list[tuple[int, int]] = []

    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b, _ in graph.edges:
        adj[a].append(b)

    def dfs(u: int) -> None:
        state[u] = 1
        for v in adj[u]:
            if state[v] == 0:
                forward.append((u, v))
                dfs(v)
            elif state[v] == 2:
                forward.append((u, v))
            # state 1 -> back edge, skipped
        state[u] = 2

    for i in range(n):
        if state[i] == 0:
            dfs(i)
    for a, b in forward:
        succ[a].append(b)
    layer = [0] * n
    for _ in range(n):  # relax; DAG so n passes suffice
        changed = False
        for a, bs in succ.items():
            for b in bs:
                if layer[b] < layer[a] + 1:
                    layer[b] = layer[a] + 1
                    changed = True
        if not changed:
            break
    return layer


def _is_chain(graph: DiagramGraph) -> bool:
    n = len(graph.nodes)
    pairs = {(a, b) for a, b, _ in graph.edges}
    return all((i, i + 1) in pairs for i in range(n - 1)) and len(pairs) <= n


def layout(graph: DiagramGraph) -> dict[int, tuple[float, float]]:
    """Node positions in cm (x right, y up). Pure and deterministic."""
    n = len(graph.nodes)
    if graph.kind in ("mindmap", "cycle"):
        center_first = graph.kind == "mindmap"
        ring = list(range(1, n)) if center_first else list(range(n))
        radius = 3.6 if len(ring) <= 6 else 4.4
        pos: dict[int, tuple[float, float]] = {}
        if center_first:
            pos[0] = (0.0, 0.0)
        for k, i in enumerate(ring):
            angle = math.radians(90 - k * 360 / max(1, len(ring)))
            pos[i] = (round(radius * 1.25 * math.cos(angle), 3), round(radius * math.sin(angle), 3))
        return pos
    if _is_chain(graph) and graph.kind != "hierarchy":
        if n <= 4:
            return {i: (i * 4.2, 0.0) for i in range(n)}
        pos = {}
        for i in range(n):
            row, col = divmod(i, _ROW_WRAP)
            if row % 2:
                col = _ROW_WRAP - 1 - col
            pos[i] = (col * 4.6, -row * 2.4)
        return pos
    layer = _layers(graph)
    rows: dict[int, list[int]] = {}
    for i, lv in enumerate(layer):
        rows.setdefault(lv, []).append(i)
    pos = {}
    y = 0.0
    for lv in sorted(rows):
        members = rows[lv]
        for start in range(0, len(members), _LAYER_WRAP):
            chunk = members[start : start + _LAYER_WRAP]
            width = (len(chunk) - 1) * 4.0
            for k, i in enumerate(chunk):
                pos[i] = (k * 4.0 - width / 2, y)
            y -= 2.3
    return pos


# ---------------------------------------------------------------------------
# TikZ emission + compile
# ---------------------------------------------------------------------------

STANDALONE_PREAMBLE = r"""\documentclass[tikz,border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\definecolor{primaryblue}{RGB}{37, 99, 235}
\definecolor{accentindigo}{RGB}{79, 70, 229}
\definecolor{softblue}{RGB}{238, 242, 255}
\definecolor{softgreen}{RGB}{236, 253, 245}
\definecolor{accentgreen}{RGB}{16, 185, 129}
\definecolor{darkslate}{RGB}{30, 41, 59}
\definecolor{lightgray}{RGB}{241, 245, 249}
\definecolor{bordergray}{RGB}{203, 213, 225}
\definecolor{softamber}{RGB}{255, 247, 237}
\definecolor{accentamber}{RGB}{234, 88, 12}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric,shadows,calc}
\tikzset{
  bfnode/.style={rectangle, rounded corners=4pt, thick, align=center,
    text width=3.1cm, minimum height=1.0cm, inner sep=5pt,
    font=\small\sffamily\color{darkslate},
    drop shadow={opacity=0.12, shadow xshift=1.2pt, shadow yshift=-1.2pt}},
  bfmain/.style={bfnode, draw=primaryblue, fill=softblue, font=\small\sffamily\bfseries\color{darkslate}},
  bfgoal/.style={bfnode, draw=accentgreen, fill=softgreen},
  bfside/.style={bfnode, draw=bordergray, fill=lightgray},
  bfhub/.style={ellipse, thick, draw=accentamber, fill=softamber, align=center,
    text width=3.0cm, inner sep=4pt, font=\small\sffamily\bfseries\color{darkslate}},
  bfarrow/.style={-{Stealth[length=6pt,width=5pt]}, thick, color=primaryblue},
  bfelabel/.style={font=\scriptsize\sffamily\color{darkslate}, fill=white,
    inner sep=1.5pt, text width=, align=center},
}
"""


def _node_style(graph: DiagramGraph, i: int) -> str:
    n = len(graph.nodes)
    if graph.kind == "mindmap" and i == 0:
        return "bfhub"
    outdeg = sum(1 for a, _, _ in graph.edges if a == i)
    indeg = sum(1 for _, b, _ in graph.edges if b == i)
    if graph.kind in ("flowchart", "process", "sequence") and outdeg == 0 and indeg > 0:
        return "bfgoal"
    if graph.kind == "hierarchy" and i == 0:
        return "bfmain"
    if graph.kind == "hierarchy":
        return "bfside" if outdeg == 0 else "bfmain"
    if graph.kind == "mindmap":
        return "bfside"
    del n
    return "bfmain"


def diagram_to_tikz(graph: DiagramGraph) -> str:
    """A tikzpicture for the graph (no preamble). Pure."""
    pos = layout(graph)
    lines = [r"\begin{tikzpicture}[node distance=1.5cm]"]
    for i, text in enumerate(graph.nodes):
        x, y = pos[i]
        lines.append(
            rf"\node[{_node_style(graph, i)}] (n{i}) at ({x:.2f},{y:.2f}) {{{tex_escape(text)}}};"
        )
    for a, b, label in graph.edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        same_row = abs(ya - yb) < 0.01
        gap = abs(xa - xb)
        backwards = graph.kind not in ("mindmap", "cycle") and (
            (same_row and gap > 4.8) or yb > ya + 0.01
        )
        path = "to[bend left=28]" if backwards else "--"
        if graph.kind == "cycle":
            path = "to[bend left=18]"
        edge_label = (
            rf" node[bfelabel, midway, auto] {{{tex_escape(label)}}}" if label else ""
        )
        if path == "--":
            lines.append(rf"\draw[bfarrow] (n{a}) --{edge_label} (n{b});")
        else:
            lines.append(rf"\draw[bfarrow] (n{a}) {path}{edge_label} (n{b});")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def diagram_to_dot(graph: DiagramGraph) -> str:
    def q(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    rankdir = "TB" if graph.kind in ("hierarchy", "flowchart") else "LR"
    engine_attrs = 'layout=twopi;' if graph.kind == "mindmap" else ""
    out = [
        "digraph G {",
        f'  {engine_attrs} rankdir={rankdir}; bgcolor="white"; pad=0.2; nodesep=0.4; ranksep=0.5;',
        '  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#2563eb", '
        'fontname="Helvetica", fontsize=11, penwidth=1.4, margin="0.15,0.08"];',
        '  edge [color="#2563eb", fontname="Helvetica", fontsize=9, penwidth=1.3];',
    ]
    for i, text in enumerate(graph.nodes):
        import textwrap

        out.append(f"  n{i} [label={q(chr(10).join(textwrap.wrap(text, 22)) or text)}];")
    for a, b, label in graph.edges:
        out.append(f"  n{a} -> n{b}" + (f" [label={q(label)}]" if label else "") + ";")
    out.append("}")
    return "\n".join(out)


def compile_standalone_tikz(tikz: str, dest_pdf: Path, timeout_sec: int = 90) -> bool:
    """Compile a tikzpicture into ``dest_pdf``. True only if pdflatex succeeded."""
    if not shutil.which("pdflatex"):
        logger.warning("pdflatex missing; cannot render TikZ diagram %s", dest_pdf.name)
        return False
    doc = STANDALONE_PREAMBLE + "\\begin{document}\n" + tikz + "\n\\end{document}\n"
    with tempfile.TemporaryDirectory(prefix="bf-tikz-") as tmp:
        src = Path(tmp) / "fig.tex"
        src.write_text(doc, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("TikZ compile for %s failed: %s", dest_pdf.name, exc)
            return False
        pdf = Path(tmp) / "fig.pdf"
        if proc.returncode != 0 or not pdf.exists() or pdf.stat().st_size == 0:
            errs = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("!")]
            logger.warning("TikZ diagram %s dropped: %s", dest_pdf.name, "; ".join(errs[:3]))
            return False
        dest_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, dest_pdf)
        (dest_pdf.with_suffix(".tikz.tex")).write_text(tikz, encoding="utf-8")
    return True


def compile_dot(dot: str, dest_pdf: Path, timeout_sec: int = 60) -> bool:
    if not shutil.which("dot"):
        return False
    try:
        proc = subprocess.run(
            ["dot", "-Tpdf", "-o", str(dest_pdf)],
            input=dot,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("graphviz failed for %s: %s", dest_pdf.name, exc)
        return False
    ok = proc.returncode == 0 and dest_pdf.exists() and dest_pdf.stat().st_size > 0
    if not ok:
        logger.warning("graphviz diagram %s dropped: %s", dest_pdf.name, (proc.stderr or "")[:300])
        dest_pdf.unlink(missing_ok=True)
    return ok


def render_diagram(spec: DiagramSpec, dest_pdf: Path, engine: str = "tikz") -> Path | None:
    """Render a diagram spec to PDF, or None (logged) if it cannot be drawn."""
    dest_pdf = Path(dest_pdf)
    if dest_pdf.exists() and dest_pdf.stat().st_size > 0:
        return dest_pdf
    try:
        graph = build_graph(spec)
        if graph is None:
            logger.warning("diagram '%s' dropped: fewer than 2 usable elements", spec.title)
            return None
        dest_pdf.parent.mkdir(parents=True, exist_ok=True)
        order = {
            "tikz": ["tikz", "graphviz"],
            "graphviz": ["graphviz", "tikz"],
            "auto": ["graphviz", "tikz"] if shutil.which("dot") else ["tikz"],
        }.get(engine, ["tikz", "graphviz"])
        for eng in order:
            ok = (
                compile_standalone_tikz(diagram_to_tikz(graph), dest_pdf)
                if eng == "tikz"
                else compile_dot(diagram_to_dot(graph), dest_pdf)
            )
            if ok:
                return dest_pdf
    except Exception:  # noqa: BLE001 - a broken visual never breaks the build
        logger.exception("diagram '%s' dropped: renderer crashed", spec.title)
    return None


# ---------------------------------------------------------------------------
# Charts: grounded in the transcript only
# ---------------------------------------------------------------------------

_TS_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.*)$")


def _transcript_lines(transcript: str) -> list[tuple[str, str]]:
    out = []
    for line in (transcript or "").splitlines():
        m = _TS_LINE_RE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _ts_seconds(ts: str) -> int:
    try:
        h, m, s = (int(p) for p in ts.split(":"))
    except ValueError:
        return -1
    return h * 3600 + m * 60 + s


def verify_chart_against_transcript(
    spec: ChartSpec, transcript: str, window_sec: int = 300
) -> tuple[bool, str, str]:
    """Check every chart value is a number the speaker actually stated.

    Returns (ok, source_timestamp, reason). The timestamp is the transcript
    marker where the values are stated: the analyst's ``source_timestamp``
    when the numbers occur within ``window_sec`` of it, otherwise the first
    marker where any value occurs. Charts with fewer than 2 values, or with
    any value absent from the transcript, are rejected (never invented data).
    """
    values = [v for v in spec.values if v is not None]
    if len(values) < 2:
        return False, "", "fewer than 2 values"
    if len(spec.categories) and len(spec.categories) != len(values):
        return False, "", "categories and values differ in length"
    lines = _transcript_lines(transcript)
    if not lines:
        corpus = {normalize_number(t) for t in extract_numbers(transcript or "")}
        missing = [v for v in values if normalize_number(str(v)) not in corpus]
        if missing:
            return False, "", f"values not stated in transcript: {missing[:4]}"
        return True, "", ""
    per_line = [(ts, {normalize_number(t) for t in extract_numbers(text)}) for ts, text in lines]
    stated = set().union(*(nums for _, nums in per_line))
    wanted = [normalize_number(repr(float(v)) if not float(v).is_integer() else str(int(v))) for v in values]
    missing = [w for w in wanted if w not in stated]
    if missing:
        return False, "", f"values not stated in transcript: {missing[:4]}"
    hits = [ts for ts, nums in per_line if nums & set(wanted)]
    ts = spec.source_timestamp.strip()
    if ts and hits:
        anchor = _ts_seconds(ts)
        near = [h for h in hits if anchor >= 0 and abs(_ts_seconds(h) - anchor) <= window_sec]
        if near:
            return True, ts if ts in {h for h, _ in lines} else near[0], ""
    return True, hits[0] if hits else ts, ""


def chart_caption(spec: ChartSpec, source_ts: str) -> str:
    base = spec.caption.strip().rstrip(".")
    if source_ts:
        return f"{base}. Source: values as stated in the video at {source_ts}."
    return f"{base}. Source: values as stated in the video."


# ---------------------------------------------------------------------------
# Illustrations (EURI image model)
# ---------------------------------------------------------------------------

ILLUSTRATION_STYLE = (
    "Clean editorial textbook illustration, flat vector style, soft muted "
    "palette, white background, no text, no letters, no numbers, no logos, "
    "no watermark."
)


def _save_image_bytes(data: bytes, dest: Path, max_side: int = 1400) -> Path | None:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            img = img.convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp" + dest.suffix)
            img.save(tmp, format="PNG" if dest.suffix.lower() == ".png" else "JPEG", quality=90)
            tmp.replace(dest)
            return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("image for %s dropped: not a readable image (%s)", dest.name, exc)
        return None


async def generate_image(
    prompt: str,
    dest: Path,
    *,
    retry_delays: tuple[float, ...] = (20.0, 60.0),
    **kwargs,
) -> Path | None:
    """Generate one image with the EURI image model into ``dest`` (or None).

    Retries with backoff when EURI is busy (empty reply / 429 / 5xx); see
    :func:`_generate_image_once` for the single attempt.
    """
    delays = list(retry_delays)
    while True:
        result = await _generate_image_once(prompt, dest, **kwargs)
        if result is not _RETRY:
            return result
        if not delays:
            logger.warning("image %s dropped: EURI still busy after retries", Path(dest).name)
            return None
        await asyncio.sleep(delays.pop(0))


_RETRY = object()


async def _generate_image_once(
    prompt: str,
    dest: Path,
    *,
    api_key: str,
    api_base: str,
    model: str = "euri-image-fast",
    max_concurrency: int = 2,
    size: str = "1024x1024",
    timeout_sec: float = 180.0,
    transport=None,
) -> Path | None:
    """Generate one image with the EURI image model into ``dest`` (or None).

    Shares the process-wide EURI semaphore with the chat agents so the
    2-concurrent-request limit holds. The key is sent as a header only and
    never logged.
    """
    import httpx

    from bookforge.agents.model_resolver import euri_semaphore

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not api_key:
        logger.warning("image %s skipped: EURI_API_KEY not set", dest.name)
        return None
    try:
        async with httpx.AsyncClient(
            base_url=api_base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_sec, connect=20),
            transport=transport,
        ) as client:
            async with euri_semaphore(max_concurrency):
                resp = await client.post(
                    "/images/generations",
                    json={"model": model, "prompt": prompt, "n": 1, "size": size},
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning("image %s: HTTP %s, will retry", dest.name, resp.status_code)
                return _RETRY
            if resp.status_code >= 400:
                logger.warning("image %s dropped: HTTP %s", dest.name, resp.status_code)
                return None
            items = (resp.json() or {}).get("data") or []
            if not items:
                logger.warning("image %s: empty response (EURI busy?), will retry", dest.name)
                return _RETRY
            first = items[0]
            if first.get("b64_json"):
                data = base64.b64decode(first["b64_json"])
            elif str(first.get("url", "")).startswith("data:"):
                data = base64.b64decode(first["url"].split(",", 1)[1])
            elif first.get("url"):
                # Fresh client: the image URL is usually a signed storage link
                # that rejects (and must never receive) the EURI bearer key.
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout_sec, connect=20),
                    transport=transport,
                    follow_redirects=True,
                ) as plain:
                    img = await plain.get(first["url"])
                if img.status_code >= 400:
                    logger.warning("image %s dropped: download HTTP %s", dest.name, img.status_code)
                    return None
                data = img.content
            else:
                return None
    except Exception as exc:  # noqa: BLE001 - network/JSON trouble -> drop the visual
        logger.warning("image %s dropped: %s", dest.name, type(exc).__name__)
        return None
    return _save_image_bytes(data, dest)


def illustration_prompt(spec: IllustrationSpec, book_context: str = "") -> str:
    ctx = f" Context: {book_context}." if book_context else ""
    return f"{spec.prompt.strip().rstrip('.')}.{ctx} {ILLUSTRATION_STYLE}"


# ---------------------------------------------------------------------------
# Screenshots: prefer informative frames (slides, code, diagrams)
# ---------------------------------------------------------------------------


def frame_information_score(path: Path) -> float:
    """Higher = more informative (text/diagram-like) frame. 0 on error.

    Combines edge density (text, lines, UI) with luminance entropy; talking
    heads and flat backgrounds score low, slides and code score high.
    """
    from PIL import Image, ImageFilter, ImageStat

    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((160, 90))
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_density = sum(1 for p in edges.getdata() if p > 40) / (160 * 90)
            hist = gray.histogram()
            total = float(sum(hist)) or 1.0
            entropy = -sum((h / total) * math.log2(h / total) for h in hist if h)
            contrast = ImageStat.Stat(gray).stddev[0] / 128.0
    except Exception:  # noqa: BLE001
        return 0.0
    return round(edge_density * 4.0 + entropy / 8.0 + contrast * 0.5, 4)


def pick_informative(scored: list[tuple[float, float, object]], k: int) -> list[object]:
    """Pick ``k`` items from (timestamp, score, item): the best-scoring item in
    each of ``k`` equal timeline buckets, returned in timeline order."""
    if k <= 0 or not scored:
        return []
    scored = sorted(scored, key=lambda t: t[0])
    if len(scored) <= k:
        return [it for _, _, it in scored]
    picked = []
    step = len(scored) / k
    for b in range(k):
        bucket = scored[int(b * step) : int((b + 1) * step)] or [scored[int(b * step)]]
        picked.append(max(bucket, key=lambda t: t[1]))
    return [it for _, _, it in sorted(picked, key=lambda t: t[0])]
