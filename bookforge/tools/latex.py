"""LaTeX helpers: deterministic asset rendering + compilation.

The book owns ONE preamble (here); chapters are preamble-free bodies, so each
chapter can also be compiled standalone by wrapping it in the same preamble.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from bookforge.schemas import ChartSpec, TableSpec
from bookforge.tools.workspace import Workspace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------

_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "≥": r"$\ge$",
    "≤": r"$\le$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "×": r"$\times$",
    "±": r"$\pm$",
    "÷": r"$\div$",
    "∞": r"$\infty$",
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "⇒": r"$\implies$",
    "µ": r"$\mu$",
    "•": r"\textbullet{}",
    "—": r"---",
    "–": r"--",
    "“": r"``",
    "”": r"''",
    "‘": r"`",
    "’": r"'",
    "…": r"\dots{}",
    "™": r"\texttrademark{}",
    "®": r"\textregistered{}",
    "©": r"\textcopyright{}",
    "°": r"$^\circ$",
}


def tex_escape(text: str) -> str:
    return "".join(_SPECIALS.get(ch, ch) for ch in str(text))


# ---------------------------------------------------------------------------
# Tables (booktabs fragments, \input by the chapter)
# ---------------------------------------------------------------------------


def render_table_fragment(spec: TableSpec, dest: Path, label: str = "") -> Path:
    """Full-width tabularx fragment: columns wrap instead of overflowing the
    margin, headers are bold, and the caption/label sit above the rules."""
    ncols = max(1, len(spec.headers))
    # First column stays a fixed narrow label column when there is room for
    # it; every other column is a wrapping, ragged-right X column (type L,
    # defined in the preamble) so wide cells can never overrun the page.
    if ncols == 1:
        col_spec = "L"
    elif ncols <= 3:
        col_spec = ">{\\raggedright\\arraybackslash}p{0.24\\textwidth}" + "L" * (ncols - 1)
    else:
        col_spec = "L" * ncols

    lines = [
        "\\begin{table}[!htb]",
        "  \\centering",
        f"  \\caption{{{tex_escape(spec.caption)}}}",
    ]
    if label:
        lines.append(f"  \\label{{{label}}}")
    if ncols >= 4:
        lines.append("  \\small")
    lines += [
        f"  \\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}",
        "    \\toprule",
        "    "
        + " & ".join(f"\\textbf{{{tex_escape(h)}}}" for h in spec.headers)
        + " \\\\",
        "    \\midrule",
    ]
    for row in spec.rows:
        padded = list(row)[:ncols] + [""] * max(0, ncols - len(row))
        lines.append("    " + " & ".join(tex_escape(c) for c in padded) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabularx}",
    ]
    if spec.note:
        lines += [
            "  \\par\\vspace{4pt}",
            f"  \\begin{{minipage}}{{0.92\\textwidth}}\\footnotesize\\itshape {tex_escape(spec.note)}\\end{{minipage}}",
        ]
    lines.append("\\end{table}")
    return Workspace.write_text(Path(dest), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Charts (matplotlib -> vector PDF)
# ---------------------------------------------------------------------------


# Palette matched to the book's primaryblue/accentindigo/accentgreen colors.
_CHART_COLORS = [
    "#2563eb", "#4f46e5", "#10b981", "#f59e0b",
    "#ef4444", "#0ea5e9", "#8b5cf6", "#14b8a6",
]
_GRID_COLOR = "#e2e8f0"
_INK_COLOR = "#334155"


def _wrap_labels(labels: list[str], width: int = 14, max_lines: int = 2) -> list[str]:
    """Wrap long category labels onto <= max_lines lines so they never
    collide with neighbours or run off the axes."""
    import textwrap

    wrapped = []
    for lab in labels:
        lines = textwrap.wrap(str(lab), width=width) or [""]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][: width - 1] + "…"
        wrapped.append("\n".join(lines))
    return wrapped


def render_chart(spec: ChartSpec, dest: Path) -> Path:
    """Render a chart spec to PDF. Values come from the validated analysis.

    Charts are saved with a tight bounding box so they center cleanly inside
    the LaTeX figure environment with no lopsided whitespace.

    Uses matplotlib's object-oriented API only (no pyplot, no rcParams
    mutation) so several charts can render concurrently from worker threads.
    """
    import textwrap

    from matplotlib.figure import Figure

    dest = Path(dest)
    if dest.exists():
        return dest

    values = list(spec.values or [])
    categories = [str(c) for c in (spec.categories or [])]
    if not categories:
        categories = [str(i + 1) for i in range(len(values))]
    elif len(categories) > len(values):
        categories = categories[: len(values)]
    elif len(values) > len(categories):
        values = values[: len(categories)]

    if not values:
        # Fallback if both empty
        categories = ["1"]
        values = [1.0]

    chart_type = spec.chart_type
    # Pie charts are only valid for positive shares that sum to something.
    if chart_type == "pie" and (min(values) < 0 or sum(values) <= 0):
        chart_type = "bar"

    figsize = (5.4, 3.5) if chart_type == "pie" else (6.0, 3.4)
    fig = Figure(figsize=figsize, facecolor="white")
    ax = fig.subplots()
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(_INK_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8.5, colors=_INK_COLOR)

    wrapped = _wrap_labels(categories)
    long_labels = any(len(c) > 10 for c in categories) or len(categories) > 7
    rotation = 30 if long_labels else 0
    anchor = "right" if long_labels else "center"

    if chart_type == "pie":
        colors = [_CHART_COLORS[i % len(_CHART_COLORS)] for i in range(len(values))]
        if len(values) > 5:
            # Many slices: labels overlap around the pie — use a legend.
            pie_wedges, _, _ = ax.pie(
                values,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                pctdistance=0.75,
                wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                textprops={"fontsize": 8, "color": "white"},
            )
            ax.legend(
                pie_wedges,
                _wrap_labels(categories, width=18, max_lines=2),
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8,
                frameon=False,
            )
        else:
            ax.pie(
                values,
                labels=wrapped,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                pctdistance=0.72,
                wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                textprops={"fontsize": 8.5},
            )
        ax.set_aspect("equal")
    elif chart_type == "line":
        ax.plot(
            range(len(values)),
            values,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=_CHART_COLORS[0],
            zorder=3,
        )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(wrapped, rotation=rotation, ha=anchor)
        ax.grid(color=_GRID_COLOR, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.12)
    else:  # bar (also the fallback for unknown types)
        bars = ax.bar(
            range(len(values)),
            values,
            width=0.62,
            color=_CHART_COLORS[0],
            zorder=3,
        )
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(wrapped, rotation=rotation, ha=anchor)
        ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, fmt="%.4g", fontsize=8, padding=2, color=_INK_COLOR)
        ax.margins(y=0.14)

    if chart_type != "pie":
        if spec.x_label:
            ax.set_xlabel(spec.x_label, fontsize=9.5, color=_INK_COLOR)
        if spec.y_label:
            ax.set_ylabel(spec.y_label, fontsize=9.5, color=_INK_COLOR)
    ax.set_title(
        textwrap.fill(spec.title, width=52),
        fontsize=10.5,
        fontweight="bold",
        color=_INK_COLOR,
        pad=10,
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Render to a temp file first so a concurrent reader never sees a partial PDF.
    tmp = dest.with_suffix(".tmp.pdf")
    fig.savefig(tmp, format="pdf", bbox_inches="tight", pad_inches=0.06)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Book assembly
# ---------------------------------------------------------------------------

PREAMBLE = r"""\documentclass[11pt,openany]{book}
\usepackage{etex}
\usepackage[utf8]{inputenc}
\usepackage{newunicodechar}
\usepackage[margin=1in,headheight=15pt]{geometry}
\usepackage{graphicx}
\usepackage[export]{adjustbox}
\usepackage{amsmath,amssymb}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{xcolor}
\definecolor{primaryblue}{RGB}{37, 99, 235}
\definecolor{accentindigo}{RGB}{79, 70, 229}
\definecolor{softblue}{RGB}{238, 242, 255}
\definecolor{softgreen}{RGB}{236, 253, 245}
\definecolor{accentgreen}{RGB}{16, 185, 129}
\definecolor{darkslate}{RGB}{30, 41, 59}
\definecolor{lightgray}{RGB}{241, 245, 249}
\definecolor{bordergray}{RGB}{203, 213, 225}

\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes,shapes.geometric,shadows,calc,fit}

\tikzset{
    every node/.style={align=center},
    modernbox/.style={
        rectangle,
        draw=primaryblue,
        fill=softblue,
        thick,
        rounded corners=4pt,
        minimum height=1.0cm,
        minimum width=2.4cm,
        inner sep=6pt,
        align=center,
        font=\small\bfseries\color{darkslate},
        drop shadow={opacity=0.15, shadow xshift=1.5pt, shadow yshift=-1.5pt}
    },
    accentbox/.style={
        rectangle,
        draw=accentgreen,
        fill=softgreen,
        thick,
        rounded corners=4pt,
        minimum height=1.0cm,
        minimum width=2.4cm,
        inner sep=6pt,
        align=center,
        font=\small\bfseries\color{darkslate},
        drop shadow={opacity=0.15, shadow xshift=1.5pt, shadow yshift=-1.5pt}
    },
    neutralbox/.style={
        rectangle,
        draw=bordergray,
        fill=lightgray,
        thick,
        rounded corners=4pt,
        minimum height=1.0cm,
        minimum width=2.4cm,
        inner sep=6pt,
        align=center,
        font=\small\color{darkslate},
        drop shadow={opacity=0.10, shadow xshift=1pt, shadow yshift=-1pt}
    },
    flowarrow/.style={
        ->,
        >={Stealth[length=6pt, width=5pt]},
        thick,
        color=primaryblue,
        font=\footnotesize\color{darkslate}
    }
}
\usepackage{morefloats}
\extrafloats{100}
% pdflatex + bitmap CM cannot micro-expand; lmodern is Type1 and
% expansion=false keeps the book compiling if a later font is bitmap.
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[expansion=false]{microtype}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{caption}
\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\hypersetup{colorlinks=true, linkcolor=primaryblue, urlcolor=accentindigo}
\sloppy
\emergencystretch=3em

% ---- Float discipline: keep figures/tables near their reference and never
% ---- let floats crowd out body text or drift to the chapter end.
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.55}
\renewcommand{\textfraction}{0.10}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setlength{\floatsep}{14pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{16pt plus 2pt minus 4pt}
\setlength{\intextsep}{14pt plus 2pt minus 2pt}

% Images included without explicit options are sized safely by default.
\setkeys{Gin}{width=0.85\linewidth,keepaspectratio}

\captionsetup{font=small,labelfont={bf,color=primaryblue},skip=8pt,width=0.9\linewidth}

% Comfortable table rows + a wrapping ragged-right tabularx column type.
\renewcommand{\arraystretch}{1.25}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}

% ---- Running header/footer -------------------------------------------------
% Per-chapter metadata is injected by main.tex via \renewcommand before each
% chapter \input. The empty defaults keep standalone chapter QA compiles valid.
\newcommand{\BFchannelName}{}
\newcommand{\BFvideoFooter}{}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\small\itshape\nouppercase{\leftmark}}
\fancyfoot[L]{\footnotesize\textcolor{darkslate}{\BFchannelName}}
\fancyfoot[C]{\small\thepage}
\fancyfoot[R]{\footnotesize\BFvideoFooter}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \fancyfoot[L]{\footnotesize\textcolor{darkslate}{\BFchannelName}}%
  \fancyfoot[C]{\small\thepage}%
  \fancyfoot[R]{\footnotesize\BFvideoFooter}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0.4pt}%
}

% ---- Callout boxes (notes, key insights, formulas) -------------------------
% Built from LaTeX primitives only, so no extra packages are required.
% Content cannot break across pages: keep each box short. The optional
% argument overrides the box title.
\newsavebox{\BFboxbody}
\newenvironment{notebox}[1][Note]
  {\par\addvspace{12pt}\begin{lrbox}{\BFboxbody}\begin{minipage}{\dimexpr\linewidth-20pt\relax}%
   \small\textbf{\textcolor{primaryblue}{#1.}}\hspace{0.4em}}
  {\end{minipage}\end{lrbox}\noindent
   {\setlength{\fboxrule}{0.9pt}\setlength{\fboxsep}{9pt}%
    \fcolorbox{primaryblue}{softblue}{\usebox{\BFboxbody}}}\par\addvspace{12pt}}
\newenvironment{keyinsight}[1][Key Insight]
  {\par\addvspace{12pt}\begin{lrbox}{\BFboxbody}\begin{minipage}{\dimexpr\linewidth-20pt\relax}%
   \small\textbf{\textcolor{accentgreen}{#1.}}\hspace{0.4em}}
  {\end{minipage}\end{lrbox}\noindent
   {\setlength{\fboxrule}{0.9pt}\setlength{\fboxsep}{9pt}%
    \fcolorbox{accentgreen}{softgreen}{\usebox{\BFboxbody}}}\par\addvspace{12pt}}
\newenvironment{formulabox}[1][Formula]
  {\par\addvspace{12pt}\begin{lrbox}{\BFboxbody}\begin{minipage}{\dimexpr\linewidth-20pt\relax}%
   \small\textbf{\textcolor{accentindigo}{#1.}}\hspace{0.4em}}
  {\end{minipage}\end{lrbox}\noindent
   {\setlength{\fboxrule}{0.9pt}\setlength{\fboxsep}{9pt}%
    \fcolorbox{accentindigo}{lightgray}{\usebox{\BFboxbody}}}\par\addvspace{12pt}}
"""


def chapter_wrapper(preamble: str, body: str) -> str:
    """Standalone-compilable wrapper used by the QA compile check."""
    return f"{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"


def _smart_quotes(text: str) -> str:
    """Typographic quotes for plain ASCII "double quotes" in titles."""
    return re.sub(r'"([^"]*)"', r"``\1''", text)


def assemble_main_tex(
    book_title: str,
    author_line: str,
    chapter_rel_dirs: list[str],
    channel_name: str = "",
    chapter_links: list[dict] | None = None,
    *,
    branding_tex: str = "",
    title_page: str = "",
    cover_page: str = "",
    list_of_figures: bool = False,
    part_titles: list[str] | None = None,
) -> str:
    """main.tex that \\input's chapter bodies via workspace-root-relative paths.

    `channel_name` appears in the left footer of every page. `chapter_links`
    aligns 1:1 with `chapter_rel_dirs`; each item may carry "url" and
    "link_text" for that chapter's clickable YouTube link in the right footer.
    """
    links = chapter_links or []
    parts = list(part_titles or [])
    use_parts = len({p for p in parts if p}) >= 2
    current_part = None
    blocks = []
    for i, rel in enumerate(chapter_rel_dirs):
        part = parts[i] if i < len(parts) else ""
        if use_parts and part and part != current_part:
            blocks.append(f"\\part{{{tex_escape(part)}}}")
            current_part = part
        meta = links[i] if i < len(links) else {}
        url = (meta.get("url") or "").strip()
        if url:
            display = tex_escape(meta.get("link_text") or url)
            footer = (
                f"\\renewcommand{{\\BFvideoFooter}}"
                f"{{\\href{{{url}}}{{\\textcolor{{accentindigo}}{{{display}}}}}}}"
            )
        else:
            footer = "\\renewcommand{\\BFvideoFooter}{}"
        blocks.append(f"{footer}\n\\input{{{rel}/chapter.tex}}")
    chapters_block = "\n\n".join(blocks)
    channel_cmd = (
        f"\\renewcommand{{\\BFchannelName}}{{{tex_escape(channel_name)}}}\n"
        if channel_name
        else ""
    )
    front = (cover_page or "") + (title_page or "\\maketitle\n")
    lof = "\\listoffigures\n" if list_of_figures else ""
    return (
        "\\input{preamble.tex}\n"
        f"{channel_cmd}"
        f"{branding_tex}"
        f"\\title{{{_smart_quotes(tex_escape(book_title))}}}\n"
        f"\\author{{{tex_escape(author_line)}}}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n"
        f"\\frontmatter\n{front}\\tableofcontents\n{lof}\\mainmatter\n\n"
        f"{chapters_block}\n\n"
        "\\backmatter\n\\end{document}\n"
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def pdflatex_available() -> bool:
    return shutil.which("pdflatex") is not None


def compile_tex(
    main_tex: Path,
    build_dir: Path,
    passes: int = 2,
    timeout_sec: int = 180,
    cwd: Path | None = None,
) -> tuple[bool, str]:
    """Compile `main_tex`; returns (ok, log_tail_for_diagnostics).

    `cwd` controls how relative paths inside the document (figure/table
    references, \\input) resolve. BookForge always compiles with cwd =
    workspace root so root-relative asset paths work everywhere.
    """
    main_tex = Path(main_tex)
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    if not pdflatex_available():
        return False, "pdflatex not found on PATH"
    work_cwd = Path(cwd) if cwd else main_tex.parent
    main_ref = str(main_tex) if main_tex.is_absolute() else main_tex.name

    last_log = ""
    for _ in range(max(1, passes)):
        proc = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                main_ref,
            ],
            cwd=work_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        last_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            return False, extract_latex_errors(last_log)
    return True, last_log[-1500:]


_LATEX_ERR = re.compile(r"^!.*(?:\n+l\.\d+.*)?", re.MULTILINE)


def extract_latex_errors(log: str, max_errors: int = 5) -> str:
    """Pull the '!' error lines + context out of a pdflatex log."""
    errors = _LATEX_ERR.findall(log)
    if not errors:
        return log[-1500:]
    return "\n---\n".join(errors[:max_errors])


# ---------------------------------------------------------------------------
# Cheap static checks (run before paying for a compile)
# ---------------------------------------------------------------------------


def find_referenced_files(tex_body: str) -> list[str]:
    """All filenames referenced via includegraphics / input."""
    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_body)
    refs += re.findall(r"\\input\{([^}]+)\}", tex_body)
    return refs


_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(?:auto|page|eq|c|C)?ref\{([^}]+)\}")


def find_labels(tex_body: str) -> set[str]:
    return {m.strip() for m in _LABEL_RE.findall(tex_body)}


def find_refs(tex_body: str) -> list[str]:
    """Every label referenced via \\ref / \\eqref / \\autoref / \\pageref."""
    out: list[str] = []
    for group in _REF_RE.findall(tex_body):
        out.extend(part.strip() for part in group.split(",") if part.strip())
    return out


_BEGIN_RE = re.compile(r"\\begin\{([A-Za-z*]+)\}")
_END_RE = re.compile(r"\\end\{([A-Za-z*]+)\}")
_VERBATIM_RE = re.compile(r"\\verb(.)(.*?)\1")


def check_environment_balance(tex_body: str) -> list[str]:
    """Return human-readable problems for unbalanced environments / braces.

    Cheap approximation of what pdflatex would report as
    '\\begin{X} ended by \\end{Y}' or 'Missing } inserted', run before
    paying for a compile so the refiner gets a precise, quotable defect.
    """
    problems: list[str] = []
    body = _VERBATIM_RE.sub("", tex_body)
    body = re.sub(r"(?<!\\)%.*", "", body)  # drop comments
    begins: dict[str, int] = {}
    for env in _BEGIN_RE.findall(body):
        begins[env] = begins.get(env, 0) + 1
    ends: dict[str, int] = {}
    for env in _END_RE.findall(body):
        ends[env] = ends.get(env, 0) + 1
    for env in sorted(set(begins) | set(ends)):
        b, e = begins.get(env, 0), ends.get(env, 0)
        if b != e:
            problems.append(
                f"environment '{env}' is unbalanced: {b} \\begin vs {e} \\end"
            )
    stripped = re.sub(r"\\[{}]", "", body)  # escaped braces are literal text
    depth = 0
    for ch in stripped:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                problems.append("stray closing brace '}' before any opening '{'")
                break
    if depth > 0:
        problems.append(f"{depth} unclosed '{{' brace(s)")
    return problems


_STRIP_BLOCKS_RE = re.compile(
    r"\\begin\{(tikzpicture|equation\*?|align\*?|gather\*?|verbatim|lstlisting)\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)
_STRIP_INLINE_MATH_RE = re.compile(r"\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
# Commands whose argument is machine text (paths, labels, sizes), never prose.
_STRIP_CMD_ARGS_RE = re.compile(
    r"\\(?:includegraphics(?:\[[^\]]*\])?|input|label|ref|eqref|autoref|pageref|"
    r"chaptermark|href|url|texttt|cite|begin|end|usetikzlibrary|renewcommand)"
    r"\{[^}]*\}"
)
_STRIP_OPTIONS_RE = re.compile(r"\[[^\]]*\]")


def extract_prose(tex_body: str) -> str:
    """Chapter text with math, TikZ, paths, labels and options removed.

    What remains is the wording a reader sees, which is the only place a
    hallucinated statistic can hide. Used by the unsupported-number check.
    """
    body = _STRIP_BLOCKS_RE.sub(" ", tex_body)
    body = _STRIP_INLINE_MATH_RE.sub(" ", body)
    body = _STRIP_CMD_ARGS_RE.sub(" ", body)
    body = _STRIP_OPTIONS_RE.sub(" ", body)
    body = re.sub(r"(?<!\\)%.*", " ", body)
    body = re.sub(r"\\[A-Za-z@]+\*?", " ", body)  # remaining commands
    return body


_NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w.])")


def extract_numbers(text: str) -> list[str]:
    """Numeric tokens as written (commas kept) in first-seen order."""
    return [m.group(1) for m in _NUMBER_RE.finditer(text or "")]


def normalize_number(token: str) -> str:
    """Canonical form so '1,000', '1000' and '1000.0' compare equal."""
    tok = token.replace(",", "")
    try:
        value = float(tok)
    except ValueError:
        return tok
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


def normalize_math(expr: str) -> str:
    """Whitespace/spacing-insensitive form of a LaTeX math expression."""
    expr = re.sub(r"\\(?:,|;|!|quad|qquad|left|right|displaystyle)", "", expr)
    expr = expr.replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", expr)


# Running-header safety: a chapter title longer than this is truncated in the
# page header (via \chaptermark) so it can never overflow or get cut off.
_MAX_HEADER_TITLE_CHARS = 58

_CHAPTER_TITLE_RE = re.compile(r"\\chapter(?!\[)\{([^{}]+)\}")


def _shorten_title(title: str, max_chars: int = _MAX_HEADER_TITLE_CHARS) -> str:
    """Cut a title at a word boundary and add an ellipsis."""
    words = title.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join(kept + [word])
        if len(candidate) > max_chars:
            break
        kept.append(word)
    if not kept:  # a single enormous word
        return title[: max_chars - 1].rstrip() + r"\,\dots{}"
    return " ".join(kept) + r"\,\dots{}"


def ensure_short_chapter_marks(
    body: str, max_chars: int = _MAX_HEADER_TITLE_CHARS
) -> str:
    """Append \\chaptermark with a truncated title when the chapter title is
    too long for the running header. The TOC keeps the full title."""

    def repl(m: re.Match) -> str:
        title = m.group(1).strip()
        # Titles containing commands are left alone — cutting could split one.
        if len(title) <= max_chars or "\\" in title:
            return m.group(0)
        return m.group(0) + f"\n\\chaptermark{{{_shorten_title(title, max_chars)}}}"

    # Idempotent: this runs on every QA/refine pass, so drop marks we added
    # before instead of stacking duplicates.
    body = re.sub(r"(?m)^\\chaptermark\{.*\}[ \t]*\n?", "", body)
    return _CHAPTER_TITLE_RE.sub(repl, body, count=1)


# Sizes per visual kind (by asset filename). `max height` (adjustbox export)
# stops tall, narrow diagrams or portrait images from filling a whole page.
VISUAL_SIZES = {
    "diagram": r"width=0.85\linewidth,max height=0.42\textheight,keepaspectratio",
    "chart": r"width=0.8\linewidth,max height=0.4\textheight,keepaspectratio",
    "illustration": r"width=0.55\linewidth,max height=0.36\textheight,keepaspectratio",
    "frame": r"width=0.8\linewidth,max height=0.38\textheight,keepaspectratio",
}
_VISUAL_INCLUDE_RE = re.compile(
    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*/figures/(diagram|chart|illustration|frame)_[^}]*)\}"
)


def normalize_visual_sizes(tex: str) -> str:
    """Give every pipeline visual a consistent, page-safe size (pure)."""
    return _VISUAL_INCLUDE_RE.sub(
        lambda m: f"\\includegraphics[{VISUAL_SIZES[m.group(2)]}]{{{m.group(1)}}}", tex
    )


def sanitize_chapter_tex(tex: str) -> str:
    """Strip markdown fences / stray commentary an LLM may add despite rules."""
    body = (tex or "").strip()
    fence = re.match(r"^```(?:latex|tex)?\s*\n(?P<body>.*)\n?```\s*$", body, re.DOTALL)
    if fence:
        body = fence.group("body").strip()
    # never trust LLM-supplied preamble material
    body = re.sub(r"\\documentclass[^\n]*\n", "", body)
    body = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}\n?", "", body)
    body = re.sub(r"\\usetikzlibrary\{[^}]*\}\n?", "", body)
    body = re.sub(r"\\begin\{document\}|\\end\{document\}", "", body)

    # Auto-convert HTML-style tags that LLMs sometimes hallucinate into LaTeX equivalents
    body = re.sub(r"<\s*/\s*figure\s*>", r"\\end{figure}", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*figure\s*>", r"\\begin{figure}[htbp]", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*/\s*table\s*>", r"\\end{table}", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*table\s*>", r"\\begin{table}[htbp]", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*/?\s*(?:div|span|p|section|article)[^>]*>", "", body, flags=re.IGNORECASE)

    # LLMs often close environments as \end{figure> (HTML leftover). Fix those.
    body = re.sub(r"\\(begin|end)\{([A-Za-z*]+)>", r"\\\1{\2}", body)

    body = normalize_visual_sizes(body)

    # Visual kinds from the assets manifest are not LaTeX environments; a
    # model that writes begin-illustration / begin-diagram means a figure.
    body = re.sub(
        r"\\begin\{(?:illustration|diagram|chart|visual|screenshot)\}(\[[^\]]*\])?",
        lambda m: "\\begin{figure}" + (m.group(1) or "[!htbp]"),
        body,
    )
    body = re.sub(r"\\end\{(?:illustration|diagram|chart|visual|screenshot)\}", r"\\end{figure}", body)

    # Ensure tikzpicture has align=center or every node/.style={align=center} so \\ in node labels doesn't crash LR mode
    body = re.sub(
        r"\\begin\{tikzpicture\}(?:\[([^\]]*)\])?",
        lambda m: r"\begin{tikzpicture}[" + (m.group(1) + ", " if m.group(1) else "") + r"every node/.append style={align=center}]" if "align=center" not in (m.group(1) or "") else m.group(0),
        body,
    )

    # Unwrap redundant \begin{table}...\input{.../tables/table_*.tex}...\end{table}
    # because table fragments already contain \begin{table}...\end{table}
    body = re.sub(
        r"\\begin\{table\}(?:\[[^\]]*\])?\s*(\\centering\s*)?(?:\\caption\{[^}]*\}\s*)?(?:\\label\{[^}]*\}\s*)?(\\input\{[^}]*tables/table_\d+\.tex\})\s*(?:\\caption\{[^}]*\}\s*)?(?:\\label\{[^}]*\}\s*)?\\end\{table\}",
        r"\2",
        body,
    )

    # Normalize float placement and guarantee centering so every figure and
    # table sits consistently within the text column (no drifting/left-hugging
    # floats regardless of what the LLM emitted).
    body = re.sub(
        r"\\begin\{figure\}(?:\[[^\]]*\])?\s*(?:\\centering\s*)?",
        lambda m: "\\begin{figure}[!htb]\n  \\centering\n  ",
        body,
    )
    body = re.sub(
        r"\\begin\{table\}(?:\[[^\]]*\])?\s*(?:\\centering\s*)?",
        lambda m: "\\begin{table}[!htb]\n  \\centering\n  ",
        body,
    )

    # Clamp over-wide graphics: anything wider than the text column shrinks
    # to a safe width instead of overflowing into the margin.
    def _clamp_width(m: re.Match) -> str:
        try:
            factor = float(m.group(1))
        except ValueError:
            return m.group(0)
        return "width=0.9\\linewidth" if factor > 0.92 else m.group(0)

    body = re.sub(
        r"width\s*=\s*([0-9]*\.?[0-9]+)\s*\\(?:textwidth|linewidth|columnwidth)",
        _clamp_width,
        body,
    )
    # Bare \includegraphics without options gets an explicit safe size.
    body = re.sub(
        r"\\includegraphics\{",
        lambda m: "\\includegraphics[width=0.85\\linewidth,keepaspectratio]{",
        body,
    )

    # Convert common Unicode characters that LLMs emit into LaTeX equivalents
    unicode_map = {
        "≈": r"$\approx$",
        "≤": r"$\le$",
        "≥": r"$\ge$",
        "≠": r"$\neq$",
        "×": r"$\times$",
        "±": r"$\pm$",
        "÷": r"$\div$",
        "∞": r"$\infty$",
        "→": r"$\to$",
        "←": r"$\leftarrow$",
        "↔": r"$\leftrightarrow$",
        "⇒": r"$\implies$",
        "⇐": r"$\impliedby$",
        "⇔": r"$\iff$",
        "µ": r"$\mu$",
        "•": r"\textbullet{}",
        "—": r"---",
        "–": r"--",
        "“": r"``",
        "”": r"''",
        "‘": r"`",
        "’": r"'",
        "…": r"\dots{}",
        "™": r"\texttrademark{}",
        "®": r"\textregistered{}",
        "©": r"\textcopyright{}",
        "°": r"$^\circ$",
    }
    for char, repl in unicode_map.items():
        body = body.replace(char, repl)

    # Bare \backslash in prose is not valid text-mode TeX.
    body = body.replace("\\backslash", "\\textbackslash{}")

    # Truncate over-long titles in the running header (TOC keeps full title).
    body = ensure_short_chapter_marks(body)

    return body.strip() + "\n"


# ---------------------------------------------------------------------------
# Guaranteed visuals: insert manifest figures the writer left out
# ---------------------------------------------------------------------------

_ANY_SECTION_RE = re.compile(r"^\\section\*?\{([^}]*)\}", re.MULTILINE)


def _figure_block(filename: str, caption: str, label: str, width: str) -> str:
    return (
        "\\begin{figure}[!htbp]\n\\centering\n"
        f"\\includegraphics[width={width}\\linewidth]{{{filename}}}\n"
        f"\\caption{{{tex_escape(caption)}}}\n\\label{{{label}}}\n\\end{{figure}}\n"
    )


def _section_end(tex: str, title: str) -> int | None:
    """Offset just before the section after the one titled `title` (or None)."""
    want = re.sub(r"\W+", " ", title).strip().lower()
    if not want:
        return None
    matches = list(_ANY_SECTION_RE.finditer(tex))
    for i, m in enumerate(matches):
        if re.sub(r"\W+", " ", m.group(1)).strip().lower() == want:
            return matches[i + 1].start() if i + 1 < len(matches) else len(tex)
    return None


def _insert_before_section(tex: str, before_titles: list[str], block: str) -> str:
    for t in before_titles:
        m = re.search(r"^\\section\*?\{" + re.escape(t) + r"\}", tex, re.MULTILINE)
        if m:
            return tex[: m.start()] + block + "\n" + tex[m.start() :]
    return tex.rstrip() + "\n\n" + block


def ensure_figures_included(tex: str, assets) -> tuple[str, list[str]]:
    """Insert every chart/diagram/illustration from the assets manifest that
    the chapter does not include yet. Returns (tex, inserted_labels).

    Diagrams go to "Conceptual Models", charts to "Key Data and Comparisons"
    (both created before "Key Takeaways" when missing), illustrations to the
    end of the concept section they belong to. Frames stay the writer's call.
    """
    if assets is None or not tex.strip():
        return tex, []
    referenced = set(find_referenced_files(tex))
    inserted: list[str] = []
    tail_sections = ["Key Takeaways", "Glossary", "Exercises", "Solutions"]
    for fig in assets.figures:
        if fig.kind not in ("chart", "diagram", "illustration") or fig.filename in referenced:
            continue
        width = "0.6" if fig.kind == "illustration" else "0.9"
        lead = f"Figure~\\ref{{{fig.label}}} illustrates this visually.\n"
        block = lead + _figure_block(fig.filename, fig.caption, fig.label, width)
        target = {
            "diagram": "Conceptual Models",
            "chart": "Key Data and Comparisons",
        }.get(fig.kind, "")
        pos = _section_end(tex, fig.section) if fig.kind == "illustration" else None
        if pos is None and fig.kind == "illustration":
            target = "Conceptual Models"
        if pos is None:
            pos = _section_end(tex, target)
        if pos is not None:
            tex = tex[:pos].rstrip() + "\n\n" + block + "\n" + tex[pos:]
        else:
            tex = _insert_before_section(
                tex, tail_sections, f"\\section*{{{target}}}\n" + block
            )
        referenced.add(fig.filename)
        inserted.append(fig.label)
    return tex, inserted
