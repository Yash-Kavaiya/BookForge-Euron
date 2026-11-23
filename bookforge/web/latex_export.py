"""Convert BookForge Markdown chapters into a standalone LaTeX book.

Pure Python (no pandoc / pdflatex needed), so it runs on serverless hosts.
Compile the result locally with ``pdflatex book.tex`` (run twice for the TOC).
"""

from __future__ import annotations

import re

_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_SPECIAL_RE = re.compile(r"[\\&%$#_{}~^]")


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return _SPECIAL_RE.sub(lambda m: _SPECIALS[m.group(0)], text)


def inline_markdown_to_latex(text: str) -> str:
    """Convert inline Markdown (code, bold, italic, links) to LaTeX."""
    tokens: list[str] = []

    def stash(value: str) -> str:
        tokens.append(value)
        return f"\x00{len(tokens) - 1}\x00"

    text = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{" + escape_latex(m.group(1)) + "}"), text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        lambda m: stash(r"\href{" + m.group(2).replace("%", r"\%").replace("#", r"\#") + "}{"
                        + escape_latex(m.group(1)) + "}"),
        text,
    )
    text = escape_latex(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"__(.+?)__", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", text)
    text = re.sub(r"\\_(?!\s)(.+?)(?<!\s)\\_", r"\\emph{\1}", text)
    text = text.replace("\u2014", "---").replace("\u2013", "--")
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], text)


def markdown_to_latex(markdown: str) -> str:
    """Convert a chapter's block-level Markdown to LaTeX."""
    out: list[str] = []
    list_stack: list[str] = []
    in_code = False
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(inline_markdown_to_latex(" ".join(s.strip() for s in paragraph)))
            out.append("")
            paragraph.clear()

    def close_lists() -> None:
        if not list_stack:
            return
        while list_stack:
            out.append(r"\end{" + list_stack.pop() + "}")
        out.append("")

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            flush_paragraph()
            close_lists()
            out.append(r"\end{verbatim}" if in_code else r"\begin{verbatim}")
            in_code = not in_code
            continue
        if in_code:
            out.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
        numbered = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            close_lists()
            level = len(heading.group(1))
            cmd = {1: "section", 2: "section", 3: "subsection"}.get(level, "subsubsection")
            out.append(f"\\{cmd}{{{inline_markdown_to_latex(heading.group(2).strip('# '))}}}")
            out.append("")
        elif bullet or numbered:
            flush_paragraph()
            env = "itemize" if bullet else "enumerate"
            if list_stack and list_stack[-1] != env:
                close_lists()
            if not list_stack:
                out.append(r"\begin{" + env + "}")
                list_stack.append(env)
            item = (bullet or numbered).group(1)  # type: ignore[union-attr]
            out.append(r"  \item " + inline_markdown_to_latex(item))
        elif stripped.startswith(">"):
            flush_paragraph()
            close_lists()
            out.append(r"\begin{quote}" + inline_markdown_to_latex(stripped.lstrip("> ")) + r"\end{quote}")
        elif re.match(r"^(-{3,}|\*{3,})$", stripped):
            flush_paragraph()
            close_lists()
            out.append(r"\medskip\hrule\medskip")
        elif stripped.startswith("|"):
            flush_paragraph()
            close_lists()
            if not re.match(r"^\|[\s:|-]+\|$", stripped):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                out.append(r"\noindent " + r" \quad\textbar\quad ".join(
                    inline_markdown_to_latex(c) for c in cells) + r"\par")
        else:
            if list_stack:
                close_lists()
            paragraph.append(stripped)
    flush_paragraph()
    close_lists()
    if in_code:
        out.append(r"\end{verbatim}")
    return "\n".join(out).strip() + "\n"


PREAMBLE = r"""\documentclass[11pt,openany]{book}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setcounter{secnumdepth}{1}
"""


def markdown_book_to_latex(
    title: str, subtitle: str, chapters: list[tuple[str, str]]
) -> str:
    """Build a complete ``book.tex`` from (chapter title, chapter Markdown) pairs."""
    parts = [PREAMBLE]
    parts.append(r"\title{" + inline_markdown_to_latex(title) + (
        r"\\[0.5em]\large " + inline_markdown_to_latex(subtitle) if subtitle else "") + "}")
    parts.append(r"\author{BookForge \\ \small Generated with the Euron EURI API}")
    parts.append(r"\date{\today}")
    parts.append("\\begin{document}\n\\maketitle\n\\tableofcontents\n")
    for chapter_title, body in chapters:
        parts.append(r"\chapter{" + inline_markdown_to_latex(chapter_title) + "}")
        parts.append(markdown_to_latex(body))
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"
