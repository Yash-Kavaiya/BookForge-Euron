"""Book branding: logos, running header/footer, title page and cover page.

All functions are pure string builders except :func:`prepare_logo` and
:func:`write_source_zip`, which touch the filesystem. Logos that cannot be
read are dropped with a warning; branding never breaks the book build.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from bookforge.tools.latex import _smart_quotes, tex_escape

logger = logging.getLogger(__name__)

LOGO_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
POSITIONS = ("title", "header", "footer")


@dataclass
class Branding:
    """Resolved branding; logo paths are workspace-root-relative."""

    header_left: str = ""
    header_right: str = ""
    footer_text: str = ""
    logos: dict[str, str] = field(default_factory=dict)  # position -> rel path

    @property
    def customized(self) -> bool:
        return bool(self.header_left or self.header_right or self.footer_text or self.logos)


def prepare_logo(src: str | Path, dest_dir: Path) -> Path | None:
    """Copy (and for SVG, convert to PDF) a logo into ``dest_dir``.

    Returns the usable file, or None (logged) if it is missing/unreadable.
    """
    src = Path(str(src)).expanduser()
    if not src.is_file():
        logger.warning("logo %s not found; skipped", src)
        return None
    suffix = src.suffix.lower()
    if suffix not in LOGO_SUFFIXES:
        logger.warning("logo %s has unsupported type %s; use PNG/JPG/PDF/SVG", src, suffix)
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = "logo_" + "".join(c if c.isalnum() else "_" for c in src.stem)[:40]
    if suffix == ".svg":
        dest = dest_dir / f"{stem}.pdf"
        if _svg_to_pdf(src, dest):
            return dest
        logger.warning("logo %s: SVG conversion failed (install librsvg2-bin or cairosvg)", src)
        return None
    dest = dest_dir / f"{stem}{'.jpg' if suffix == '.jpeg' else suffix}"
    try:
        if suffix == ".pdf":
            if not src.read_bytes()[:5].startswith(b"%PDF"):
                raise ValueError("not a PDF")
        else:
            from PIL import Image

            with Image.open(src) as img:
                img.verify()
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
    except Exception as exc:  # noqa: BLE001
        logger.warning("logo %s unreadable (%s); skipped", src, exc)
        return None
    return dest


def _svg_to_pdf(src: Path, dest: Path) -> bool:
    if shutil.which("rsvg-convert"):
        proc = subprocess.run(
            ["rsvg-convert", "-f", "pdf", "-o", str(dest), str(src)],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size:
            return True
    try:
        import cairosvg  # type: ignore[import-not-found]

        cairosvg.svg2pdf(url=str(src), write_to=str(dest))
        return dest.exists() and dest.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def resolve_branding(settings, ws_root: Path) -> Branding:
    """Build :class:`Branding` from settings, copying logos into the workspace."""
    positions = {p.strip().lower() for p in (settings.logo_positions or "").split(",") if p.strip()}
    per_position = {
        "title": settings.logo_title,
        "header": settings.logo_header,
        "footer": settings.logo_footer,
    }
    logos: dict[str, str] = {}
    dest_dir = Path(ws_root) / "branding"
    for pos in POSITIONS:
        src = per_position[pos] or (settings.logo if pos in positions else "")
        if not src:
            continue
        logo = prepare_logo(src, dest_dir)
        if logo is not None:
            logos[pos] = logo.relative_to(ws_root).as_posix()
    return Branding(
        header_left=settings.header_left,
        header_right=settings.header_right,
        footer_text=settings.footer_text,
        logos=logos,
    )


def header_footer_tex(b: Branding) -> str:
    """fancyhdr overrides (placed after the shared preamble in main.tex)."""
    if not b.customized:
        return ""

    def logo(pos: str, height: str) -> str:
        rel = b.logos.get(pos)
        if not rel:
            return ""
        return f"\\raisebox{{-0.3\\height}}{{\\includegraphics[height={height},width=!,keepaspectratio]{{{rel}}}}}\\hspace{{6pt}}"

    left = tex_escape(b.header_left) if b.header_left else r"\nouppercase{\leftmark}"
    right = tex_escape(b.header_right)
    head_l = f"{logo('header', '14pt')}{{\\small\\itshape {left}}}"
    head_r = f"{{\\small\\itshape {right}}}" if right else ""
    foot_l_text = (
        tex_escape(b.footer_text) if b.footer_text else r"\textcolor{darkslate}{\BFchannelName}"
    )
    foot_l = f"{logo('footer', '10pt')}{{\\footnotesize {foot_l_text}}}"
    foot = (
        f"  \\fancyfoot[L]{{{foot_l}}}%\n"
        "  \\fancyfoot[C]{\\small\\thepage}%\n"
        "  \\fancyfoot[R]{\\footnotesize\\BFvideoFooter}%\n"
    )
    return (
        "% ---- Branding: custom running header/footer ----\n"
        "\\setlength{\\headheight}{22pt}\n"
        "\\fancypagestyle{fancy}{%\n"
        "  \\fancyhf{}%\n"
        f"  \\fancyhead[L]{{{head_l}}}%\n"
        f"  \\fancyhead[R]{{{head_r}}}%\n"
        f"{foot}"
        "  \\renewcommand{\\headrulewidth}{0.4pt}%\n"
        "  \\renewcommand{\\footrulewidth}{0.4pt}%\n"
        "}\n"
        "\\fancypagestyle{plain}{%\n"
        "  \\fancyhf{}%\n"
        f"{foot}"
        "  \\renewcommand{\\headrulewidth}{0pt}%\n"
        "  \\renewcommand{\\footrulewidth}{0.4pt}%\n"
        "}\n"
        "\\pagestyle{fancy}\n"
    )


def title_page_tex(
    title: str, subtitle: str, author: str, logo_rel: str = "", credit: str = ""
) -> str:
    """A designed title page (replaces \\maketitle)."""
    parts = ["\\begin{titlepage}", "\\centering", "\\vspace*{1.2cm}"]
    if logo_rel:
        parts.append(
            f"\\includegraphics[height=2.6cm,width=0.6\\linewidth,keepaspectratio]{{{logo_rel}}}\\par\\vspace{{1.4cm}}"
        )
    else:
        parts.append("\\vspace{2.5cm}")
    parts.append(
        "{\\color{primaryblue}\\rule{0.7\\linewidth}{1.2pt}}\\par\\vspace{0.8cm}"
    )
    parts.append(
        f"{{\\Huge\\bfseries\\color{{darkslate}} {_smart_quotes(tex_escape(title))}\\par}}\\vspace{{0.6cm}}"
    )
    if subtitle:
        parts.append(
            f"{{\\Large\\itshape\\color{{accentindigo}} {_smart_quotes(tex_escape(subtitle))}\\par}}\\vspace{{0.6cm}}"
        )
    parts.append("{\\color{primaryblue}\\rule{0.7\\linewidth}{1.2pt}}\\par\\vspace{1.6cm}")
    if author:
        parts.append(f"{{\\large {tex_escape(author)}\\par}}\\vspace{{0.4cm}}")
    parts.append("{\\normalsize \\today\\par}")
    parts.append("\\vfill")
    if credit:
        parts.append(f"{{\\footnotesize\\color{{darkslate}} {tex_escape(credit)}\\par}}")
    parts.append("\\end{titlepage}")
    return "\n".join(parts) + "\n"


def cover_page_tex(cover_rel: str, title: str, subtitle: str = "", author: str = "") -> str:
    """Full-bleed cover as the very first page (dark page, art, title band)."""
    sub = (
        f"\\\\[6pt]{{\\Large\\itshape {_smart_quotes(tex_escape(subtitle))}}}" if subtitle else ""
    )
    by = f"{{\\large {tex_escape(author)}}}" if author else ""
    return (
        "\\begin{titlepage}\n"
        "\\thispagestyle{empty}\n"
        "\\begin{tikzpicture}[remember picture,overlay]\n"
        "\\fill[darkslate] (current page.south west) rectangle (current page.north east);\n"
        "\\node[anchor=center,inner sep=0] at ([yshift=-0.4in]current page.center)"
        f" {{\\includegraphics[width=\\paperwidth,height=0.72\\paperheight,keepaspectratio]{{{cover_rel}}}}};\n"
        "\\node[anchor=north,text width=0.84\\paperwidth,align=center,text=white]"
        " at ([yshift=-0.55in]current page.north)"
        f" {{{{\\fontsize{{30}}{{36}}\\selectfont\\bfseries {_smart_quotes(tex_escape(title))}}}{sub}}};\n"
        "\\node[anchor=south,text=white,align=center] at ([yshift=0.45in]current page.south)"
        f" {{{by}}};\n"
        "\\end{tikzpicture}\n"
        "\\end{titlepage}\n"
    )


def write_source_zip(ws_root: Path, dest_zip: Path, chapter_rel_dirs: list[str]) -> Path:
    """Zip the LaTeX sources (main/preamble, chapters, figures, tables,
    branding) so the book can be rebuilt with ``pdflatex main.tex`` twice."""
    ws_root = Path(ws_root)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_suffix(".tmp.zip")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ("main.tex", "preamble.tex"):
            if (ws_root / name).exists():
                zf.write(ws_root / name, name)
        for rel in ["branding", *chapter_rel_dirs]:
            base = ws_root / rel
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                if path.name in ("metrics.json", "analysis.json") or path.suffix in (".log", ".aux"):
                    continue
                if path.name.startswith("qa_") or path.name.endswith(".tmp.pdf"):
                    continue
                zf.write(path, path.relative_to(ws_root).as_posix())
    tmp.replace(dest_zip)
    return dest_zip
