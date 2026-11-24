"""Deterministic chapter verification — the machine gate of the QA loop.

Everything here is cheap, reproducible and quotable: structure, balanced
environments, asset references, label/ref integrity, numbers that do not
appear in the analysis or transcript (hallucination guard), equations and
concepts the analysis promised, and finally a real pdflatex pass. The
resulting `QAReport` drives the loop: a clean report exits it, a dirty one
becomes the refiner's exact to-do list without spending an LLM call on the
critic.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bookforge.config import Settings
from bookforge.schemas import (
    AssetsManifest,
    ChapterAnalysis,
    QADefect,
    QAReport,
    VideoRecord,
)
from bookforge.tools import latex as latex_tools
from bookforge.tools.workspace import Workspace

logger = logging.getLogger(__name__)

# Sections every chapter must contain, in the order the writer is told to use.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Overview",
    "Learning Objectives",
    "Key Takeaways",
    "Glossary",
    "Exercises",
    "Solutions",
)
DATA_SECTION = "Key Data and Comparisons"
MODELS_SECTION = "Conceptual Models"

MIN_PROSE_WORDS = 500
# Small integers are routinely spelled out in speech ("three steps") and used
# structurally (step 2, part 1); only larger / fractional values are audited.
NUMBER_AUDIT_MIN = 13
MAX_REPORTED_NUMBERS = 15

_SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
_NUMBERED_SECTION_RE = re.compile(r"\\section(?!\*)\{([^}]*)\}")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")


def _has_section(tex: str, title: str) -> bool:
    pattern = re.compile(
        r"\\section\*?\{\s*" + re.escape(title) + r"\s*\}", re.IGNORECASE
    )
    return bool(pattern.search(tex))


def _significant_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _concept_covered(heading: str, tex_lower: str, section_titles: list[str]) -> bool:
    escaped = latex_tools.tex_escape(heading).lower()
    if escaped and escaped in tex_lower:
        return True
    words = _significant_words(heading)
    if not words:
        return True
    titles_words = _significant_words(" ".join(section_titles))
    hits = sum(1 for w in words if w in titles_words)
    if hits / len(words) >= 0.6:
        return True
    body_hits = sum(1 for w in words if w in tex_lower)
    return body_hits / len(words) >= 0.8


# ---------------------------------------------------------------------------
# Static checks (no subprocess, microseconds)
# ---------------------------------------------------------------------------


def run_static_checks(
    tex: str,
    analysis: ChapterAnalysis | None,
    assets: AssetsManifest | None,
    transcript: str,
    ws_root: Path | None,
    *,
    strict_numbers: bool = True,
) -> tuple[list[QADefect], dict[str, float]]:
    """Return (defects, metrics) for a sanitized chapter body."""
    defects: list[QADefect] = []
    metrics: dict[str, float] = {}

    if not tex.strip():
        return (
            [QADefect(severity="critical", code="empty_chapter", message="chapter body is empty")],
            metrics,
        )

    # -- structure -----------------------------------------------------------
    if not re.search(r"\\chapter\{", tex):
        defects.append(
            QADefect(
                severity="critical",
                code="missing_chapter_command",
                message="no \\chapter{...} command found at the top of the body",
                fix="start the body with the chapter command and its ch:NN-slug label",
            )
        )
    required = list(REQUIRED_SECTIONS)
    if assets and (assets.tables or any(f.kind == "chart" for f in assets.figures)):
        required.append(DATA_SECTION)
    if assets is not None:
        if any(f.kind == "diagram" for f in assets.figures):
            required.append(MODELS_SECTION)
    elif analysis and analysis.diagrams:
        required.append(MODELS_SECTION)
    for title in required:
        if not _has_section(tex, title):
            defects.append(
                QADefect(
                    severity="major",
                    code="missing_section",
                    message=f"required section '{title}' is missing",
                    fix=f"add an unnumbered section titled exactly '{title}' in the required order",
                    location=title,
                )
            )

    # -- LaTeX well-formedness ----------------------------------------------
    for problem in latex_tools.check_environment_balance(tex):
        defects.append(
            QADefect(
                severity="critical",
                code="unbalanced_latex",
                message=problem,
                fix="close every environment/brace you open; do not leave partial environments",
            )
        )
    if re.search(r"\\(?:documentclass|usepackage)|\\begin\{document\}", tex):
        defects.append(
            QADefect(
                severity="major",
                code="preamble_in_body",
                message="chapter body contains preamble commands (documentclass/usepackage/document)",
                fix="remove all preamble material; the book preamble is shared",
            )
        )

    # -- asset references -----------------------------------------------------
    referenced = latex_tools.find_referenced_files(tex)
    if ws_root is not None:
        for ref in referenced:
            if not (Path(ws_root) / ref).exists():
                defects.append(
                    QADefect(
                        severity="critical",
                        code="missing_file",
                        message=f"referenced file does not exist on disk: {ref}",
                        fix="reference only filenames listed verbatim in the assets manifest, or remove this include",
                        location=ref,
                    )
                )
    manifest_labels: set[str] = set()
    if assets is not None:
        allowed = {f.filename for f in assets.figures} | {t.tex_file for t in assets.tables}
        for ref in referenced:
            if ref not in allowed:
                defects.append(
                    QADefect(
                        severity="critical",
                        code="unknown_asset",
                        message=f"'{ref}' is not in the assets manifest",
                        fix="use exactly one of the manifest filenames or delete the reference",
                        location=ref,
                    )
                )
        for table in assets.tables:
            manifest_labels.add(table.label)
            if table.tex_file not in referenced:
                defects.append(
                    QADefect(
                        severity="major",
                        code="unused_table",
                        message=f"table fragment '{table.tex_file}' ({table.caption}) is never input",
                        fix=f"add \\input with path {table.tex_file} inside '{DATA_SECTION}' and reference Table~\\ref with label {table.label} in the prose",
                        location=DATA_SECTION,
                    )
                )
        for fig in assets.figures:
            if fig.kind in ("diagram", "illustration") and fig.filename not in referenced:
                # Minor: the orchestrator inserts any visual the writer left
                # out (see latex.ensure_figures_included), so never loop on it.
                defects.append(
                    QADefect(
                        severity="minor",
                        code=f"unused_{fig.kind}",
                        message=f"{fig.kind} '{fig.filename}' ({fig.caption}) is never included",
                        fix=f"include it in a figure environment with label {fig.label}",
                        location=MODELS_SECTION if fig.kind == "diagram" else fig.section,
                    )
                )
            if fig.kind == "chart" and fig.filename not in referenced:
                defects.append(
                    QADefect(
                        severity="major",
                        code="unused_chart",
                        message=f"chart '{fig.filename}' ({fig.caption}) is never included",
                        fix=f"include it in a figure environment with label {fig.label} inside '{DATA_SECTION}' and reference it in the prose",
                        location=DATA_SECTION,
                    )
                )
        metrics["figures_used"] = float(
            sum(1 for f in assets.figures if f.filename in referenced)
        )
        metrics["tables_used"] = float(
            sum(1 for t in assets.tables if t.tex_file in referenced)
        )

    # -- labels / refs --------------------------------------------------------
    labels_in_tex = latex_tools.find_labels(tex)
    all_labels = labels_in_tex | manifest_labels
    refs = latex_tools.find_refs(tex)
    metrics["refs"] = float(len(refs))
    for label in sorted({r for r in refs if r not in all_labels}):
        defects.append(
            QADefect(
                severity="major",
                code="undefined_ref",
                message=f"\\ref{{{label}}} has no matching \\label (would print '??')",
                fix="reference an existing label or add the missing \\label to the float",
                location=label,
            )
        )
    counts: dict[str, int] = {}
    for label in re.findall(r"\\label\{([^}]+)\}", tex):
        counts[label] = counts.get(label, 0) + 1
    for label, n in counts.items():
        if n > 1 or label in manifest_labels:
            defects.append(
                QADefect(
                    severity="major",
                    code="duplicate_label",
                    message=(
                        f"label '{label}' is defined {n} time(s) in the chapter"
                        + (" and again inside a table fragment" if label in manifest_labels else "")
                    ),
                    fix="keep exactly one \\label per float; table fragments already carry their label",
                    location=label,
                )
            )

    # -- faithfulness: numbers must come from analysis/transcript ------------
    prose = latex_tools.extract_prose(tex)
    words = _WORD_RE.findall(prose)
    metrics["prose_words"] = float(len(words))
    if len(words) < MIN_PROSE_WORDS:
        defects.append(
            QADefect(
                severity="major",
                code="chapter_too_short",
                message=f"only ~{len(words)} words of prose; target is 1200-2500",
                fix="expand each concept section with substance from the analysis (no invented facts)",
            )
        )
    source_text = (transcript or "") + "\n" + (analysis.model_dump_json() if analysis else "")
    supported = {latex_tools.normalize_number(n) for n in latex_tools.extract_numbers(source_text)}
    unsupported: list[str] = []
    seen: set[str] = set()
    for token in latex_tools.extract_numbers(prose):
        norm = latex_tools.normalize_number(token)
        if norm in seen:
            continue
        try:
            if float(norm) < NUMBER_AUDIT_MIN and "." not in norm:
                continue
        except ValueError:
            continue
        seen.add(norm)
        if norm not in supported:
            unsupported.append(token)
    metrics["unsupported_numbers"] = float(len(unsupported))
    if unsupported:
        shown = ", ".join(unsupported[:MAX_REPORTED_NUMBERS])
        more = "" if len(unsupported) <= MAX_REPORTED_NUMBERS else f" (+{len(unsupported) - MAX_REPORTED_NUMBERS} more)"
        defects.append(
            QADefect(
                severity="major" if strict_numbers else "minor",
                code="unsupported_number",
                message=f"numbers in the prose that appear nowhere in the analysis or transcript: {shown}{more}",
                fix="replace each with the exact value from the analysis/transcript, or remove the statistic",
            )
        )

    # -- coverage of what the analysis promised ------------------------------
    if analysis is not None:
        tex_lower = tex.lower()
        section_titles = _SECTION_RE.findall(tex)
        numbered = _NUMBERED_SECTION_RE.findall(tex)
        metrics["numbered_sections"] = float(len(numbered))
        for concept in analysis.concepts:
            if not _concept_covered(concept.heading, tex_lower, section_titles):
                defects.append(
                    QADefect(
                        severity="major",
                        code="missing_concept",
                        message=f"concept '{concept.heading}' from the analysis has no section",
                        fix=f"add a numbered section covering '{concept.heading}' using the analysis explanation and key points",
                        location=concept.heading,
                    )
                )
        norm_tex = latex_tools.normalize_math(tex)
        for concept in analysis.concepts:
            for eq in concept.equations:
                if eq.strip() and latex_tools.normalize_math(eq) not in norm_tex:
                    defects.append(
                        QADefect(
                            severity="major",
                            code="missing_equation",
                            message=f"equation from concept '{concept.heading}' is not typeset: {eq}",
                            fix="typeset it verbatim inside an equation environment in that concept's section",
                            location=concept.heading,
                        )
                    )
        n_tikz = len(re.findall(r"\\begin\{tikzpicture\}", tex))
        metrics["tikz_diagrams"] = float(n_tikz)
        if assets is None and len(analysis.diagrams) > n_tikz:
            defects.append(
                QADefect(
                    severity="major",
                    code="missing_diagram",
                    message=f"analysis specifies {len(analysis.diagrams)} diagram(s) but only {n_tikz} tikzpicture environment(s) exist",
                    fix=f"render every diagram spec as a TikZ figure inside '{MODELS_SECTION}'",
                    location=MODELS_SECTION,
                )
            )
        if transcript:
            stamps = set(re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", transcript))
            bad = [
                ts
                for c in analysis.concepts
                for ts in c.evidence_timestamps
                if stamps and ts.strip() not in stamps
            ]
            if bad:
                defects.append(
                    QADefect(
                        severity="minor",
                        code="evidence_timestamp_unknown",
                        message=f"analysis cites transcript timestamps that do not exist: {', '.join(bad[:6])}",
                    )
                )

    metrics["defects"] = float(len(defects))
    metrics["blocking_defects"] = float(
        sum(1 for d in defects if d.severity in ("critical", "major"))
    )
    return defects, metrics


# ---------------------------------------------------------------------------
# Compile check
# ---------------------------------------------------------------------------


def compile_chapter_standalone(
    ws: Workspace, record: VideoRecord, tex: str
) -> tuple[bool, str]:
    """Wrap the body in the shared preamble and run one pdflatex pass with
    cwd = workspace root so asset paths resolve exactly as in the book."""
    qa_dir = ws.build_dir / f"qa_{record.video_id}"
    wrapper = latex_tools.chapter_wrapper(latex_tools.PREAMBLE, tex)
    Workspace.write_text(qa_dir / "main_qa.tex", wrapper)
    return latex_tools.compile_tex(qa_dir / "main_qa.tex", qa_dir, passes=1, cwd=ws.root)


# ---------------------------------------------------------------------------
# Full machine gate
# ---------------------------------------------------------------------------


def run_machine_checks(
    tex: str,
    analysis: ChapterAnalysis | None,
    assets: AssetsManifest | None,
    transcript: str,
    ws: Workspace | None,
    record: VideoRecord | None,
    settings: Settings,
    iteration: int = 0,
) -> QAReport:
    """Static checks, then (only if they are clean) a real pdflatex pass."""
    tex = latex_tools.sanitize_chapter_tex(tex)
    checks = ["structure", "latex_balance", "asset_refs", "labels", "numbers", "coverage"]
    defects, metrics = run_static_checks(
        tex,
        analysis,
        assets,
        transcript,
        ws.root if ws else None,
        strict_numbers=settings.qa_strict_numbers,
    )
    blocking = [d for d in defects if d.severity in ("critical", "major")]

    compile_ok: bool | None = None
    if settings.compile_latex and ws is not None and record is not None:
        if any(d.severity == "critical" for d in defects):
            # A compile cannot succeed with missing files / broken environments;
            # skip the expensive pass and let the refiner fix the cause first.
            compile_ok = False
        elif latex_tools.pdflatex_available():
            checks.append("pdflatex")
            ok, log = compile_chapter_standalone(ws, record, tex)
            compile_ok = ok
            if not ok:
                defects.append(
                    QADefect(
                        severity="critical",
                        code="compile_error",
                        message=f"pdflatex failed: {log[:1200]}",
                        fix="fix the LaTeX error at the quoted line; the callout environments notebox/keyinsight/formulabox and TikZ styles ARE defined",
                    )
                )
                blocking.append(defects[-1])
        else:
            logger.warning("pdflatex not on PATH; QA compile check skipped")

    metrics["defects"] = float(len(defects))
    metrics["blocking_defects"] = float(len(blocking))
    return QAReport(
        passed=not blocking and compile_ok is not False,
        compile_ok=compile_ok,
        defects=defects,
        checks_run=checks,
        metrics=metrics,
        iteration=iteration,
    )


def format_defects(defects: list[QADefect], header: str = "") -> str:
    """Numbered, severity-tagged defect list — the refiner's exact to-do list."""
    if not defects:
        return header.strip()
    lines = [header.strip()] if header.strip() else []
    for i, d in enumerate(defects, start=1):
        where = f" [{d.location}]" if d.location else ""
        fix = f" FIX: {d.fix}" if d.fix else ""
        lines.append(f"{i}. {d.severity.upper()} {d.code}{where}: {d.message}.{fix}")
    return "\n".join(lines)
