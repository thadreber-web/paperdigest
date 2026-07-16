from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import mermaid
from .extract import Figure, Paper
from .llm import Backend, LLMError, VisionUnsupportedError, complete_with_retry, repair_json, run_tasks, strip_fences


@dataclass
class ConceptNote:
    title: str
    body_md: str
    section: str


@dataclass
class FigureNote:
    caption: str
    body_md: str
    image_path: Path
    concept_title: str | None


@dataclass
class Digest:
    paper: Paper
    tldr: str
    why_it_matters: str
    concepts: list[ConceptNote]
    jargon: list[str]           # every term of art in the paper (for linking)
    glossary: dict[str, str]    # definitions for terms NOT already in the vault
    self_test: list[str]
    model: str
    level: str
    figures: list[FigureNote] = field(default_factory=list)


LEVEL_GUIDANCE = {
    "beginner": "Assume no ML background at all. Use everyday analogies. No unexplained jargon.",
    "intermediate": (
        "Assume a technically capable reader (writes code, knows basic statistics) who is not "
        "fluent in ML paper vocabulary. Plain English; define terms of art on first use."
    ),
    "advanced": (
        "Assume an ML practitioner. Be precise and compact; standard terminology is fine, "
        "but still explain the paper's novel jargon."
    ),
}

_OUTLINE_SYSTEM = """\
You are an expert teacher who breaks AI/ML research papers into teachable chunks for self-learners.
Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"tldr": "<2-3 plain-English sentences>",
 "why_it_matters": "<2-3 sentences on why this paper matters>",
 "concepts": [{"title": "<concept name>", "section": "<paper section heading it mainly comes from>"}],
 "jargon": ["<term>", ...],
 "self_test": ["<question>", ...]}
Rules: pick 4-8 concepts that each deserve their own explainer note.
"jargon" lists every term of art a newcomer would need defined (10-25 terms, lowercase).
"self_test" is 3-5 questions that check understanding of the whole paper."""

_CONCEPT_SYSTEM = """\
You write ONE markdown explainer note about a single concept from a research paper, for a self-learner.
Reader level: {guidance}
Rules:
- Plain English. Strip jargon or define it inline in parentheses.
- Use an analogy if it genuinely helps.
{diagram_guidance}
- If the concept involves key equations, show them in LaTeX ($...$ or $$...$$) followed by a line-by-line plain-English walkthrough.
- Cite where in the paper the idea comes from (e.g. "from §3.2").
Output ONLY the markdown body: no top-level heading, no frontmatter."""

DIAGRAM_GUIDANCE = {
    "ascii": (
        "- If the concept is structural (architecture, data flow, training loop, algorithm), "
        "include an ASCII diagram in a fenced code block."
    ),
    "mermaid": (
        "- If the concept is structural (architecture, data flow, training loop, algorithm), "
        "include a Mermaid diagram in a ```mermaid fenced code block "
        "(flowchart TD or LR; simple node/edge syntax only; quote any node label "
        "containing parentheses, math, or special characters)."
    ),
}

_FIGURE_SYSTEM = """\
You explain figures from research papers to a {level}-level learner. Be concise: at most 250 words. \
Explain what the figure shows and why it matters for the paper. If any detail is unclear in the image, \
say so instead of guessing."""

_GLOSSARY_SYSTEM = """\
You define research-paper jargon in plain English for a {level}-level reader.
Respond with ONLY valid JSON (no markdown fences): {{"terms": {{"<term>": "<1-2 sentence plain-English definition>"}}}}.
Every requested term must appear as a key."""


def _default_progress(msg: str) -> None:
    print(msg, file=sys.stderr)


def _call_json(backend: Backend, system: str, user: str) -> dict:
    raw = complete_with_retry(backend, system, user, json_mode=True)
    try:
        return json.loads(strip_fences(raw))
    except json.JSONDecodeError:
        repaired = repair_json(backend, raw)
        try:
            return json.loads(strip_fences(repaired))
        except json.JSONDecodeError as e:
            raise LLMError(
                f"model returned unparseable JSON even after a repair attempt: {e}\n---\n{raw[:2000]}"
            ) from e


def _paper_body(paper: Paper, max_chars: int, progress: Callable[[str], None]) -> str:
    parts = [f"## {s.title}\n{s.text}" for s in paper.sections]
    body = "\n\n".join(parts)
    if len(body) <= max_chars:
        return body
    progress(f"Warning: paper is {len(body)} chars; trimming sections to fit {max_chars}-char budget.")
    ratio = max_chars / len(body)
    trimmed = [f"## {s.title}\n{s.text[: max(200, int(len(s.text) * ratio))]}" for s in paper.sections]
    return "\n\n".join(trimmed)


_SECTION_NUM_RE = re.compile(r"^\d+(\.\d+)*\.?\s*")


def _strip_section_num(title: str) -> str:
    return _SECTION_NUM_RE.sub("", title).strip()


def _find_section_text(paper: Paper, section_title: str, fallback: str) -> str:
    want = section_title.strip().lower()
    if not want:
        return fallback
    haves = [(s, s.title.strip().lower()) for s in paper.sections]
    stripped_want = _strip_section_num(want)
    if stripped_want:  # tier 1: exact match ignoring leading section numbers
        for s, have in haves:
            if _strip_section_num(have) == stripped_want:
                return s.text
    for s, have in haves:  # tier 2: exact match, numbers included
        if have == want:
            return s.text
    best, best_overlap = None, -1  # tier 3: bidirectional substring, longest overlap wins
    for s, have in haves:
        if (want in have or have in want) and min(len(want), len(have)) > best_overlap:
            best, best_overlap = s, min(len(want), len(have))
    return best.text if best else fallback


def _figure_concept_title(figure_section: str | None, concepts: list[ConceptNote]) -> str | None:
    """Match a figure's section against concept sections; unmatched figures go to the overview."""
    if not figure_section:
        return None
    want = figure_section.strip().lower()
    if not want:
        return None
    stripped_want = _strip_section_num(want)
    if stripped_want:  # tier 1: exact match ignoring leading section numbers
        for c in concepts:
            if _strip_section_num(c.section.strip().lower()) == stripped_want:
                return c.title
    for c in concepts:  # tier 2: exact match, numbers included
        if c.section.strip().lower() == want:
            return c.title
    return None


def _caption_snippet(caption: str, words: int = 6) -> str:
    parts = caption.split()
    snippet = " ".join(parts[:words])
    return snippet + ("..." if len(parts) > words else "")


def _explain_figure(
    i: int,
    fig: Figure,
    image_path: Path,
    total: int,
    backend: Backend,
    fig_system: str,
    paper: Paper,
    tldr: str,
    concepts: list[ConceptNote],
    progress: Callable[[str], None],
) -> FigureNote:
    progress(f"Explaining figure {i}/{total}: {_caption_snippet(fig.caption)}")
    image_bytes = image_path.read_bytes()
    user = (
        f"PAPER TITLE: {paper.title}\nPAPER TLDR: {tldr}\nFIGURE CAPTION: {fig.caption}\n\n"
        "Explain what this figure shows in this paper and why it matters."
    )
    body = complete_with_retry(backend, fig_system, user, images=[image_bytes]).strip()
    return FigureNote(
        caption=fig.caption,
        body_md=body,
        image_path=image_path,
        concept_title=_figure_concept_title(fig.section, concepts),
    )


def _explain_figures(
    paper: Paper,
    figure_paths: dict[str, Path],
    backend: Backend,
    fig_system: str,
    tldr: str,
    concepts: list[ConceptNote],
    progress: Callable[[str], None],
    workers: int,
) -> list[FigureNote]:
    figs = [f for f in paper.figures if f.src in figure_paths]
    if not figs:
        return []
    total = len(figs)
    first, rest = figs[0], figs[1:]

    def _explain(i: int, fig: Figure) -> FigureNote:
        return _explain_figure(
            i, fig, figure_paths[fig.src], total, backend, fig_system, paper, tldr, concepts, progress
        )

    try:
        first_note: FigureNote | None = _explain(1, first)
    except VisionUnsupportedError:
        # Probe serially with the first figure only: if the backend rejects images at all,
        # every subsequent call would fail identically, so drop all figures after one warning
        # rather than firing (and warning about) N doomed calls.
        progress(f"Warning: backend has no vision support; skipping {total} figures.")
        return []
    except LLMError as e:
        progress(f"Warning: could not explain figure 1 ({_caption_snippet(first.caption)}): {e}; skipping.")
        first_note = None

    def _explain_or_skip(numbered: tuple[int, Figure]) -> FigureNote | None:
        i, fig = numbered
        try:
            return _explain(i, fig)
        except LLMError as e:
            progress(f"Warning: could not explain figure {i} ({_caption_snippet(fig.caption)}): {e}; skipping.")
            return None

    rest_notes = run_tasks(list(enumerate(rest, start=2)), _explain_or_skip, workers=workers) if rest else []
    notes = ([first_note] if first_note is not None else []) + [n for n in rest_notes if n is not None]
    return notes


def build_digest(
    paper: Paper,
    backend: Backend,
    level: str,
    existing_terms: set[str],
    max_chars: int,
    progress: Callable[[str], None] = _default_progress,
    diagram: str = "mermaid",
    workers: int = 1,
    figure_paths: dict[str, Path] | None = None,
) -> Digest:
    if level not in LEVEL_GUIDANCE:
        raise ValueError(f"level must be one of {tuple(LEVEL_GUIDANCE)}, got {level!r}")
    if diagram not in DIAGRAM_GUIDANCE:
        raise ValueError(f"diagram must be one of {tuple(DIAGRAM_GUIDANCE)}, got {diagram!r}")
    guidance = LEVEL_GUIDANCE[level]
    concept_system = _CONCEPT_SYSTEM.format(guidance=guidance, diagram_guidance=DIAGRAM_GUIDANCE[diagram])
    body = _paper_body(paper, max_chars, progress)

    progress(f"Outlining '{paper.title}'...")
    outline = _call_json(
        backend,
        _OUTLINE_SYSTEM,
        f"READER LEVEL: {level} — {guidance}\n\nPAPER TITLE: {paper.title}\n\n"
        f"ABSTRACT: {paper.abstract}\n\nFULL TEXT:\n{body}",
    )
    for key in ("tldr", "why_it_matters", "concepts", "jargon", "self_test"):
        if key not in outline:
            raise LLMError(f"outline response is missing the {key!r} field")

    total = len(outline["concepts"])
    for c in outline["concepts"]:
        if "title" not in c:
            raise LLMError(f"outline concept entry is missing 'title': {c!r}")

    def _explain(numbered: tuple[int, dict]) -> ConceptNote:
        i, c = numbered
        title = c["title"]
        section = c.get("section", "")
        progress(f"Explaining concept {i}/{total}: {title}")
        section_text = _find_section_text(paper, section, fallback=body[:20_000])
        md = complete_with_retry(
            backend,
            concept_system,
            f"PAPER: {paper.title}\nPAPER TLDR: {outline['tldr']}\n"
            f"CONCEPT TO EXPLAIN: {title}\nSOURCE SECTION: {section}\n\n"
            f"ABSTRACT: {paper.abstract}\n\nSECTION TEXT:\n{section_text}",
        ).strip()
        if diagram == "mermaid":
            md = mermaid.sanitize_markdown(md)
            for block in mermaid.mermaid_blocks(md):
                if mermaid.validate(block) is False:  # None = no local parser; skip silently
                    progress(f"Warning: mermaid diagram in '{title}' failed parse validation; kept as-is.")
        return ConceptNote(title=title, body_md=md, section=section)

    concepts = run_tasks(list(enumerate(outline["concepts"], 1)), _explain, workers=workers)

    figures: list[FigureNote] = []
    if figure_paths:
        fig_system = _FIGURE_SYSTEM.format(level=level)
        figures = _explain_figures(
            paper, figure_paths, backend, fig_system, str(outline["tldr"]), concepts, progress, workers
        )

    jargon = [str(t) for t in outline["jargon"]]
    new_terms = [t for t in jargon if t.lower() not in existing_terms]
    glossary: dict[str, str] = {}
    if new_terms:
        progress(f"Defining {len(new_terms)} glossary terms...")
        g = _call_json(
            backend,
            _GLOSSARY_SYSTEM.format(level=level),
            f"PAPER: {paper.title}\nABSTRACT: {paper.abstract}\n\nTERMS TO DEFINE:\n"
            + "\n".join(f"- {t}" for t in new_terms),
        )
        glossary = {str(k): str(v) for k, v in g.get("terms", {}).items()}

    return Digest(
        paper=paper,
        tldr=str(outline["tldr"]),
        why_it_matters=str(outline["why_it_matters"]),
        concepts=concepts,
        jargon=jargon,
        glossary=glossary,
        self_test=[str(q) for q in outline["self_test"]],
        model=backend.model,
        level=level,
        figures=figures,
    )
