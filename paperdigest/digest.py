from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable

from . import mermaid
from .extract import Paper
from .llm import Backend, LLMError, complete_with_retry, repair_json, strip_fences


@dataclass
class ConceptNote:
    title: str
    body_md: str
    section: str


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


def _find_section_text(paper: Paper, section_title: str, fallback: str) -> str:
    want = section_title.strip().lower()
    for s in paper.sections:
        have = s.title.strip().lower()
        if want and (want in have or have in want):
            return s.text
    return fallback


def build_digest(
    paper: Paper,
    backend: Backend,
    level: str,
    existing_terms: set[str],
    max_chars: int,
    progress: Callable[[str], None] = _default_progress,
    diagram: str = "mermaid",
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

    concepts: list[ConceptNote] = []
    total = len(outline["concepts"])
    for i, c in enumerate(outline["concepts"], 1):
        if "title" not in c:
            raise LLMError(f"outline concept entry is missing 'title': {c!r}")
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
        concepts.append(ConceptNote(title=title, body_md=md, section=section))

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
    )
