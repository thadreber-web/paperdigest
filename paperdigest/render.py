from __future__ import annotations

import datetime
import re
import shutil
from pathlib import Path

from .digest import Digest


class OutputExistsError(Exception):
    pass


def slugify(text: str, max_words: int = 6) -> str:
    words = re.sub(r"[^a-z0-9\s-]", "", text.lower()).split()
    return "-".join(words[:max_words]) or "untitled"


def note_name(text: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\[\]#^]', "", text)
    return re.sub(r"\s+", " ", cleaned).strip() or "Untitled"


def _frontmatter(digest: Digest, extra_tags: str = "") -> str:
    today = datetime.date.today().isoformat()
    tags = f"[paper-digest{extra_tags}]"
    return (
        "---\n"
        f"source: {digest.paper.url}\n"
        f"date: {today}\n"
        f"model: {digest.model}\n"
        f"level: {digest.level}\n"
        f"tags: {tags}\n"
        "---\n\n"
    )


def paper_folder(arxiv_id: str, title: str, vault: Path) -> Path:
    year = "20" + arxiv_id[:2]
    return vault / "Papers" / f"{year}-{slugify(title)}"


def check_output_free(folder: Path, force: bool) -> None:
    if folder.exists() and not force:
        raise OutputExistsError(
            f"{folder} already exists — re-run with --force to overwrite (your edits there will be lost)"
        )


def render_digest(digest: Digest, vault: Path, force: bool = False) -> Path:
    folder = paper_folder(digest.paper.arxiv_id, digest.paper.title, vault)
    check_output_free(folder, force)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    concept_names = [f"{i:02d} {note_name(c.title)}" for i, c in enumerate(digest.concepts, 1)]

    fm = _frontmatter(digest)
    overview = fm + f"# {digest.paper.title}\n\n"
    overview += f"**TLDR:** {digest.tldr}\n\n"
    overview += f"## Why it matters\n\n{digest.why_it_matters}\n\n"
    overview += "## Concepts\n\n" + "\n".join(f"- [[{n}]]" for n in concept_names) + "\n\n"
    if digest.jargon:
        links = " · ".join(f"[[{note_name(t.lower())}]]" for t in digest.jargon)
        overview += f"## Glossary terms\n\n{links}\n\n"
    if digest.self_test:
        overview += "## Check your understanding\n\n"
        overview += "\n".join(f"{i}. {q}" for i, q in enumerate(digest.self_test, 1)) + "\n"
    (folder / "00 Overview.md").write_text(overview)

    for name, concept in zip(concept_names, digest.concepts):
        note = fm + f"# {note_name(concept.title)}\n\n"
        if concept.section:
            note += f"*Source: {concept.section} of [{digest.paper.title}]({digest.paper.url})*\n\n"
        note += concept.body_md + "\n"
        (folder / f"{name}.md").write_text(note)

    if digest.glossary:
        gdir = vault / "Glossary"
        gdir.mkdir(parents=True, exist_ok=True)
        gfm = _frontmatter(digest, extra_tags=", glossary")
        for term, definition in digest.glossary.items():
            gfile = gdir / f"{note_name(term.lower())}.md"
            if gfile.exists():
                continue  # never clobber accumulated/edited term notes
            gfile.write_text(gfm + f"# {note_name(term)}\n\n{definition}\n")

    return folder
