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


def folder_name(arxiv_id: str, title: str) -> str:
    year = "20" + arxiv_id[:2]
    return f"{year}-{slugify(title)}"


def paper_folder(arxiv_id: str, title: str, vault: Path) -> Path:
    return vault / "Papers" / folder_name(arxiv_id, title)


def check_output_free(folder: Path, force: bool) -> None:
    if folder.exists() and not force:
        raise OutputExistsError(
            f"{folder} already exists — re-run with --force to overwrite (your edits there will be lost)"
        )


_SECTION_REF_RE = re.compile(r"§(\d+(?:\.\d+)*)")


def _section_num(section: str) -> str:
    """Extract section number from a heading.

    Finds prefixed forms ('Section 3.2', '§3.2') anywhere, or leading bare
    numbers. Examples: '3.2 Attention' -> '3.2', 'Section 3.2' -> '3.2',
    'Intro, Section 4.1 (why)' -> '4.1'.
    """
    m = re.search(r"(?:section|sec\.?|§)\s*(\d+(?:\.\d+)*)", section, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"(\d+(?:\.\d+)*)", section.strip())
    return m.group(1) if m else ""


def _project_section_index(project: Path) -> dict[str, list[str]]:
    """Relative module path -> ordered unique paper-section numbers cited in the file."""
    index: dict[str, list[str]] = {}
    src = project / "src"
    if not src.is_dir():
        return index
    for py in sorted(src.rglob("*.py")):
        if py.name in ("__init__.py", "tracking.py"):
            continue
        try:
            refs = list(dict.fromkeys(_SECTION_REF_RE.findall(py.read_text())))
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read (broken symlinks, permission denied, invalid encoding)
            continue
        if refs:
            index[str(py.relative_to(project))] = refs
    return index


def _sections_match(concept_num: str, cited: str) -> bool:
    if not concept_num or not cited:
        return False
    return concept_num == cited or cited.startswith(concept_num + ".") or concept_num.startswith(cited + ".")


def render_digest(digest: Digest, vault: Path, force: bool = False, project_dir: Path | None = None) -> Path:
    folder = paper_folder(digest.paper.arxiv_id, digest.paper.title, vault)
    check_output_free(folder, force)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    try:
        concept_names = [f"{i:02d} {note_name(c.title)}" for i, c in enumerate(digest.concepts, 1)]

        fm = _frontmatter(digest)
        overview = fm + f"# {digest.paper.title}\n\n"
        overview += f"**TLDR:** {digest.tldr}\n\n"
        overview += f"## Why it matters\n\n{digest.why_it_matters}\n\n"
        project = None
        if project_dir is not None:
            project = project_dir / folder_name(digest.paper.arxiv_id, digest.paper.title)
            overview += (
                "## Build it\n\n"
                "A scaffolded research project for this paper lives at\n"
                f"[`{project}`]({project.as_uri()})\n"
                "— open `AGENTS.md` there and hand it to a coding agent.\n"
                "If the project doesn't exist yet, generate it:\n"
                f"`paperdigest scaffold {digest.paper.arxiv_id} --dest {project_dir}`\n\n"
            )
        overview += "## Concepts\n\n" + "\n".join(f"- [[{n}]]" for n in concept_names) + "\n\n"
        if digest.jargon:
            links = " · ".join(f"[[{note_name(t.lower())}]]" for t in digest.jargon)
            overview += f"## Glossary terms\n\n{links}\n\n"
        if digest.self_test:
            overview += "## Check your understanding\n\n"
            overview += "\n".join(f"{i}. {q}" for i, q in enumerate(digest.self_test, 1)) + "\n"
        (folder / "00 Overview.md").write_text(overview)

        concept_bodies: dict[str, str] = {}
        overview_extra = ""
        for i, fnote in enumerate(digest.figures, 1):
            dest_name = f"fig{i}{fnote.image_path.suffix.lower()}"
            shutil.copyfile(fnote.image_path, folder / dest_name)
            block = f"\n\n## Figure: {fnote.caption}\n\n![[{dest_name}]]\n\n{fnote.body_md}\n"
            if fnote.concept_title is not None:
                concept_bodies[fnote.concept_title] = concept_bodies.get(fnote.concept_title, "") + block
            else:
                overview_extra += block

        section_index = _project_section_index(project) if project is not None and project.is_dir() else {}
        for name, concept in zip(concept_names, digest.concepts):
            note = fm + f"# {note_name(concept.title)}\n\n"
            if concept.section:
                note += f"*Source: {concept.section} of [{digest.paper.title}]({digest.paper.url})*\n\n"
            if section_index:
                num = _section_num(concept.section)
                matched = [rel for rel, refs in section_index.items() if any(_sections_match(num, r) for r in refs)]
                if matched:
                    links = ", ".join(f"[`{rel}`]({(project / rel).as_uri()})" for rel in matched)
                    note += f"*Build it: {links}*\n\n"
            note += concept.body_md + "\n"
            note += concept_bodies.get(concept.title, "")
            (folder / f"{name}.md").write_text(note)

        if overview_extra:
            (folder / "00 Overview.md").write_text((folder / "00 Overview.md").read_text() + overview_extra)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)  # never leave a half-written paper folder
        raise

    # Glossary notes live outside the paper folder and accumulate across runs, so a
    # failure here must not trigger the rollback above (folder is already complete)
    # nor clobber pre-existing/edited term notes.
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
