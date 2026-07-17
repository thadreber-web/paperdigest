from pathlib import Path

import pytest

from paperdigest.digest import ConceptNote, Digest, FigureNote
from paperdigest.extract import Paper, Section
from paperdigest.render import (
    OutputExistsError,
    _project_section_index,
    _section_num,
    _sections_match,
    folder_name,
    note_name,
    render_digest,
    slugify,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIGURE_PATH = FIXTURES / "figure1.png"


def _digest():
    paper = Paper(
        arxiv_id="1706.03762",
        title="Tiny Transformers Explained",
        abstract="We study tiny transformers.",
        sections=[Section("1 Introduction", "text")],
        url="https://arxiv.org/abs/1706.03762",
    )
    return Digest(
        paper=paper,
        tldr="Tiny transformers work.",
        why_it_matters="Cheap models are useful.",
        concepts=[
            ConceptNote("Self-Attention", "Attention body.", "1 Introduction"),
            ConceptNote("Layer Stacking", "Stacking body.", "2 Method"),
        ],
        jargon=["attention", "softmax"],
        glossary={"attention": "A weighting scheme."},  # softmax pre-exists in vault
        self_test=["What is attention?"],
        model="fake-model",
        level="intermediate",
    )


def test_slugify_and_note_name():
    assert slugify("Attention Is All You Need, Really!") == "attention-is-all-you-need-really"
    assert note_name('Bad: "name" [x] #tag?') == "Bad name x tag"


def test_folder_name_composes_year_and_slug():
    assert folder_name("1706.03762", "Attention Is All You Need") == "2017-attention-is-all-you-need"


def test_render_creates_folder_and_notes(tmp_path):
    folder = render_digest(_digest(), tmp_path)
    assert folder == tmp_path / "Papers" / "2017-tiny-transformers-explained"
    overview = (folder / "00 Overview.md").read_text()
    assert "Tiny transformers work." in overview
    assert "[[01 Self-Attention]]" in overview
    assert "[[02 Layer Stacking]]" in overview
    assert "[[attention]]" in overview and "[[softmax]]" in overview
    assert "What is attention?" in overview
    assert "source: https://arxiv.org/abs/1706.03762" in overview
    assert "level: intermediate" in overview
    concept = (folder / "01 Self-Attention.md").read_text()
    assert "Attention body." in concept
    assert "1 Introduction" in concept


def test_render_writes_new_glossary_terms_only(tmp_path):
    gdir = tmp_path / "Glossary"
    gdir.mkdir(parents=True)
    (gdir / "softmax.md").write_text("MY HAND-WRITTEN NOTE")
    render_digest(_digest(), tmp_path)
    assert "A weighting scheme." in (gdir / "attention.md").read_text()
    assert (gdir / "softmax.md").read_text() == "MY HAND-WRITTEN NOTE"


def test_render_refuses_existing_folder(tmp_path):
    render_digest(_digest(), tmp_path)
    with pytest.raises(OutputExistsError, match="--force"):
        render_digest(_digest(), tmp_path)


def test_render_force_overwrites(tmp_path):
    folder = render_digest(_digest(), tmp_path)
    (folder / "00 Overview.md").write_text("my edits")
    folder2 = render_digest(_digest(), tmp_path, force=True)
    assert folder2 == folder
    assert "Tiny transformers work." in (folder / "00 Overview.md").read_text()


def test_force_still_never_touches_glossary(tmp_path):
    render_digest(_digest(), tmp_path)
    gnote = tmp_path / "Glossary" / "attention.md"
    gnote.write_text("MY EDITS")
    render_digest(_digest(), tmp_path, force=True)
    assert gnote.read_text() == "MY EDITS"


def test_render_rolls_back_paper_folder_on_partial_write_failure(tmp_path, monkeypatch):
    # A pre-existing glossary note must survive an aborted run untouched.
    gdir = tmp_path / "Glossary"
    gdir.mkdir(parents=True)
    (gdir / "softmax.md").write_text("MY HAND-WRITTEN NOTE")

    from pathlib import Path

    real_write_text = Path.write_text
    calls = []

    def flaky_write_text(self, *args, **kwargs):
        calls.append(self)
        if self.name == "01 Self-Attention.md":
            raise OSError("disk full (simulated)")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)

    folder = tmp_path / "Papers" / "2017-tiny-transformers-explained"
    with pytest.raises(OSError, match="disk full"):
        render_digest(_digest(), tmp_path)

    assert not folder.exists()  # rolled back: no half-written paper folder
    assert (gdir / "softmax.md").read_text() == "MY HAND-WRITTEN NOTE"  # untouched


def test_render_embeds_figure_in_matched_concept_note(tmp_path):
    d = _digest()
    d.figures = [
        FigureNote(
            caption="Fig 1: the architecture.",
            body_md="This shows the architecture.",
            image_path=FIGURE_PATH,
            concept_title="Self-Attention",
        )
    ]
    folder = render_digest(d, tmp_path)
    assert (folder / "fig1.png").read_bytes() == FIGURE_PATH.read_bytes()
    concept = (folder / "01 Self-Attention.md").read_text()
    assert "## Figure: Fig 1: the architecture." in concept
    assert "![[fig1.png]]" in concept
    assert "This shows the architecture." in concept
    other_concept = (folder / "02 Layer Stacking.md").read_text()
    assert "Figure:" not in other_concept
    overview = (folder / "00 Overview.md").read_text()
    assert "Figure:" not in overview


def test_render_embeds_unmatched_figure_in_overview(tmp_path):
    d = _digest()
    d.figures = [
        FigureNote(
            caption="Fig 2: results.",
            body_md="This shows results.",
            image_path=FIGURE_PATH,
            concept_title=None,
        )
    ]
    folder = render_digest(d, tmp_path)
    assert (folder / "fig1.png").exists()
    overview = (folder / "00 Overview.md").read_text()
    assert "## Figure: Fig 2: results." in overview
    assert "![[fig1.png]]" in overview
    assert "This shows results." in overview


def _fake_project(tmp_path):
    proj = tmp_path / "projects" / "2017-tiny-transformers-explained"
    (proj / "src" / "pkg").mkdir(parents=True)
    (proj / "src" / "pkg" / "layers.py").write_text("# TODO(paper §3.2): attention\n")
    (proj / "src" / "pkg" / "utils.py").write_text("# no section refs\n")
    return proj


def test_overview_build_it_section_when_project_dir_set(tmp_path):
    folder = render_digest(_digest(), tmp_path, project_dir=tmp_path / "projects")
    overview = (folder / "00 Overview.md").read_text()
    assert "## Build it" in overview
    assert "2017-tiny-transformers-explained" in overview
    assert "paperdigest scaffold 1706.03762 --dest" in overview
    assert overview.index("## Why it matters") < overview.index("## Build it") < overview.index("## Concepts")


def test_no_project_dir_output_unchanged(tmp_path):
    folder = render_digest(_digest(), tmp_path)
    overview = (folder / "00 Overview.md").read_text()
    assert "Build it" not in overview
    concept = (folder / "01 Self-Attention.md").read_text()
    assert "Build it" not in concept


def test_concept_note_links_matching_module(tmp_path):
    _fake_project(tmp_path)
    d = _digest()
    d.concepts[0] = ConceptNote("Self-Attention", "Attention body.", "3.2 Attention")
    folder = render_digest(d, tmp_path, project_dir=tmp_path / "projects")
    concept = (folder / "01 Self-Attention.md").read_text()
    assert "*Build it: " in concept
    assert "layers.py" in concept
    assert "file://" in concept
    assert "utils.py" not in concept
    other_concept = (folder / "02 Layer Stacking.md").read_text()
    assert "Build it" not in other_concept  # "2 Method" doesn't cite §3.2


def test_concept_note_links_with_section_prefix(tmp_path):
    # Regression test: LLMs emit "Section 3.2.1" format which must link to modules citing §3.2.1
    _fake_project(tmp_path)
    d = _digest()
    d.concepts[0] = ConceptNote("Self-Attention", "Attention body.", "Section 3.2 Attention")
    folder = render_digest(d, tmp_path, project_dir=tmp_path / "projects")
    concept = (folder / "01 Self-Attention.md").read_text()
    # Should produce link even though section format is "Section 3.2"
    assert "*Build it: " in concept
    assert "layers.py" in concept
    assert "file://" in concept


def test_concept_links_skipped_when_project_missing(tmp_path):
    folder = render_digest(_digest(), tmp_path, project_dir=tmp_path / "projects")  # nothing on disk
    overview = (folder / "00 Overview.md").read_text()
    assert "## Build it" in overview
    concept = (folder / "01 Self-Attention.md").read_text()
    assert "Build it" not in concept


def test_project_section_index_skips_init_and_tracking(tmp_path):
    proj = tmp_path / "proj"
    (proj / "src" / "pkg").mkdir(parents=True)
    (proj / "src" / "pkg" / "__init__.py").write_text("# §1\n")
    (proj / "src" / "pkg" / "tracking.py").write_text("# §1\n")
    (proj / "src" / "pkg" / "model.py").write_text("# §1.1\n")
    assert set(_project_section_index(proj)) == {"src/pkg/model.py"}


def test_project_section_index_no_src_dir(tmp_path):
    assert _project_section_index(tmp_path / "proj") == {}


def test_sections_match_empty_inputs():
    assert _sections_match("", "3.2") is False
    assert _sections_match("3.2", "") is False


def test_render_multiple_figures_numbered_and_extensions_preserved(tmp_path):
    d = _digest()
    d.figures = [
        FigureNote("Fig 1", "Body 1", FIGURE_PATH, "Self-Attention"),
        FigureNote("Fig 2", "Body 2", FIGURE_PATH, None),
    ]
    folder = render_digest(d, tmp_path)
    assert (folder / "fig1.png").exists()
    assert (folder / "fig2.png").exists()
    assert "![[fig1.png]]" in (folder / "01 Self-Attention.md").read_text()
    assert "![[fig2.png]]" in (folder / "00 Overview.md").read_text()


def test_sections_match_dotted_prefix_logic():
    # Basic match
    assert _sections_match("3", "3") is True
    # Concept subsection matches cited
    assert _sections_match("3", "3.2") is True
    # Cited subsection matches concept
    assert _sections_match("3.2.1", "3.2") is True
    # False positive guard: "3" should not match "31"
    assert _sections_match("3", "31") is False


def test_section_num_non_numeric_sections():
    # Non-numeric section should return ""
    assert _section_num("Appendix A") == ""
    assert _section_num("Introduction") == ""
    assert _section_num("") == ""
    # Numeric sections should work
    assert _section_num("3.2 Attention") == "3.2"
    assert _section_num("1 Introduction") == "1"


def test_section_num_with_optional_prefix():
    # Test "Section" prefix (case-insensitive)
    assert _section_num("Section 3.2.1") == "3.2.1"
    assert _section_num("section 3 Method") == "3"
    assert _section_num("SECTION 2.1") == "2.1"
    # Test "Sec." or "Sec" prefix (case-insensitive)
    assert _section_num("Sec. 4.1") == "4.1"
    assert _section_num("sec. 1.2.3") == "1.2.3"
    assert _section_num("SEC 2") == "2"
    # Test "§" prefix
    assert _section_num("§3.2") == "3.2"
    assert _section_num("§1") == "1"
    # Non-numeric still returns ""
    assert _section_num("Introduction") == ""
    # Preserve existing behavior: no prefix
    assert _section_num("3.2 Attention") == "3.2"


def test_section_num_regression_section_anywhere_in_heading():
    # Regression test: live-observed "Attention Is All You Need, Section 4 (Why Self-Attention)" -> "4"
    assert _section_num("Attention Is All You Need, Section 4 (Why Self-Attention)") == "4"


def test_section_num_symbol_in_midstring():
    # Regression test: "see § 3.2 for details" -> "3.2"
    assert _section_num("see § 3.2 for details") == "3.2"


def test_section_num_bare_midstring_number_no_match():
    # Bare mid-string numbers must NOT match
    assert _section_num("Top 10 results") == ""


def test_section_num_existing_anchored_behaviors_preserved():
    # Existing behaviors must still work
    assert _section_num("3.2 Attention") == "3.2"
    assert _section_num("Section 3.2.1") == "3.2.1"


def test_project_section_index_skips_unreadable_files(tmp_path):
    # Create a project with one good module, one broken symlink, and one undecodable file
    proj = tmp_path / "proj"
    src = proj / "src" / "pkg"
    src.mkdir(parents=True)

    # Good module
    (src / "good.py").write_text("# TODO(paper §3.2): attention\n")

    # Broken symlink
    (src / "broken.py").symlink_to(src / "does-not-exist.py")

    # Undecodable file
    (src / "bad.py").write_bytes(b"\xff\xfe\x00garbage")

    # Should not raise and should contain only the good file
    index = _project_section_index(proj)
    assert set(index.keys()) == {"src/pkg/good.py"}
    assert index["src/pkg/good.py"] == ["3.2"]
