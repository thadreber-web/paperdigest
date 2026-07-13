import pytest

from paperdigest.digest import ConceptNote, Digest
from paperdigest.extract import Paper, Section
from paperdigest.render import OutputExistsError, note_name, render_digest, slugify


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
