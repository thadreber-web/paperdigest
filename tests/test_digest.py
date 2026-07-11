import json

import pytest

from paperdigest.digest import Digest, build_digest, _call_json
from paperdigest.extract import Paper, Section
from paperdigest.llm import LLMError
from conftest import FakeBackend  # tests/ is on sys.path under pytest


@pytest.fixture
def paper():
    return Paper(
        arxiv_id="1706.03762",
        title="Tiny Transformers Explained",
        abstract="We study tiny transformers.",
        sections=[
            Section("1 Introduction", "Transformers are neural networks."),
            Section("2 Method", "Our method stacks two attention layers."),
        ],
        url="https://arxiv.org/abs/1706.03762",
    )


OUTLINE = json.dumps(
    {
        "tldr": "Tiny transformers work.",
        "why_it_matters": "Cheap models are useful.",
        "concepts": [
            {"title": "Self-Attention", "section": "1 Introduction"},
            {"title": "Layer Stacking", "section": "2 Method"},
        ],
        "jargon": ["attention", "softmax"],
        "self_test": ["What is attention?", "Why stack layers?", "What is softmax?"],
    }
)

GLOSSARY = json.dumps({"terms": {"attention": "A weighting scheme.", "softmax": "Turns scores into probabilities."}})


def test_build_digest_happy_path(paper):
    backend = FakeBackend([OUTLINE, "Body about attention.", "Body about stacking.", GLOSSARY])
    d = build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None)
    assert isinstance(d, Digest)
    assert d.tldr == "Tiny transformers work."
    assert [c.title for c in d.concepts] == ["Self-Attention", "Layer Stacking"]
    assert d.concepts[0].body_md == "Body about attention."
    assert d.jargon == ["attention", "softmax"]
    assert d.glossary["softmax"] == "Turns scores into probabilities."
    assert len(d.self_test) == 3
    assert d.model == "fake-model"
    assert d.level == "intermediate"
    assert len(backend.calls) == 4  # outline + 2 concepts + glossary


def test_concept_call_gets_matching_section_text(paper):
    backend = FakeBackend([OUTLINE, "c1", "c2", GLOSSARY])
    build_digest(paper, backend, "beginner", set(), 400_000, progress=lambda m: None)
    concept1_user = backend.calls[1][1]
    assert "Transformers are neural networks." in concept1_user
    concept2_user = backend.calls[2][1]
    assert "stacks two attention layers" in concept2_user


def test_existing_terms_skip_glossary_call(paper):
    backend = FakeBackend([OUTLINE, "c1", "c2"])
    d = build_digest(paper, backend, "intermediate", {"attention", "softmax"}, 400_000, progress=lambda m: None)
    assert d.glossary == {}
    assert d.jargon == ["attention", "softmax"]  # still linkable
    assert len(backend.calls) == 3  # no glossary call


def test_long_paper_is_trimmed(paper):
    paper.sections[0].text = "x" * 100_000
    paper.sections[1].text = "y" * 100_000
    backend = FakeBackend([OUTLINE, "c1", "c2", GLOSSARY])
    warnings = []
    build_digest(paper, backend, "intermediate", set(), 10_000, progress=warnings.append)
    outline_user = backend.calls[0][1]
    assert len(outline_user) < 20_000
    assert any("trimm" in w.lower() for w in warnings)


def test_call_json_repairs_broken_json():
    backend = FakeBackend(['{"a": 1,,,}', '{"a": 1}'])
    assert _call_json(backend, "sys", "user") == {"a": 1}
    assert len(backend.calls) == 2


def test_call_json_strips_markdown_fences():
    backend = FakeBackend(['```json\n{"a": 1}\n```'])
    assert _call_json(backend, "sys", "user") == {"a": 1}


def test_call_json_raises_after_failed_repair():
    backend = FakeBackend(["nope", "still nope"])
    with pytest.raises(LLMError, match="unparseable JSON"):
        _call_json(backend, "sys", "user")


def test_invalid_level_raises_value_error(paper):
    backend = FakeBackend([])
    with pytest.raises(ValueError, match="level must be one of"):
        build_digest(paper, backend, "expert", set(), 400_000, progress=lambda m: None)


def test_concept_missing_title_raises_llmerror(paper):
    bad_outline = json.dumps(
        {
            "tldr": "t",
            "why_it_matters": "w",
            "concepts": [{"section": "1 Introduction"}],
            "jargon": [],
            "self_test": [],
        }
    )
    backend = FakeBackend([bad_outline])
    with pytest.raises(LLMError, match="missing 'title'"):
        build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None)
