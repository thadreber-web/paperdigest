import json

import pytest
from conftest import FakeBackend, KeyedFakeBackend  # tests/ is on sys.path under pytest

from paperdigest.digest import Digest, _call_json, _find_section_text, build_digest
from paperdigest.extract import Paper, Section
from paperdigest.llm import LLMError


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


def test_diagram_mermaid_is_default_prompt_and_sanitizes(paper):
    body = "Intro.\n```mermaid\nflowchart LR\n    A[Foo (bar)] --> B[Ok]\n```\n"
    backend = FakeBackend([OUTLINE, body, "c2", GLOSSARY])
    d = build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None)
    concept_system = backend.calls[1][0]
    assert "Mermaid" in concept_system and "ASCII" not in concept_system
    assert 'A["Foo (bar)"]' in d.concepts[0].body_md


def test_diagram_ascii_option_keeps_old_prompt(paper):
    backend = FakeBackend([OUTLINE, "c1", "c2", GLOSSARY])
    build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None, diagram="ascii")
    concept_system = backend.calls[1][0]
    assert "ASCII" in concept_system and "Mermaid" not in concept_system


def test_invalid_diagram_raises(paper):
    backend = FakeBackend([OUTLINE])
    with pytest.raises(ValueError, match="diagram"):
        build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None, diagram="svg")


def test_find_section_text_exact_title_wins_over_substring_collision():
    p = Paper(
        arxiv_id="x",
        title="t",
        abstract="a",
        sections=[
            Section("Method", "exact method text"),
            Section("Methodology", "methodology text"),
        ],
        url="https://example.com",
    )
    assert _find_section_text(p, "method", fallback="FALLBACK") == "exact method text"


def test_find_section_text_matches_numeric_prefixed_title():
    p = Paper(
        arxiv_id="x",
        title="t",
        abstract="a",
        sections=[
            Section("3 Method", "method text"),
            Section("3.1 Ablations", "ablations text"),
        ],
        url="https://example.com",
    )
    assert _find_section_text(p, "Method", fallback="FALLBACK") == "method text"
    assert _find_section_text(p, "Ablations", fallback="FALLBACK") == "ablations text"


def test_find_section_text_numeric_only_title_matches_correct_section():
    p = Paper(
        arxiv_id="x",
        title="t",
        abstract="a",
        sections=[
            Section("4", "four text"),
            Section("3", "three text"),
        ],
        url="https://example.com",
    )
    assert _find_section_text(p, "3", fallback="FALLBACK") == "three text"


def test_build_digest_parallel_returns_concepts_in_order(paper):
    paper.sections.append(Section("3 Results", "We report numbers."))
    paper.sections.append(Section("4 Discussion", "We discuss limitations."))
    titles = ["Self-Attention", "Layer Stacking", "Evaluation", "Limitations"]
    sections = ["1 Introduction", "2 Method", "3 Results", "4 Discussion"]
    outline = json.dumps(
        {
            "tldr": "t",
            "why_it_matters": "w",
            "concepts": [{"title": t, "section": s} for t, s in zip(titles, sections)],
            "jargon": [],
            "self_test": ["q1"],
        }
    )
    backend = KeyedFakeBackend(
        {
            "READER LEVEL:": outline,
            **{f"CONCEPT TO EXPLAIN: {t}": f"body for {t}" for t in titles},
        }
    )
    d = build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None, workers=4)
    assert [c.title for c in d.concepts] == titles
    assert [c.body_md for c in d.concepts] == [f"body for {t}" for t in titles]


def test_build_digest_parallel_propagates_error_from_one_worker(paper):
    outline = json.dumps(
        {
            "tldr": "t",
            "why_it_matters": "w",
            "concepts": [{"title": "Good", "section": "1 Introduction"}, {"title": "Bad", "section": "2 Method"}],
            "jargon": [],
            "self_test": ["q1"],
        }
    )

    class _BoomBackend(KeyedFakeBackend):
        def complete(self, system, user, json_mode=False):
            if "CONCEPT TO EXPLAIN: Bad" in user:
                raise RuntimeError("worker exploded")
            return super().complete(system, user, json_mode=json_mode)

    backend = _BoomBackend({"READER LEVEL:": outline, "CONCEPT TO EXPLAIN: Good": "fine"})
    with pytest.raises(LLMError, match="worker exploded"):
        build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None, workers=4)


def test_find_section_text_no_match_returns_fallback():
    p = Paper(
        arxiv_id="x",
        title="t",
        abstract="a",
        sections=[Section("Introduction", "intro text")],
        url="https://example.com",
    )
    assert _find_section_text(p, "Nonexistent Section", fallback="FALLBACK") == "FALLBACK"
