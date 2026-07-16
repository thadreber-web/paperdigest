import json
from pathlib import Path

import pytest
from conftest import FakeBackend, KeyedFakeBackend  # tests/ is on sys.path under pytest

from paperdigest.digest import Digest, FigureNote, _call_json, _find_section_text, build_digest
from paperdigest.extract import Figure, Paper, Section
from paperdigest.llm import LLMError, VisionUnsupportedError

FIXTURES = Path(__file__).parent / "fixtures"
FIGURE_BYTES = (FIXTURES / "figure1.png").read_bytes()


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


def test_build_digest_figures_get_image_caption_tldr_and_concept_match(paper, tmp_path):
    fig_path = tmp_path / "fig1.png"
    fig_path.write_bytes(FIGURE_BYTES)
    fig = Figure(caption="Fig 1: the architecture.", src="fig1.png", section="1 Introduction")
    paper.figures = [fig]
    backend = FakeBackend([OUTLINE, "Body about attention.", "Body about stacking.", "Figure explanation.", GLOSSARY])
    d = build_digest(
        paper, backend, "intermediate", set(), 400_000, progress=lambda m: None,
        figure_paths={fig.src: fig_path},
    )
    assert len(d.figures) == 1
    fnote = d.figures[0]
    assert isinstance(fnote, FigureNote)
    assert fnote.body_md == "Figure explanation."
    assert fnote.image_path == fig_path
    assert fnote.caption == fig.caption
    assert fnote.concept_title == "Self-Attention"  # section "1 Introduction" matches that concept

    figure_call_index = 3  # outline, concept1, concept2, figure
    assert backend.images_calls[figure_call_index] == [FIGURE_BYTES]
    assert backend.images_calls[0] is None  # outline: no images
    assert backend.images_calls[1] is None  # concept calls: no images
    figure_user = backend.calls[figure_call_index][1]
    assert "Tiny Transformers Explained" in figure_user  # paper title
    assert "Tiny transformers work." in figure_user  # tldr
    assert "Fig 1: the architecture." in figure_user  # caption verbatim


def test_build_digest_figure_unmatched_section_goes_to_overview(paper, tmp_path):
    fig_path = tmp_path / "fig1.png"
    fig_path.write_bytes(FIGURE_BYTES)
    fig = Figure(caption="Fig 1", src="fig1.png", section="9 Nonexistent Section")
    paper.figures = [fig]
    backend = FakeBackend([OUTLINE, "c1", "c2", "Figure explanation.", GLOSSARY])
    d = build_digest(
        paper, backend, "intermediate", set(), 400_000, progress=lambda m: None,
        figure_paths={fig.src: fig_path},
    )
    assert d.figures[0].concept_title is None


def test_build_digest_no_figure_paths_skips_figure_step(paper, tmp_path):
    fig = Figure(caption="Fig 1", src="fig1.png", section="1 Introduction")
    paper.figures = [fig]
    backend = FakeBackend([OUTLINE, "c1", "c2", GLOSSARY])
    d = build_digest(paper, backend, "intermediate", set(), 400_000, progress=lambda m: None)
    assert d.figures == []
    assert len(backend.calls) == 4  # no figure call made


class _VisionRejectingBackend:
    model = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.images_calls = []

    def complete(self, system, user, json_mode=False, images=None):
        self.calls.append((system, user))
        self.images_calls.append(images)
        if images:
            raise VisionUnsupportedError("backend rejected image content")
        if not self.responses:
            raise AssertionError("VisionRejectingBackend ran out of responses")
        return self.responses.pop(0)


def test_build_digest_vision_unsupported_skips_all_figures_with_one_warning(paper, tmp_path):
    fig_path = tmp_path / "fig1.png"
    fig_path.write_bytes(FIGURE_BYTES)
    fig1 = Figure(caption="Fig 1", src="fig1.png", section="1 Introduction")
    fig2 = Figure(caption="Fig 2", src="fig2.png", section="2 Method")
    paper.figures = [fig1, fig2]
    backend = _VisionRejectingBackend([OUTLINE, "c1", "c2", GLOSSARY])
    warnings = []
    d = build_digest(
        paper, backend, "intermediate", set(), 400_000, progress=warnings.append,
        figure_paths={fig1.src: fig_path, fig2.src: fig_path},
    )
    assert d.figures == []
    vision_warnings = [w for w in warnings if "vision support" in w.lower()]
    assert len(vision_warnings) == 1
    assert "skipping 2 figures" in vision_warnings[0]
    # digest still completes normally
    assert d.tldr == "Tiny transformers work."


class _ImgQueueBackend:
    """Fake backend that answers image-bearing calls from a separate ordered queue,
    where an entry may be an Exception instance to raise instead of a response string."""

    model = "fake-model"

    def __init__(self, non_image_responses, image_responses):
        self.non_image_responses = list(non_image_responses)
        self.image_responses = list(image_responses)
        self.calls = []
        self.images_calls = []

    def complete(self, system, user, json_mode=False, images=None):
        self.calls.append((system, user))
        self.images_calls.append(images)
        if images:
            item = self.image_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if not self.non_image_responses:
            raise AssertionError("ImgQueueBackend ran out of non-image responses")
        return self.non_image_responses.pop(0)


def test_build_digest_single_figure_llmerror_skips_just_that_figure(paper, tmp_path):
    fig_path = tmp_path / "fig1.png"
    fig_path.write_bytes(FIGURE_BYTES)
    fig1 = Figure(caption="Fig 1", src="fig1.png", section=None)
    fig2 = Figure(caption="Fig 2", src="fig2.png", section=None)
    fig3 = Figure(caption="Fig 3", src="fig3.png", section=None)
    paper.figures = [fig1, fig2, fig3]
    backend = _ImgQueueBackend(
        [OUTLINE, "c1", "c2", GLOSSARY],
        ["Note 1", LLMError("boom"), "Note 3"],
    )
    warnings = []
    d = build_digest(
        paper, backend, "intermediate", set(), 400_000, progress=warnings.append, workers=1,
        figure_paths={fig1.src: fig_path, fig2.src: fig_path, fig3.src: fig_path},
    )
    assert [f.caption for f in d.figures] == ["Fig 1", "Fig 3"]
    assert any("figure 2" in w.lower() and "skip" in w.lower() for w in warnings)


def test_find_section_text_no_match_returns_fallback():
    p = Paper(
        arxiv_id="x",
        title="t",
        abstract="a",
        sections=[Section("Introduction", "intro text")],
        url="https://example.com",
    )
    assert _find_section_text(p, "Nonexistent Section", fallback="FALLBACK") == "FALLBACK"
