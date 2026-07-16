import pytest

from paperdigest.extract import ExtractError, Figure, extract_paper


def test_extracts_title_abstract_sections(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert paper.title == "Tiny Transformers Explained"
    assert paper.abstract.startswith("We study tiny transformers")
    assert paper.url == "https://arxiv.org/abs/1706.03762"
    assert [s.title for s in paper.sections] == ["1 Introduction", "2 Method"]


def test_figure_parsed_with_caption_src_section(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert paper.figures == [
        Figure(caption="Figure 1: The tiny attention block.", src="figure1.png", section="1 Introduction")
    ]


def test_captionless_and_svg_figures_skipped(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    srcs = [f.src for f in paper.figures]
    assert "figure2.svg" not in srcs
    assert "figure3.png" not in srcs
    assert len(paper.figures) == 1


def test_math_becomes_latex(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert "$A=QK^{T}$" in paper.abstract
    assert r"$\mathrm{softmax}(QK^{T})V$" in paper.sections[0].text


def test_references_section_dropped(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert not any("References" in s.title for s in paper.sections)


def test_non_paper_html_raises():
    with pytest.raises(ExtractError):
        extract_paper("<html><body><p>hello</p></body></html>", "1234.56789")
