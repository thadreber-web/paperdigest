import pytest

from paperdigest.extract import ExtractError, extract_paper


def test_extracts_title_abstract_sections(fixture_html):
    paper = extract_paper(fixture_html, "1706.03762")
    assert paper.title == "Tiny Transformers Explained"
    assert paper.abstract.startswith("We study tiny transformers")
    assert paper.url == "https://arxiv.org/abs/1706.03762"
    assert [s.title for s in paper.sections] == ["1 Introduction", "2 Method"]


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
