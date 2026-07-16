from __future__ import annotations

from dataclasses import dataclass, field

from bs4 import BeautifulSoup


class ExtractError(Exception):
    pass


@dataclass
class Section:
    title: str
    text: str


@dataclass
class Figure:
    caption: str
    src: str
    section: str | None


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    sections: list[Section]
    url: str
    figures: list[Figure] = field(default_factory=list)


_RASTER_EXTS = (".png", ".jpg", ".jpeg")


def _extract_figures(soup: BeautifulSoup) -> list[Figure]:
    figures: list[Figure] = []
    for fig in soup.find_all("figure", class_="ltx_figure"):
        img = fig.find("img")
        if img is None:
            continue
        src = img.get("src", "")
        if not src.lower().endswith(_RASTER_EXTS):
            continue
        caption_el = fig.find(class_="ltx_caption")
        if caption_el is None:
            continue
        caption = caption_el.get_text(" ", strip=True)
        if not caption:
            continue
        section = None
        sec_el = fig.find_parent("section", class_="ltx_section")
        if sec_el is not None:
            heading = sec_el.find(["h2", "h3"])
            if heading is not None:
                section = heading.get_text(" ", strip=True)
        figures.append(Figure(caption=caption, src=src, section=section))
    return figures


def _mathify(soup: BeautifulSoup) -> None:
    for math in soup.find_all("math"):
        latex = math.get("alttext", "")
        math.replace_with(f"${latex}$" if latex else "")


def extract_paper(html: str, arxiv_id: str) -> Paper:
    soup = BeautifulSoup(html, "html.parser")
    _mathify(soup)

    title_el = soup.find(class_="ltx_title_document")
    if title_el is None:
        raise ExtractError("could not find a paper title — is this arXiv LaTeXML HTML?")
    title = title_el.get_text(" ", strip=True)

    abstract = ""
    abstract_el = soup.find(class_="ltx_abstract")
    if abstract_el is not None:
        heading = abstract_el.find(["h6", "h2", "h3"])
        if heading:
            heading.extract()
        abstract = abstract_el.get_text(" ", strip=True)

    figures = _extract_figures(soup)

    sections: list[Section] = []
    for sec in soup.find_all("section", class_="ltx_section"):
        heading = sec.find(["h2", "h3"])
        sec_title = heading.get_text(" ", strip=True) if heading else "Untitled"
        if heading:
            heading.extract()
        low = sec_title.lower()
        if "bib" in (sec.get("id") or "") or low.startswith(("references", "bibliography", "acknowledg")):
            continue
        text = sec.get_text(" ", strip=True)
        if text:
            sections.append(Section(title=sec_title, text=text))
    if not sections:
        raise ExtractError("no sections found — is this arXiv LaTeXML HTML?")

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        sections=sections,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        figures=figures,
    )
