from __future__ import annotations

import re
from pathlib import Path

import httpx


class FetchError(Exception):
    pass


_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/)?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)

_SOURCES = (
    "https://arxiv.org/html/{id}",
    "https://ar5iv.labs.arxiv.org/html/{id}",
)


def parse_arxiv_id(ref: str) -> str:
    ref = ref.strip()
    if "/" in ref and "arxiv.org" not in ref.lower():
        raise FetchError(f"not an arXiv reference: {ref!r} (v1 accepts arXiv URLs or IDs only)")
    m = _ID_RE.search(ref)
    if not m:
        raise FetchError(f"not an arXiv reference: {ref!r} (v1 accepts arXiv URLs or IDs only)")
    return m.group(1) + (m.group(2) or "")


def fetch_html(arxiv_id: str, cache_dir: Path, client: httpx.Client | None = None) -> str:
    cache_file = cache_dir / f"{arxiv_id}.html"
    if cache_file.exists():
        return cache_file.read_text()
    client = client or httpx.Client(follow_redirects=True, timeout=30)
    errors = []
    for tpl in _SOURCES:
        url = tpl.format(id=arxiv_id)
        try:
            resp = client.get(url)
        except httpx.HTTPError as e:
            errors.append(f"{url} -> {e}")
            continue
        if resp.status_code == 200 and "ltx_title" in resp.text:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text)
            return resp.text
        errors.append(f"{url} -> HTTP {resp.status_code}")
    raise FetchError(
        "No HTML rendering available for this paper (v1 supports arXiv HTML only).\n"
        "Tried:\n  " + "\n  ".join(errors)
    )
