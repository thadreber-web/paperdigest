from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from .extract import Figure


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


def _write_cache_atomically(cache_file: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=cache_file.parent, prefix=f".{cache_file.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_name, cache_file)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _write_cache_atomically_bytes(cache_file: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=cache_file.parent, prefix=f".{cache_file.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, cache_file)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


_FIGURE_EXTS = (".png", ".jpg", ".jpeg")


def fetch_figures(
    figures: list[Figure],
    arxiv_id: str,
    cache_dir: Path,
    base_url: str,
    client: httpx.Client | None = None,
    *,
    refresh: bool = False,
) -> dict[str, Path]:
    """Download figure images, keyed by Figure.src, into <cache_dir>/<arxiv_id>/figures/.

    `base_url` is the final URL of the fetched HTML document (post-redirect/fallback) —
    figure `src` attributes in arXiv HTML are relative to it, not to a guessed path.
    A single figure's download failure is skipped with a warning, never raised — figures
    are an enhancement, not core to the digest.
    """
    out_dir = cache_dir / arxiv_id / "figures"
    client = client or httpx.Client(follow_redirects=True, timeout=30)
    results: dict[str, Path] = {}
    for i, fig in enumerate(figures, start=1):
        ext = Path(urlparse(fig.src).path).suffix.lower()
        if ext not in _FIGURE_EXTS:
            print(f"Warning: skipping figure {i} (unsupported format {ext or '?'}): {fig.src}", file=sys.stderr)
            continue
        dest = out_dir / f"fig{i}{ext}"
        if not refresh and dest.exists():
            results[fig.src] = dest
            continue
        url = urljoin(base_url, fig.src)
        try:
            resp = client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            print(f"Warning: could not fetch figure {i} ({url}): {e}", file=sys.stderr)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_cache_atomically_bytes(dest, resp.content)
        results[fig.src] = dest
    return results


def _url_sidecar_file(arxiv_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{arxiv_id}.url"


def html_base_url(arxiv_id: str, cache_dir: Path) -> str:
    """Return the source URL fetch_html last succeeded with, for resolving relative figure src.

    Falls back to the primary arXiv HTML URL template if the sidecar is missing
    (e.g. pre-existing caches written before this sidecar existed).
    """
    sidecar = _url_sidecar_file(arxiv_id, cache_dir)
    if sidecar.exists():
        url = sidecar.read_text().strip()
        if url:
            return url
    return _SOURCES[0].format(id=arxiv_id)


def fetch_html(arxiv_id: str, cache_dir: Path, client: httpx.Client | None = None, *, refresh: bool = False) -> str:
    cache_file = cache_dir / f"{arxiv_id}.html"
    if not refresh and cache_file.exists():
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
            _write_cache_atomically(cache_file, resp.text)
            _write_cache_atomically(_url_sidecar_file(arxiv_id, cache_dir), url)
            return resp.text
        errors.append(f"{url} -> HTTP {resp.status_code}")
    raise FetchError(
        "No HTML rendering available for this paper (v1 supports arXiv HTML only).\n"
        "Tried:\n  " + "\n  ".join(errors)
    )
