import httpx
import pytest

from paperdigest.extract import Figure
from paperdigest.fetch import FetchError, fetch_figures, fetch_html, html_base_url, parse_arxiv_id


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("1706.03762", "1706.03762"),
        ("1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/html/2405.12345v1", "2405.12345v1"),
    ],
)
def test_parse_arxiv_id(ref, expected):
    assert parse_arxiv_id(ref) == expected


def test_parse_rejects_non_arxiv():
    with pytest.raises(FetchError):
        parse_arxiv_id("https://example.com/1234.5678.pdf")
    with pytest.raises(FetchError):
        parse_arxiv_id("not a paper")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_uses_arxiv_html(tmp_path):
    def handler(request):
        assert "arxiv.org/html/1706.03762" in str(request.url)
        return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')

    html = fetch_html("1706.03762", tmp_path, client=_client(handler))
    assert "ltx_title" in html
    assert (tmp_path / "1706.03762.html").read_text() == html


def test_fetch_falls_back_to_ar5iv(tmp_path):
    def handler(request):
        if "ar5iv" in str(request.url):
            return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')
        return httpx.Response(404)

    html = fetch_html("1706.03762", tmp_path, client=_client(handler))
    assert "ltx_title" in html


def test_fetch_rejects_non_latexml_page(tmp_path):
    def handler(request):
        return httpx.Response(200, text="<html>abstract landing page</html>")

    with pytest.raises(FetchError, match="No HTML rendering"):
        fetch_html("1706.03762", tmp_path, client=_client(handler))


def test_fetch_reads_cache_without_network(tmp_path):
    (tmp_path / "1706.03762.html").write_text("cached!")

    def handler(request):
        raise AssertionError("network should not be hit")

    assert fetch_html("1706.03762", tmp_path, client=_client(handler)) == "cached!"


def test_fetch_refresh_bypasses_cache(tmp_path):
    (tmp_path / "1706.03762.html").write_text("stale cache")

    def handler(request):
        return httpx.Response(200, text='<h1 class="ltx_title">Fresh</h1>')

    html = fetch_html("1706.03762", tmp_path, client=_client(handler), refresh=True)
    assert html == '<h1 class="ltx_title">Fresh</h1>'
    assert (tmp_path / "1706.03762.html").read_text() == html


def test_fetch_cache_write_leaves_no_leftover_temp_file(tmp_path):
    def handler(request):
        return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')

    fetch_html("1706.03762", tmp_path, client=_client(handler))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["1706.03762.html", "1706.03762.url"]


def test_fetch_html_writes_url_sidecar(tmp_path):
    def handler(request):
        return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')

    fetch_html("1706.03762", tmp_path, client=_client(handler))
    sidecar = tmp_path / "1706.03762.url"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == "https://arxiv.org/html/1706.03762"


def test_fetch_html_writes_url_sidecar_for_fallback_source(tmp_path):
    def handler(request):
        if "ar5iv" in str(request.url):
            return httpx.Response(200, text='<h1 class="ltx_title">Hi</h1>')
        return httpx.Response(404)

    fetch_html("1706.03762", tmp_path, client=_client(handler))
    sidecar = tmp_path / "1706.03762.url"
    assert "ar5iv" in sidecar.read_text()


def test_html_base_url_reads_sidecar(tmp_path):
    (tmp_path / "1706.03762.url").write_text("https://ar5iv.labs.arxiv.org/html/1706.03762")
    assert html_base_url("1706.03762", tmp_path) == "https://ar5iv.labs.arxiv.org/html/1706.03762"


def test_html_base_url_falls_back_when_sidecar_missing(tmp_path):
    assert html_base_url("1706.03762", tmp_path) == "https://arxiv.org/html/1706.03762"


def test_fetch_figures_writes_to_cache_path(tmp_path):
    fig = Figure(caption="Fig 1", src="figure1.png", section=None)

    def handler(request):
        return httpx.Response(200, content=b"\x89PNGfakebytes")

    result = fetch_figures(
        [fig], "1706.03762", tmp_path, "https://arxiv.org/html/1706.03762v7/", client=_client(handler)
    )
    dest = tmp_path / "1706.03762" / "figures" / "fig1.png"
    assert result == {"figure1.png": dest}
    assert dest.read_bytes() == b"\x89PNGfakebytes"


def test_fetch_figures_resolves_relative_src_against_base_url(tmp_path):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, content=b"data")

    fig = Figure(caption="Fig 1", src="figures/fig1.png", section=None)
    fetch_figures([fig], "1706.03762", tmp_path, "https://arxiv.org/html/1706.03762v7/", client=_client(handler))
    assert seen["url"] == "https://arxiv.org/html/1706.03762v7/figures/fig1.png"


def test_fetch_figures_skips_unsupported_extension_without_network(tmp_path):
    def handler(request):
        raise AssertionError("network should not be hit for a non-raster figure")

    fig = Figure(caption="Fig 2", src="figure2.svg", section=None)
    base_url = "https://arxiv.org/html/1706.03762v7/"
    result = fetch_figures([fig], "1706.03762", tmp_path, base_url, client=_client(handler))
    assert result == {}


def test_fetch_figures_skips_on_download_failure_without_raising(tmp_path):
    def handler(request):
        return httpx.Response(404)

    fig = Figure(caption="Fig 1", src="figure1.png", section=None)
    base_url = "https://arxiv.org/html/1706.03762v7/"
    result = fetch_figures([fig], "1706.03762", tmp_path, base_url, client=_client(handler))
    assert result == {}
    assert not (tmp_path / "1706.03762" / "figures").exists()


def test_fetch_figures_respects_refresh(tmp_path):
    dest_dir = tmp_path / "1706.03762" / "figures"
    dest_dir.mkdir(parents=True)
    (dest_dir / "fig1.png").write_bytes(b"stale")

    def handler(request):
        return httpx.Response(200, content=b"fresh")

    fig = Figure(caption="Fig 1", src="figure1.png", section=None)

    def no_network(request):
        raise AssertionError("network should not be hit when cache exists and refresh=False")

    base_url = "https://arxiv.org/html/1706.03762v7/"
    result = fetch_figures([fig], "1706.03762", tmp_path, base_url, client=_client(no_network))
    assert result[fig.src].read_bytes() == b"stale"

    result = fetch_figures(
        [fig], "1706.03762", tmp_path, "https://arxiv.org/html/1706.03762v7/", client=_client(handler), refresh=True
    )
    assert result[fig.src].read_bytes() == b"fresh"
