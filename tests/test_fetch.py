import httpx
import pytest

from paperdigest.fetch import FetchError, fetch_html, parse_arxiv_id


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
    names = [p.name for p in tmp_path.iterdir()]
    assert names == ["1706.03762.html"]
