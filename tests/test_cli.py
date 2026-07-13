import json

from typer.testing import CliRunner

from paperdigest import cli
from conftest import FakeBackend  # tests/ is on sys.path under pytest

runner = CliRunner()

OUTLINE = json.dumps(
    {
        "tldr": "Tiny transformers work.",
        "why_it_matters": "Cheap models are useful.",
        "concepts": [{"title": "Self-Attention", "section": "1 Introduction"}],
        "jargon": ["attention"],
        "self_test": ["What is attention?"],
    }
)
GLOSSARY = json.dumps({"terms": {"attention": "A weighting scheme."}})


def _fake_responses():
    return [OUTLINE, "Attention body.", GLOSSARY]


def _patch(monkeypatch, fixture_html):
    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: FakeBackend(_fake_responses()))


def test_end_to_end_offline(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "Papers" / "2017-tiny-transformers-explained"
    assert (folder / "00 Overview.md").exists()
    assert (folder / "01 Self-Attention.md").exists()
    assert (tmp_path / "Glossary" / "attention.md").exists()


def test_rerun_refuses_without_force(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    assert runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)]).exit_code == 0
    _patch(monkeypatch, fixture_html)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_rerun_guard_fires_before_llm_calls(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    assert runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)]).exit_code == 0

    def boom(*a, **k):
        raise AssertionError("LLM backend should not be constructed when output exists")

    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", boom)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_rerun_with_force_succeeds(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html)
    assert runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)]).exit_code == 0
    _patch(monkeypatch, fixture_html)
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output


def test_existing_glossary_terms_skip_llm_definitions(tmp_path, monkeypatch, fixture_html):
    gdir = tmp_path / "Glossary"
    gdir.mkdir(parents=True)
    (gdir / "attention.md").write_text("MINE")
    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: FakeBackend([OUTLINE, "Attention body."]))
    result = runner.invoke(cli.app, ["1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (gdir / "attention.md").read_text() == "MINE"


def test_bad_ref_errors_cleanly(tmp_path):
    result = runner.invoke(cli.app, ["https://example.com/nope", "--vault", str(tmp_path)])
    assert result.exit_code == 1
    assert "error:" in result.output
