import json
import subprocess

from conftest import FakeBackend
from test_scaffold_stages import full_responses  # tests/ is on sys.path under pytest
from typer.testing import CliRunner

from paperdigest import cli

runner = CliRunner()


def _patch(monkeypatch, fixture_html, responses):
    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None, refresh=False: fixture_html)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: FakeBackend(responses))


def test_scaffold_end_to_end(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html, full_responses())
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "2017-tiny-transformers-explained"
    assert (folder / "README.md").exists()
    assert (folder / "train.py").exists()
    assert (folder / ".git").is_dir()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=folder, capture_output=True, text=True, check=True
    ).stdout
    assert "paperdigest" in log


def test_scaffold_stage_failure_is_loud(tmp_path, monkeypatch, fixture_html):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(f'cache_dir = "{tmp_path / "cache"}"\n')
    _patch(monkeypatch, fixture_html, ["this is not json"])
    dest = tmp_path / "projects"
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(dest)])
    assert result.exit_code == 1
    assert "ERROR" in result.output
    assert "stage:" in result.output and "analyze" in result.output
    debug = tmp_path / "cache" / "scaffold-debug-1706.03762-analyze.txt"
    assert debug.exists()
    assert debug.read_text() == "this is not json"
    assert not (dest / "2017-tiny-transformers-explained").exists()


def test_scaffold_refuses_existing_before_llm(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html, full_responses())
    assert runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)]).exit_code == 0

    def boom(*a, **k):
        raise AssertionError("LLM backend must not be constructed when output exists")

    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None, refresh=False: fixture_html)
    monkeypatch.setattr(cli, "make_backend", boom)
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_scaffold_force_overwrites(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html, full_responses())
    assert runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)]).exit_code == 0
    _patch(monkeypatch, fixture_html, full_responses())
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output


def test_explicit_digest_subcommand_works(tmp_path, monkeypatch, fixture_html):
    outline = json.dumps(
        {
            "tldr": "Tiny transformers work.",
            "why_it_matters": "Cheap models are useful.",
            "concepts": [{"title": "Self-Attention", "section": "1 Introduction"}],
            "jargon": ["attention"],
            "self_test": ["What is attention?"],
        }
    )
    glossary = json.dumps({"terms": {"attention": "A weighting scheme."}})
    _patch(monkeypatch, fixture_html, [outline, "Attention body.", glossary])
    result = runner.invoke(cli.app, ["digest", "1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "Papers" / "2017-tiny-transformers-explained" / "00 Overview.md").exists()
