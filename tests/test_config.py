from pathlib import Path

import pytest

from paperdigest.config import load_config


def test_defaults_are_local_first():
    cfg = load_config(None)
    assert cfg.backend == "local"
    assert cfg.model == "local"
    assert cfg.level == "intermediate"
    assert cfg.vault == Path("/vault")
    assert cfg.base_url is None


def test_model_defaults_follow_backend():
    assert load_config(None, backend="anthropic").model == "claude-sonnet-5"
    assert load_config(None, backend="openai").model == "gpt-5"
    assert load_config(None, backend="anthropic", model="claude-haiku-4-5").model == "claude-haiku-4-5"


def test_toml_file_overrides_defaults(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('backend = "openai"\nvault = "/data/vault"\nlevel = "beginner"\n')
    cfg = load_config(f)
    assert cfg.backend == "openai"
    assert cfg.vault == Path("/data/vault")
    assert cfg.level == "beginner"


def test_tilde_paths_are_expanded(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('vault = "~/vault"\ncache_dir = "~/.cache/pd"\n')
    cfg = load_config(f)
    assert cfg.vault == Path.home() / "vault"
    assert cfg.cache_dir == Path.home() / ".cache" / "pd"


def test_kwargs_override_toml_and_none_is_ignored(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('level = "beginner"\n')
    cfg = load_config(f, level="advanced", backend=None)
    assert cfg.level == "advanced"
    assert cfg.backend == "local"


def test_invalid_level_raises():
    with pytest.raises(ValueError, match="level"):
        load_config(None, level="expert")


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        load_config(None, backend="gemini")


def test_unknown_toml_key_raises(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('api_key = "sk-nope"\n')
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(f)


def test_diagram_defaults_to_mermaid():
    assert load_config(None).diagram == "mermaid"


def test_invalid_diagram_raises():
    with pytest.raises(ValueError, match="diagram"):
        load_config(None, diagram="png")


def test_max_tokens_defaults_and_overrides(tmp_path):
    assert load_config(None).max_tokens == 8192
    f = tmp_path / "config.toml"
    f.write_text("max_tokens = 2048\n")
    assert load_config(f).max_tokens == 2048


def test_figures_defaults_and_overrides(tmp_path):
    assert load_config(None).figures is True
    f = tmp_path / "config.toml"
    f.write_text("figures = false\n")
    assert load_config(f).figures is False
    assert load_config(None, figures=False).figures is False


def test_max_figures_defaults_and_overrides(tmp_path):
    assert load_config(None).max_figures == 8
    f = tmp_path / "config.toml"
    f.write_text("max_figures = 3\n")
    assert load_config(f).max_figures == 3
    assert load_config(None, max_figures=2).max_figures == 2
