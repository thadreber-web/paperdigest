from pathlib import Path

import pytest

from paperdigest.config import Config, load_config


def test_defaults():
    cfg = load_config(None)
    assert cfg.backend == "anthropic"
    assert cfg.level == "intermediate"
    assert cfg.vault == Path("/vault")
    assert cfg.base_url is None


def test_toml_file_overrides_defaults(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('backend = "openai"\nvault = "/data/vault"\nlevel = "beginner"\n')
    cfg = load_config(f)
    assert cfg.backend == "openai"
    assert cfg.vault == Path("/data/vault")
    assert cfg.level == "beginner"


def test_kwargs_override_toml_and_none_is_ignored(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('level = "beginner"\n')
    cfg = load_config(f, level="advanced", backend=None)
    assert cfg.level == "advanced"
    assert cfg.backend == "anthropic"


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
