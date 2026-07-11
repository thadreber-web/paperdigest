from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

LEVELS = ("beginner", "intermediate", "advanced")
BACKENDS = ("anthropic", "openai")


@dataclass
class Config:
    backend: str = "anthropic"
    model: str = "claude-sonnet-5"
    base_url: str | None = None  # for local OpenAI-compatible servers (Ollama/vLLM)
    vault: Path = Path("/vault")
    level: str = "intermediate"
    max_input_chars: int = 400_000  # lower this for small-context local models
    cache_dir: Path = Path.home() / ".cache" / "paperdigest"


def load_config(path: Path | None = None, **overrides) -> Config:
    data: dict = {}
    if path is not None and path.exists():
        data = tomllib.loads(path.read_text())
    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    merged = {**data, **{k: v for k, v in overrides.items() if v is not None}}
    for key in ("vault", "cache_dir"):
        if key in merged:
            merged[key] = Path(merged[key])
    cfg = replace(Config(), **merged)
    if cfg.level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {cfg.level!r}")
    if cfg.backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {cfg.backend!r}")
    return cfg
