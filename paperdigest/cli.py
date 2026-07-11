from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import digest as digest_mod
from . import extract, fetch, render
from .config import load_config
from .llm import LLMError, make_backend

app = typer.Typer(add_completion=False)


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)


@app.command()
def main(
    ref: str = typer.Argument(..., help="arXiv URL or ID, e.g. https://arxiv.org/abs/1706.03762"),
    level: str = typer.Option(None, "--level", help="beginner | intermediate | advanced"),
    backend: str = typer.Option(None, "--backend", help="anthropic | openai (openai also covers local servers)"),
    model: str = typer.Option(None, "--model"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible server URL, e.g. http://localhost:11434/v1"),
    vault: Path = typer.Option(None, "--vault", help="Obsidian vault root (notes go to Papers/ and Glossary/)"),
    config: Path = typer.Option(Path("config.toml"), "--config"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing paper folder"),
):
    """Digest an AI/ML paper into plain-English Obsidian notes."""
    try:
        cfg = load_config(
            config,
            level=level,
            backend=backend,
            model=model,
            base_url=base_url,
            vault=vault,
        )
        arxiv_id = fetch.parse_arxiv_id(ref)
        _progress(f"Fetching arXiv {arxiv_id}...")
        html = fetch.fetch_html(arxiv_id, cfg.cache_dir)
        paper = extract.extract_paper(html, arxiv_id)
        _progress(f"Extracted '{paper.title}' ({len(paper.sections)} sections)")
        render.check_output_free(render.paper_folder(arxiv_id, paper.title, cfg.vault), force)
        llm_backend = make_backend(cfg.backend, cfg.model, cfg.base_url)
        gdir = cfg.vault / "Glossary"
        existing_terms = {p.stem.lower() for p in gdir.glob("*.md")} if gdir.exists() else set()
        d = digest_mod.build_digest(
            paper, llm_backend, cfg.level, existing_terms, cfg.max_input_chars, progress=_progress
        )
        folder = render.render_digest(d, cfg.vault, force=force)
        print(folder)
    except (fetch.FetchError, extract.ExtractError, LLMError, render.OutputExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(1)
