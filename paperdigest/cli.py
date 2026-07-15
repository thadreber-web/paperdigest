from __future__ import annotations

import sys
from pathlib import Path

import typer
from typer.core import TyperGroup

try:
    from click import UsageError
except ImportError:  # typer >=0.16 vendors click internally as typer._click
    from typer._click.exceptions import UsageError

from . import digest as digest_mod
from . import extract, fetch, render
from . import scaffold as scaffold_mod
from .config import load_config
from .llm import LLMError, make_backend


class _DefaultDigestGroup(TyperGroup):
    """Bare `paperdigest <ref>` keeps working: unknown first arg falls back to `digest`."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except UsageError:
            return super().resolve_command(ctx, ["digest", *args])


app = typer.Typer(add_completion=False, cls=_DefaultDigestGroup)


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)


@app.command()
def digest(
    ref: str = typer.Argument(..., help="arXiv URL or ID, e.g. https://arxiv.org/abs/1706.03762"),
    level: str = typer.Option(None, "--level", help="beginner | intermediate | advanced"),
    backend: str = typer.Option(None, "--backend", help="local (default) | anthropic | openai"),
    model: str = typer.Option(None, "--model"),
    base_url: str = typer.Option(None, "--base-url", help="local server URL (default http://localhost:8080/v1)"),
    vault: Path = typer.Option(None, "--vault", help="Obsidian vault root (notes go to Papers/ and Glossary/)"),
    diagram: str = typer.Option(None, "--diagram", help="mermaid (default) | ascii"),
    config: Path = typer.Option(Path("config.toml"), "--config"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing paper folder"),
    refresh: bool = typer.Option(False, "--refresh", help="re-fetch the paper, ignoring the HTML cache"),
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
            diagram=diagram,
        )
        arxiv_id = fetch.parse_arxiv_id(ref)
        _progress(f"Fetching arXiv {arxiv_id}...")
        html = fetch.fetch_html(arxiv_id, cfg.cache_dir, refresh=refresh)
        paper = extract.extract_paper(html, arxiv_id)
        _progress(f"Extracted '{paper.title}' ({len(paper.sections)} sections)")
        render.check_output_free(render.paper_folder(arxiv_id, paper.title, cfg.vault), force)
        llm_backend = make_backend(cfg.backend, cfg.model, cfg.base_url)
        gdir = cfg.vault / "Glossary"
        existing_terms = {p.stem.lower() for p in gdir.glob("*.md")} if gdir.exists() else set()
        d = digest_mod.build_digest(
            paper, llm_backend, cfg.level, existing_terms, cfg.max_input_chars,
            progress=_progress, diagram=cfg.diagram,
        )
        folder = render.render_digest(d, cfg.vault, force=force)
        print(folder)
    except (
        fetch.FetchError, extract.ExtractError, LLMError, render.OutputExistsError, ValueError, FileExistsError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(1)


@app.command()
def scaffold(
    ref: str = typer.Argument(..., help="arXiv URL or ID, e.g. https://arxiv.org/abs/1706.03762"),
    dest: Path = typer.Option(..., "--dest", help="directory the project folder is created under"),
    backend: str = typer.Option(None, "--backend", help="local (default) | anthropic | openai"),
    model: str = typer.Option(None, "--model"),
    base_url: str = typer.Option(None, "--base-url", help="local server URL (default http://localhost:8080/v1)"),
    config: Path = typer.Option(Path("config.toml"), "--config"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing project folder"),
    refresh: bool = typer.Option(False, "--refresh", help="re-fetch the paper, ignoring the HTML cache"),
):
    """Scaffold a git-initialized research project (structure, docs, stubs) from a paper."""
    try:
        cfg = load_config(config, backend=backend, model=model, base_url=base_url)
        arxiv_id = fetch.parse_arxiv_id(ref)
        _progress(f"Fetching arXiv {arxiv_id}...")
        html = fetch.fetch_html(arxiv_id, cfg.cache_dir, refresh=refresh)
        paper = extract.extract_paper(html, arxiv_id)
        _progress(f"Extracted '{paper.title}' ({len(paper.sections)} sections)")
        folder = scaffold_mod.project_folder(arxiv_id, paper.title, dest)
        render.check_output_free(folder, force)
        llm_backend = make_backend(cfg.backend, cfg.model, cfg.base_url)
        project = scaffold_mod.build_scaffold(paper, llm_backend, cfg.max_input_chars, progress=_progress)
        out = scaffold_mod.write_project(project, folder, force=force, progress=_progress)
        print(out)
    except scaffold_mod.ScaffoldError as e:
        safe_stage = e.stage.replace(":", "-").replace("/", "-")
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        debug = cfg.cache_dir / f"scaffold-debug-{arxiv_id}-{safe_stage}.txt"
        debug.write_text(e.raw or "(no raw model output captured)")
        bar = "=" * 64
        print(bar, file=sys.stderr)
        print("ERROR: scaffold stage failed — aborting, nothing was written", file=sys.stderr)
        print(f"  stage:  {e.stage}", file=sys.stderr)
        print(f"  model:  {cfg.model}", file=sys.stderr)
        print(f"  server: {cfg.base_url or '(backend default)'}", file=sys.stderr)
        print(f"  cause:  {e}", file=sys.stderr)
        print(f"  raw model output saved to: {debug.resolve()}", file=sys.stderr)
        print(bar, file=sys.stderr)
        raise typer.Exit(1)
    except (
        fetch.FetchError, extract.ExtractError, LLMError, render.OutputExistsError, ValueError, FileExistsError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(1)
