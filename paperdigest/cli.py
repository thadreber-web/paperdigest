from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import typer
from typer.core import TyperGroup

try:
    from click import UsageError
except ImportError:  # typer >=0.16 vendors click internally as typer._click
    from typer._click.exceptions import UsageError

from . import __version__, extract, fetch, render
from . import digest as digest_mod
from . import scaffold as scaffold_mod
from .config import Config, load_config
from .extract import Paper
from .llm import LLMError, make_backend


class _DefaultDigestGroup(TyperGroup):
    """Bare `paperdigest <ref>` keeps working: unknown first arg falls back to `digest`."""

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except UsageError:
            return super().resolve_command(ctx, ["digest", *args])


app = typer.Typer(add_completion=False, cls=_DefaultDigestGroup)


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="show version and exit"
    ),
) -> None:
    pass


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)


def _no_progress(msg: str) -> None:
    pass


def _fetch_and_extract(
    cfg: Config, ref: str, refresh: bool, progress: Callable[[str], None] = _progress
) -> tuple[str, Paper]:
    arxiv_id = fetch.parse_arxiv_id(ref)
    progress(f"Fetching arXiv {arxiv_id}...")
    html = fetch.fetch_html(arxiv_id, cfg.cache_dir, refresh=refresh)
    paper = extract.extract_paper(html, arxiv_id)
    progress(f"Extracted '{paper.title}' ({len(paper.sections)} sections)")
    return arxiv_id, paper


@contextmanager
def _handle_common_errors():
    try:
        yield
    except (
        fetch.FetchError, extract.ExtractError, LLMError, render.OutputExistsError, ValueError, FileExistsError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(1)


@app.command()
def digest(
    ref: str = typer.Argument(..., help="arXiv URL or ID, e.g. https://arxiv.org/abs/1706.03762"),
    level: str = typer.Option(None, "--level", help="beginner | intermediate | advanced"),
    backend: str = typer.Option(None, "--backend", help="local (default) | anthropic | openai"),
    model: str = typer.Option(None, "--model"),
    base_url: str = typer.Option(None, "--base-url", help="local server URL (default http://localhost:8080/v1)"),
    vault: Path = typer.Option(None, "--vault", help="Obsidian vault root (notes go to Papers/ and Glossary/)"),
    diagram: str = typer.Option(None, "--diagram", help="mermaid (default) | ascii"),
    figures: bool = typer.Option(
        None, "--figures/--no-figures",
        help="download and explain paper figures; needs a vision-capable backend (default from config, on)",
    ),
    config: Path = typer.Option(Path("config.toml"), "--config"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing paper folder"),
    refresh: bool = typer.Option(False, "--refresh", help="re-fetch the paper, ignoring the HTML cache"),
    quiet: bool = typer.Option(False, "--quiet", help="suppress progress output"),
    max_input_chars: int = typer.Option(None, "--max-input-chars", help="paper body char budget before trimming"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="LLM output token cap"),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="directory for cached fetched HTML"),
):
    """Digest an AI/ML paper into plain-English Obsidian notes."""
    progress = _no_progress if quiet else _progress
    with _handle_common_errors():
        cfg = load_config(
            config,
            level=level,
            backend=backend,
            model=model,
            base_url=base_url,
            vault=vault,
            diagram=diagram,
            figures=figures,
            max_input_chars=max_input_chars,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
        )
        arxiv_id, paper = _fetch_and_extract(cfg, ref, refresh, progress)
        render.check_output_free(render.paper_folder(arxiv_id, paper.title, cfg.vault), force)
        llm_backend = make_backend(cfg.backend, cfg.model, cfg.base_url, cfg.max_tokens)
        gdir = cfg.vault / "Glossary"
        existing_terms = {p.stem.lower() for p in gdir.glob("*.md")} if gdir.exists() else set()
        figure_paths = None
        if cfg.figures and paper.figures:
            capped_figures = paper.figures[: cfg.max_figures]
            progress(f"Fetching {len(capped_figures)} figures...")
            figure_paths = fetch.fetch_figures(
                capped_figures, arxiv_id, cfg.cache_dir,
                base_url=fetch.html_base_url(arxiv_id, cfg.cache_dir), refresh=refresh,
            )
        d = digest_mod.build_digest(
            paper, llm_backend, cfg.level, existing_terms, cfg.max_input_chars,
            progress=progress, diagram=cfg.diagram,
            workers=4 if cfg.backend != "local" else 1,
            figure_paths=figure_paths,
        )
        folder = render.render_digest(d, cfg.vault, force=force)
        print(folder)


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
    quiet: bool = typer.Option(False, "--quiet", help="suppress progress output"),
    max_input_chars: int = typer.Option(None, "--max-input-chars", help="paper body char budget before trimming"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="LLM output token cap"),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="directory for cached fetched HTML"),
):
    """Scaffold a git-initialized research project (structure, docs, stubs) from a paper."""
    progress = _no_progress if quiet else _progress
    try:
        with _handle_common_errors():
            cfg = load_config(
                config,
                backend=backend,
                model=model,
                base_url=base_url,
                max_input_chars=max_input_chars,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
            )
            arxiv_id, paper = _fetch_and_extract(cfg, ref, refresh, progress)
            folder = scaffold_mod.project_folder(arxiv_id, paper.title, dest)
            render.check_output_free(folder, force)
            llm_backend = make_backend(cfg.backend, cfg.model, cfg.base_url, cfg.max_tokens)
            project = scaffold_mod.build_scaffold(
                paper, llm_backend, cfg.max_input_chars, progress=progress,
                workers=4 if cfg.backend != "local" else 1,
            )
            out = scaffold_mod.write_project(project, folder, force=force, progress=progress)
            print(out)
    except scaffold_mod.ScaffoldError as e:
        safe_stage = e.stage.replace(":", "-").replace("/", "-")
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        debug = cfg.cache_dir / f"scaffold-debug-{arxiv_id}-{safe_stage}.txt"
        debug.write_text(e.raw or "(no raw model output captured)")
        bar = "=" * 64
        print(bar, file=sys.stderr)
        print("ERROR: scaffold stage failed — aborting, nothing was written", file=sys.stderr)
        print(f"  paperdigest: {__version__}", file=sys.stderr)
        print(f"  stage:  {e.stage}", file=sys.stderr)
        print(f"  model:  {cfg.model}", file=sys.stderr)
        print(f"  server: {cfg.base_url or '(backend default)'}", file=sys.stderr)
        print(f"  cause:  {e}", file=sys.stderr)
        print(f"  raw model output saved to: {debug.resolve()}", file=sys.stderr)
        print(bar, file=sys.stderr)
        raise typer.Exit(1)
