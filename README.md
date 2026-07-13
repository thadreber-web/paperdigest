# paperdigest

Turn an AI/ML arXiv paper into a folder of plain-English, wikilinked Obsidian
notes: concept-by-concept explainers, ASCII diagrams, equations walked through
line by line, and a vault-wide jargon glossary.

**Local-first.** By default paperdigest talks to a local OpenAI-compatible
server (llama.cpp, Ollama, vLLM) on your own machine — no API key, no cost.
Cloud backends (Anthropic, OpenAI) are opt-in for those who want them.

## Quick start (local, free)

Have a llama.cpp server running (`llama-server -m model.gguf` serves
`http://localhost:8080/v1` by default), then:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/paperdigest https://arxiv.org/abs/1706.03762 --vault ~/ObsidianVault
```

Non-default port or engine? Point at it:

```bash
# llama.cpp on another port
.venv/bin/paperdigest 1706.03762 --vault ~/ObsidianVault --base-url http://localhost:8001/v1
# Ollama (needs the real model name)
.venv/bin/paperdigest 1706.03762 --vault ~/ObsidianVault \
  --base-url http://localhost:11434/v1 --model llama3.1
```

Use a capable instruct model (~7B+); the pipeline needs structured JSON
output, and very small models fail cleanly with an `error:` message. Set
`max_input_chars` in `config.toml` to fit your model's context window (e.g.
24000 for ~8k-token contexts) — long papers are trimmed with a warning.

Output lands in `<vault>/Papers/<year>-<title-slug>/` plus new term notes in
`<vault>/Glossary/`. Re-running the same paper refuses to overwrite unless you
pass `--force`. Glossary term notes are never overwritten.

## Scaffold a research project

Turn a paper into a standalone, git-initialized project skeleton (Cookiecutter-DS-style
layout, JSONL experiment tracking, stub modules mapped to paper sections):

```bash
.venv/bin/paperdigest scaffold https://arxiv.org/abs/1706.03762 --dest ~/projects
```

Output lands in `<dest>/<year>-<title-slug>/` with its own git history. The code is a
*skeleton*: module stubs with `TODO(paper §x.y)` markers plus a complete train/evaluate
harness and a toy smoke experiment — not a working reimplementation. Any generation
failure aborts loudly (nothing half-written); the raw model output is saved to
`scaffold-debug-<stage>.txt` for inspection. Re-running refuses to overwrite an
existing project unless you pass `--force`.

You don't need a big model: scaffolding is tested live end-to-end on a 9B via
llama.cpp (see `docs/smoke-test-scaffold.md`). JSON stages request structured
output (`response_format: json_object`) from the server when supported, which
is what makes small local models reliable here; generated code is validated
(syntax + prompt-echo rejection) before anything is written.

## Cloud backends (optional, cost money)

| Backend | Flags | Key (env var only) |
|---|---|---|
| Local (default) | `--base-url http://localhost:8080/v1 --model local` | none |
| OpenAI | `--backend openai --model gpt-5` | `OPENAI_API_KEY` |
| Anthropic | `--backend anthropic --model claude-sonnet-5` | `ANTHROPIC_API_KEY` |

## Docker

```bash
docker build -t paperdigest .
docker run --rm -v ~/ObsidianVault:/vault \
  --add-host=host.docker.internal:host-gateway \
  paperdigest https://arxiv.org/abs/1706.03762 \
  --base-url http://host.docker.internal:8080/v1
```

(`--add-host` is needed on Linux to reach the host-side local server; for a
cloud backend pass `-e ANTHROPIC_API_KEY` or `-e OPENAI_API_KEY` instead.)

## Config

Copy `config.example.toml` to `config.toml` (or pass `--config`) so you don't
retype flags. CLI flags override config values. API keys are env-only and
never read from config files.

## Inputs

v1 accepts arXiv URLs (`abs`/`pdf`/`html`) or bare IDs (`1706.03762`). It uses
the paper's arXiv HTML rendering (ar5iv as fallback); papers with no HTML
rendering fail with a clear message. Fetched papers are cached in
`~/.cache/paperdigest/`.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # fully offline: fixtures + fake LLM backend
```
