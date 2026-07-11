# paperdigest

Turn an AI/ML arXiv paper into a folder of plain-English, wikilinked Obsidian
notes: concept-by-concept explainers, ASCII diagrams, equations walked through
line by line, and a vault-wide jargon glossary.

## Quick start (local install)

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/paperdigest https://arxiv.org/abs/1706.03762 --vault ~/ObsidianVault --level intermediate
```

Output lands in `<vault>/Papers/<year>-<title-slug>/` plus new term notes in
`<vault>/Glossary/`. Re-running the same paper refuses to overwrite unless you
pass `--force`. Glossary term notes are never overwritten.

## Docker

```bash
docker build -t paperdigest .
docker run --rm -v ~/ObsidianVault:/vault -e ANTHROPIC_API_KEY \
  paperdigest https://arxiv.org/abs/1706.03762 --level beginner
```

## Backends

| Backend | Flags | Key |
|---|---|---|
| Claude (default) | `--backend anthropic --model claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `--backend openai --model gpt-5` | `OPENAI_API_KEY` |
| Local (Ollama/vLLM) | `--backend openai --model llama3.1 --base-url http://localhost:11434/v1` | none |

For local models with small context windows, set `max_input_chars` (e.g.
24000) in `config.toml` — long papers get proportionally trimmed with a
warning. From Docker, reach a host-side local server with
`--base-url http://host.docker.internal:11434/v1` (add
`--add-host=host.docker.internal:host-gateway` on Linux).

## Config

Copy `config.example.toml` to `config.toml` (or pass `--config`). CLI flags
override config values. API keys are env-only.

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
