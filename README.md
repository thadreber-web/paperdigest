# paperdigest

Turn an AI/ML arXiv paper into a folder of plain-English, wikilinked Obsidian
notes: concept-by-concept explainers, Mermaid diagrams, equations walked through
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
24000 for ~8k-token contexts) — long papers are trimmed with a warning; set
`max_tokens` alongside it to cap output length for the same small context.
Big *thinking* models need the opposite: reasoning tokens count against the
output cap, so if a run aborts with a `truncated at max_tokens` error, raise
it (e.g. `--max-tokens 32768` — see `docs/smoke-test-crosslinks.md` for a
live example on a 122B).

Output lands in `<vault>/Papers/<year>-<title-slug>/` plus new term notes in
`<vault>/Glossary/`. Re-running the same paper refuses to overwrite unless you
pass `--force`. Glossary term notes are never overwritten. Pass `--quiet` to
suppress progress output, `--version` to print the installed version, or
`--max-input-chars`/`--max-tokens`/`--cache-dir` to override those config
keys per run (both `digest` and `scaffold` accept all of these).

### Diagrams: Mermaid or ASCII

Structural concepts get a diagram. The default is **Mermaid** (renders
natively in Obsidian); pass `--diagram ascii` (or set `diagram = "ascii"` in
`config.toml`) for plain-text diagrams that can never have syntax errors.
Small local models sometimes emit Mermaid with unquoted special characters in
node labels, so every diagram goes through a deterministic sanitizer that
auto-quotes them. Optionally, if `node` is on your PATH and the `mermaid` npm
package is importable from the directory you run paperdigest in
(`npm install mermaid jsdom global-jsdom`), each diagram is also parse-checked
and a warning names any note whose diagram still fails. No node? Validation is
skipped silently — the worst case is one diagram rendering as an error box in
Obsidian instead of a picture.

### Figures

By default, figures are downloaded from the paper and explained by a
vision-capable model, with the image and its explanation embedded together
in the matching concept note (or the overview, if unmatched). This needs a
vision-capable backend — for local llama.cpp, serve with an `--mmproj`
projector file. Backends without vision skip figures automatically after a
warning on the first figure, and the rest of the digest completes normally.
Pass `--no-figures` (or set `figures = false` in `config.toml`) to turn this
off; it's also the budget option on cloud backends, where images are
token-expensive. `max_figures` (default 8) caps how many figures per paper
are explained, by document order.

### Linking to a scaffolded project

Pass `--project-dir` (or set `project_dir` in `config.toml`) and notes gain a
`## Build it` section pointing at `<project-dir>/<year>-<title-slug>` — the
project `scaffold` would create for the same paper. If that project exists on
disk, concept notes also get a `*Build it: …*` line linking the stub module(s)
whose cited `§x.y` paper sections match the concept, so you can jump straight
from the explanation to the code that implements it.

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
llama.cpp (see `docs/smoke-test-scaffold.md`; the cross-linking features are
live-tested on a 122B via vLLM in `docs/smoke-test-crosslinks.md`). JSON
stages request structured
output (`response_format: json_object`) from the server when supported, which
is what makes small local models reliable here; generated code is validated
(syntax + prompt-echo rejection) before anything is written. Besides
`log_run()` for JSONL experiment tracking, the scaffolded `tracking.py` has
`get_logger()` for per-run log files under `logs/`.

**A note on trust:** every generated `.py` file is still LLM output. Before writing it,
paperdigest checks it parses, rejects leftover prompt-echo lines, and scans it for
dangerous calls (`subprocess`, `socket`, `eval`/`exec`, `os.system`, `pickle.load`,
`shutil.rmtree`, and similar) plus networking/FFI imports (`urllib`, `requests`,
`ctypes`, and similar) — aborting loudly if it finds any. That scan catches obviously
hazardous patterns, not everything — read the scaffolded code before you run it.

Every project also gets a generated `AGENTS.md` — a coding-agent brief with a
dependency-ordered implementation plan, a TODO count and cited paper sections per
module, a definition of done, and ground rules for the harness. It's assembled
deterministically from the same stage data, no extra LLM call. Pass `--vault` (or set
`vault` in your config file) and it links back to the paper's Obsidian notes so an
agent can read the plain-English explanation before touching code.

## Cloud backends (optional, cost money)

| Backend | Flags | Key (env var only) |
|---|---|---|
| Local (default) | `--base-url http://localhost:8080/v1 --model local` | none |
| OpenAI | `--backend openai --model gpt-5` | `OPENAI_API_KEY` |
| Anthropic | `--backend anthropic --model claude-sonnet-5` | `ANTHROPIC_API_KEY` |

Every LLM call is retried up to 2 more times with exponential backoff on
transient errors (connection drops, timeouts, rate limits, 5xx); malformed
JSON gets one repair round-trip; other errors fail immediately. Requests
time out after 300s. On cloud backends, concept and module-stub generation
runs a few requests in parallel to cut wall-clock time; against a local
server it stays serial, since a single llama.cpp instance generally can't
service concurrent requests.

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
`~/.cache/paperdigest/` (configurable via `cache_dir` in `config.toml`); pass
`--refresh` to re-fetch and overwrite the cached HTML, e.g. after arXiv
publishes a revision.

## Development

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # fully offline: fixtures + fake LLM backend
```

Known minor cleanups (all reviewed, none load-bearing) are tracked in `TODOS.md`.
