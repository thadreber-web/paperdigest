# paperdigest — Design Spec

Date: 2026-07-10
Status: Approved (brainstormed and approved interactively)

## Purpose

A learning tool: hand it an arXiv link, get back a folder of plain-English,
wikilinked markdown notes in an Obsidian vault — concepts chunked into
digestible notes, jargon stripped and defined in a shared glossary, ASCII
diagrams for structural ideas, key equations explained line-by-line.
Single user, but built to be widely usable.

## Decisions

| Decision | Choice |
|---|---|
| Approach | Thin hand-rolled pipeline (~600–800 lines), no LLM framework |
| LLM backends | Pluggable: OpenAI-compatible client (covers OpenAI **and** local Ollama/vLLM) + small Anthropic adapter for Claude |
| Interface | CLI only (typer), containerized Python |
| Inputs (v1) | arXiv links / bare IDs only — fetch arXiv HTML (ar5iv fallback); clear error if no HTML rendering exists |
| Output | One folder of wikilinked notes per paper, written into a volume-mounted Obsidian vault |
| Glossary | **Global**: one note per term in `Glossary/<term>.md`, shared across papers; new papers only add missing terms, never touch existing ones |
| Depth | Configurable `--level` (beginner / intermediate / advanced) |
| Re-runs | Refuse if the paper folder exists; `--force` overwrites (protects hand annotations) |
| Math/figures | Key equations kept as LaTeX with plain-English walkthrough; no figure images in v1; ASCII diagrams for structure |
| Extras | Section citations ("from §3.2") in every concept note; 3–5 self-test questions in the overview |
| Long papers | Full text to cloud models; for local backends, warn and trim section bodies proportionally to fit context |
| Caching | Fetched/extracted papers cached to a work dir; no mid-digest resume in v1 |

## Architecture

```
paperdigest/
├── cli.py       # typer entry: paperdigest <url|id> [--level] [--backend] [--model] [--force]
├── config.py    # TOML + env vars (API keys env-only) + CLI-flag overrides
├── fetch.py     # arXiv URL/ID → HTML (arxiv.org/html/<id>, ar5iv fallback); disk cache
├── extract.py   # HTML → Paper dataclass (title, abstract, sections w/ LaTeX preserved)
├── llm.py       # Backend protocol; OpenAICompatibleBackend, AnthropicBackend
├── digest.py    # LLM orchestration → Digest dataclass
└── render.py    # Digest → markdown note folder + global glossary notes
```

Data flows one direction through plain dataclasses: `fetch → extract → digest → render`.

## Digest strategy (LLM calls per paper)

1. **Outline call** — TLDR, why-it-matters, 4–8 core concepts (with source
   sections), full jargon term list. Structured JSON.
2. **Concept calls** (one per concept) — plain-English explanation at
   `--level`, analogy where apt, ASCII diagram when structural, key equations
   as LaTeX + line-by-line walkthrough, section citation.
3. **Glossary call** — each jargon term → 1–2 sentence English definition
   (skipping terms already in the vault's global glossary).

Malformed JSON → one repair retry. API errors → 2 retries with backoff, then
fail loudly with the raw error. Progress printed per call.

## Output artifact

```
<vault>/Papers/2024-attention-is-all-you-need/
├── 00 Overview.md        # TLDR, why it matters, [[links]], self-test questions
├── 01 Self-Attention.md  # explanation, ASCII diagram, equations, "from §3.2"
└── ...
<vault>/Glossary/
├── attention.md          # global, cross-paper term notes
└── ...
```

YAML frontmatter on every note: source URL, date, model, level,
`#paper-digest` tag.

## Container & backends

- `docker run -v ~/Vault:/vault -e ANTHROPIC_API_KEY paperdigest <arxiv-url>`
- Local models: container calls the host's Ollama/vLLM OpenAI-compatible
  endpoint via `--base-url` — no GPU inside the container.
- Also works as a plain `pip install -e .` CLI during development.

## Testing

- Unit tests: `extract` and `render` against saved HTML/JSON fixtures;
  `digest` against a fake LLM backend. No network or API keys in tests.
- Manual smoke: real arXiv paper end-to-end against a real backend.
