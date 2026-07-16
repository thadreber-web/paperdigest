# Changelog

## Unreleased

### Added
- Figures: paper figures are downloaded and explained by a vision-capable
  model, with the image and explanation embedded in the matching concept
  note. `--figures/--no-figures` and the `max_figures` config key (default
  8) control it; backends without vision skip figures after one warning.
- `--version` flag; the scaffold failure debug dump now includes the version.
- `--quiet` flag on both commands to suppress progress output.
- `--max-input-chars`, `--max-tokens`, and `--cache-dir` CLI overrides for
  their config keys.
- Concept explanations and module stubs are generated with up to 4 parallel
  requests on cloud backends (local servers stay serial).
- CI and the release workflow enforce a 93% test-coverage floor.
- `max_tokens` config option: the LLM output-token cap (default 8192) is now
  set explicitly on both backends instead of hard-coded (Anthropic) or left
  to the server default (OpenAI-compatible).
- Scaffold stages now get the same one-shot JSON repair round-trip as digest:
  a malformed response is sent back to the model once for correction before
  the run aborts.
- Tag-triggered release workflow: pushing a `v*` tag lints, tests, builds,
  verifies the tag matches `pyproject.toml`, and attaches the wheel and sdist
  to a GitHub Release.

### Fixed
- `__version__` is now derived from package metadata instead of a hard-coded
  string that had drifted behind `pyproject.toml`.
- `~` in `vault`/`cache_dir` config values is now expanded.
- Concept-to-section matching is tiered (number-stripped exact, exact, then
  longest-overlap containment), so short or numeric section titles can no
  longer attach the wrong section's text to a concept.

## 0.3.0 — 2026-07-15

Robustness and safety hardening.

### Added
- Safety scan of all LLM-generated Python in `scaffold`: imports of
  `subprocess`/`socket`/`ctypes`/`urllib`/`requests` and friends, calls to
  `eval`/`exec`/`compile`/`__import__`, and dangerous attribute calls
  (`os.system`, `pickle.load`, `shutil.rmtree`, ...) abort the pipeline
  loudly with filename and line numbers.
- `--refresh` flag on `digest` and `scaffold` to re-fetch a paper, bypassing
  the HTML cache.
- CI now runs ruff lint, reports test coverage, and verifies the package
  builds (sdist + wheel).

### Changed
- LLM calls have a 300 s request timeout (previously unbounded).
- Responses truncated at the token limit now raise a clear error instead of
  being silently written as complete notes.
- Only transient errors (connection, timeout, rate limit, 5xx) are retried;
  permanent errors fail immediately with the original cause preserved.
- Module planning rejects duplicate filenames instead of silently
  overwriting one generated module with another.
- Scaffold debug dumps are written to the cache directory (keyed by paper id
  and stage) instead of the current working directory.

### Fixed
- HTML cache writes are atomic; an interrupted fetch can no longer leave a
  corrupt cache file that is served forever.
- A failure partway through rendering removes the half-written paper folder
  (existing glossary notes are never touched).
- Mermaid fence sanitization and validation now handle CRLF line endings and
  trailing whitespace instead of silently skipping those blocks.
- Concurrent runs against the same paper report a clean error instead of a
  traceback.

## 0.2.0 — 2026-07-12

- Mermaid diagrams by default with sanitizer and optional node-based
  validation; `--diagram` option.
- JSON-mode structured output with trailing-junk-tolerant parsing.
- Prompt-echo rejection in generated scaffold code.

## 0.1.0 — 2026-07-11

- Initial release: `digest` (arXiv paper to plain-English Obsidian notes)
  and `scaffold` (paper to git-initialized research project skeleton),
  local-first with optional Anthropic/OpenAI backends.
