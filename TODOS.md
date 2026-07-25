# TODOS

Minor cleanups deferred from review (2026-07-17, AGENTS.md + cross-links effort).
All were triaged "safe to ride" by the final whole-branch review — none carry
correctness or safety risk. Pick up opportunistically.

## Known bugs (deferred, real correctness impact)

- [ ] **Generated files sometimes import sibling/infra modules by bare name** —
  observed on a live 9B scaffold run (2026-07-24, arXiv:1706.03762). The model wrote
  `tests/test_smoke.py` with `from attention import ...`, `from model import ...`,
  `from tracking import ...` instead of `from attention_is_all_you_need import attention`
  etc. Bare names don't resolve once the project is pip-installed as its package, so the
  generated smoke test fails to import (and cross-module stub imports can fail the same
  way). It is **intermittent** — other runs emit the correct `from <pkg> import <module>`
  form and pass. This does *not* affect the scaffold's core purpose (stubs, section
  citations, hyperparameters, and AGENTS.md brief are all correct and agent-ready); an
  implementing agent fixes the import in seconds. It is a paperdigest robustness gap:
  the model improvises import style the same way it improvised config separators and
  mermaid labels.
  **Planned fix:** deterministically rewrite bare sibling/infra imports in generated
  files to the full package path (`from <pkg>.<module> import ...`), same approach as the
  mermaid label sanitizer and the tolerant config loader. The dependency scanner already
  ignores these names (fixed 2026-07-24), so pyproject stays correct; only the emitted
  import statements still need rewriting. Until then, the generated project is a correct
  spec but its own smoke test may not import as-shipped.

## Open

- [ ] **`config.py`: `project_dir` field line is exactly 120 chars** — zero margin
  against ruff's line-length limit; the next one-word edit to that line trips lint.
  Shorten the trailing comment when touching the file.
- [ ] **`config_file_has` re-parses TOML with no error handling** — a malformed
  config file raises `tomllib.TOMLDecodeError` uncaught. Matches `load_config`'s
  existing behavior (consistent, not a regression); if either grows a friendly
  parse-error message, give it to both.
- [ ] **`analysis_ctx` string-building duplicated** — the same 2-line f-string is
  built in `_build_stub_files` and again in `build_scaffold` (scaffold.py). Extract
  a small `_analysis_ctx(analysis, modules)` helper.
- [ ] **AGENTS.md "depends on:" prints unknown dependency names verbatim** — the
  Kahn ordering correctly ignores dependencies that don't match a planned module,
  but the display line shows them unfiltered, so a hallucinated dep name appears in
  the text while not affecting order. Filter the display list the same way.
- [ ] **No test for `_module_order`'s cycle-fallback branch** (falls back to plan
  order) **or the missing-file `.get(..., "")` fallback** in `build_agents_md`.
  Both branches were manually verified correct in review; add tests opportunistically.
- [ ] **No dedicated `PermissionError` test for `_project_section_index`** —
  structurally covered by the `except (OSError, UnicodeDecodeError)` catch (it's an
  `OSError` subclass) and the broken-symlink test exercises the same clause. Low value;
  add only if touching those tests anyway.
