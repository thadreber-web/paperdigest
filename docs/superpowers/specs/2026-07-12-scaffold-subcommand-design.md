# Design: `paperdigest scaffold` — paper-to-project generator

Date: 2026-07-12
Status: approved (design discussion 2026-07-12)

## Purpose

Given an arXiv paper, generate a standalone, git-initialized research project
that a human will use to reproduce/explore the paper: Cookiecutter-DS-style
folder structure, documentation, experiment logging, and skeleton code with
stubs mapped to paper sections. Complements the existing `digest` pipeline
(Obsidian notes); reuses its fetch/extract/LLM layers.

## Decisions (from design discussion)

- **Code depth:** skeleton + stubs. Module files with signatures, docstrings
  citing paper sections/equations, and `TODO(paper §x.y)` markers; a working
  train/eval harness; one toy smoke-test experiment. No best-effort full
  reimplementation.
- **Convention:** Cookiecutter-Data-Science-style layout with zero-dependency
  plain logging (JSONL runs + Python logging). No MLflow/W&B/trackio.
- **Integration:** subcommand in this repo (`paperdigest scaffold`), developed
  on a feature branch and merged. Existing `digest` behavior unchanged and
  remains the default when no subcommand is given.
- **Output:** standalone directory per paper, `git init` + initial commit.
- **Model routing:** single model per run via the existing
  `--backend/--base-url/--model` flags and `config.toml`. No auto-escalation,
  no retries. The user picks the model per run (typical local options: 9B on
  `:8001`, 35B-A3B MoE on `:8000`, 27B dense). Quality bar over convenience.
- **Failure mode:** loud. Any stage failure aborts the entire run.

## CLI

```
paperdigest scaffold <arxiv-id-or-url> --dest <dir> [--force]
                     [--backend ...] [--base-url ...] [--model ...]
```

- Output path: `<dest>/<year>-<title-slug>/` (same slug logic as digest).
- If the output directory already exists: refuse before any LLM call
  (mirrors existing digest behavior), unless `--force` (which removes and
  regenerates it).
- `--dest` is required (no default projects directory in v1).
- CLI grows subcommand dispatch: `paperdigest <paper>` == `paperdigest digest
  <paper>` for backward compatibility.

## Architecture

New module `paperdigest/scaffold.py` orchestrates; new
`paperdigest/templates.py` holds static file templates as string constants
(no data-dir packaging concerns). Reused untouched: `fetch.py`, `extract.py`, `llm.py`, `config.py`.
`cli.py` gains dispatch only.

### Generated project layout (static skeleton)

```
<year>-<slug>/
  README.md               # LLM: paper summary, repo map, how to run
  EXPERIMENTS.md          # LLM: planned experiments mirroring paper tables,
                          #      status checkboxes
  pyproject.toml          # static template (project name filled in)
  .gitignore              # static
  configs/
    base.yaml             # LLM: hyperparameters extracted from the paper
    smoke.yaml            # LLM: tiny overrides for the smoke experiment
  src/<pkg>/
    __init__.py           # static
    tracking.py           # static: JSONL experiment logger
    <module>.py ...       # LLM: one file per planned module (stubs)
  train.py                # LLM: harness wired to configs + tracking
  evaluate.py             # LLM: harness
  experiments/
    exp001_smoke/README.md  # LLM: what the smoke run does, how to run it
    runs.jsonl              # created empty; appended by tracking.log_run
  data/{raw,processed}/.gitkeep   # static
  notebooks/.gitkeep      # static
  reports/figures/.gitkeep# static
  logs/.gitkeep           # static
  tests/test_smoke.py     # LLM: imports modules, instantiates, shape checks
```

`tracking.py` (static, no dependencies): `log_run(params: dict, metrics:
dict, **meta)` appends one JSON line to `experiments/runs.jsonl` and sets up
a per-run log file under `logs/`.

### LLM stages

Each stage is a separate chat call returning schema-validated JSON, in order:

1. **Analyze** — from extracted paper text: method components, datasets,
   key hyperparameters, list of experiments/tables in the paper.
2. **Plan** — map components to a module list: for each module a filename,
   responsibility, public functions/classes with signatures, dependencies on
   other modules.
3. **Stub generation** — one call **per module** (keeps outputs small for a
   9B): full file content with signatures, docstrings citing paper
   sections/equations, `# TODO(paper §x.y):` bodies (`raise
   NotImplementedError` where appropriate). Also `tests/test_smoke.py`.
4. **Harness + smoke experiment** — `train.py`, `evaluate.py`,
   `configs/base.yaml`, `configs/smoke.yaml`, `experiments/exp001_smoke/
   README.md`. The smoke config must be runnable end-to-end on toy/random
   data once stubs are filled in; harness code itself must be complete (not
   a stub) and use `tracking.log_run`.
5. **Docs** — `README.md`, `EXPERIMENTS.md`.

Stage outputs are validated: JSON parses, matches expected schema, generated
Python files parse with `ast.parse`. Prompts follow the existing digest
prompt style in `digest.py`.

### Loud failure

On any stage failure (bad JSON, schema mismatch, unparseable Python):

- Print a multi-line `ERROR` banner: stage name, model, base URL, and the
  path of a debug file containing the raw model output (written to the
  current directory as `scaffold-debug-<stage>.txt`).
- Delete the partially created project directory — never leave a
  half-project.
- Exit nonzero.
- No retries, no model fallback.

### Finalization

After all stages succeed: write files, `git init`, `git add -A`, initial
commit ("Scaffold generated from arXiv:<id> by paperdigest"). If `git` is
unavailable, warn loudly but keep the generated project (this is the one
non-fatal failure).

## Testing

- Unit tests with a fake LLM (existing `tests/` pattern): happy path
  produces the full tree and valid Python; existing-dir refusal happens
  before any LLM call; stage failure deletes the output dir and exits
  nonzero with the debug file written; git init produces a commit;
  `tracking.py` logger round-trips a run record.
- Live smoke test against the local 9B (`:8001`) on a small paper, results
  recorded in `docs/` like the existing smoke-test note.

## Out of scope (v1)

- Auto model routing / escalation between local models.
- Full method reimplementation.
- MLflow/W&B integration.
- Figure/image handling in the generated project (digest already extracts
  figures into the vault).
- Config knobs beyond what digest already exposes.
