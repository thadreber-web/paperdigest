# `paperdigest scaffold` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `paperdigest scaffold <arxiv-ref> --dest <dir>` subcommand that turns a paper into a standalone, git-initialized research project: Cookiecutter-DS-style layout, JSONL experiment tracking, and LLM-generated skeleton code (stubs mapped to paper sections), with loud abort-on-failure.

**Architecture:** New `paperdigest/scaffold.py` orchestrates five schema-validated LLM stages (analyze → plan → per-module stubs + smoke test → harness → docs), all in memory; only after every stage succeeds does `write_project` create the output directory (static templates from new `paperdigest/templates.py` + the LLM files) and git-init it. `cli.py` becomes a two-command Typer group (`digest`, `scaffold`) with a click-level fallback so bare `paperdigest <ref>` still means digest. Reuses `fetch.py`, `extract.py`, `llm.py`, `config.py` unchanged (except moving `_strip_fences` from `digest.py` into `llm.py` as public `strip_fences`).

**Tech Stack:** Python ≥3.11, typer/click, pytest with the existing `FakeBackend` from `tests/conftest.py`. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-07-12-scaffold-subcommand-design.md`

## Global Constraints

- Python `>=3.11`; no new entries in `pyproject.toml` `dependencies`.
- Single model per run via existing `--backend/--model/--base-url` + `config.toml`; **no JSON-repair retries, no model fallback** in scaffold stages (unlike digest).
- Loud failure: any stage failure → multi-line `ERROR` banner on stderr (stage, model, server, cause, debug-file path), raw model output saved to `scaffold-debug-<stage>.txt` in the cwd, exit code 1, **no output directory left behind**.
- Existing-output refusal happens **before any LLM call**; `--force` overrides.
- Existing digest behavior and all existing tests must keep passing unchanged (test files may not be edited except where a task says so).
- Match existing style: `from __future__ import annotations`, dataclasses, module-level exception classes, `_progress` printing to stderr.
- Run tests with: `.venv/bin/python -m pytest` from `~/code/paperdigest` (create the venv with `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'` if missing).
- Commit after every task; do not push.

---

### Task 1: Move `_strip_fences` into `llm.py` as public `strip_fences`

Scaffold needs fence-stripping but not digest's JSON-repair flow; make the helper shared instead of duplicating it.

**Files:**
- Modify: `paperdigest/llm.py` (add `strip_fences` at module level, after `LOCAL_BASE_URL`)
- Modify: `paperdigest/digest.py` (delete `_strip_fences` [lines 80-85] and its `re` usage; import and use `strip_fences`)
- Test: `tests/test_llm.py` (append tests)

**Interfaces:**
- Produces: `paperdigest.llm.strip_fences(text: str) -> str` — strips a leading ```` ```lang ```` fence line and trailing ```` ``` ````, plus surrounding whitespace. Later tasks import this.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_llm.py`:

```python
from paperdigest.llm import strip_fences


def test_strip_fences_removes_markdown_fences():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_passes_plain_text_through():
    assert strip_fences('  {"a": 1}\n') == '{"a": 1}'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ImportError: cannot import name 'strip_fences'`

- [ ] **Step 3: Implement** — in `paperdigest/llm.py`, add `import re` to the imports and this function after the `LOCAL_BASE_URL` line:

```python
def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text
```

In `paperdigest/digest.py`: change the llm import line to
`from .llm import Backend, LLMError, complete_with_retry, strip_fences`,
delete the whole `_strip_fences` function, replace both `_strip_fences(` call
sites inside `_call_json` with `strip_fences(`, and remove `import re` **only
if** `re` is now unused in digest.py (check with a search — it is only used by
`_strip_fences`, so remove it).

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests PASS (including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add paperdigest/llm.py paperdigest/digest.py tests/test_llm.py
git commit -m "refactor: move fence-stripping into llm.strip_fences"
```

---

### Task 2: Static templates module

**Files:**
- Create: `paperdigest/templates.py`
- Test: `tests/test_templates.py`

**Interfaces:**
- Produces (all consumed by Task 5's `write_project`):
  - `templates.STATIC_DIRS: tuple[str, ...]` — relative dirs that get a `.gitkeep`
  - `templates.GITIGNORE: str`
  - `templates.PYPROJECT: str` — `.format(name=<package>)` fills the only placeholder
  - `templates.TRACKING_PY: str` — full content of the generated `src/<pkg>/tracking.py`; its API is `log_run(params: dict, metrics: dict, **meta) -> str` (returns run id, appends to `experiments/runs.jsonl` two levels above the package dir) and `get_logger(name: str, run_id: str = "adhoc") -> logging.Logger` (stderr + `logs/<run_id>.log`)

- [ ] **Step 1: Write the failing tests** — create `tests/test_templates.py`:

```python
import ast
import importlib.util
import json

from paperdigest import templates


def test_tracking_template_is_valid_python():
    ast.parse(templates.TRACKING_PY)


def test_pyproject_template_formats_with_package_name():
    text = templates.PYPROJECT.format(name="my_paper")
    assert 'name = "my_paper"' in text
    assert 'packages = ["src/my_paper"]' in text


def test_tracking_log_run_round_trips(tmp_path):
    pkg = tmp_path / "src" / "demo"
    pkg.mkdir(parents=True)
    tracking = pkg / "tracking.py"
    tracking.write_text(templates.TRACKING_PY)
    spec = importlib.util.spec_from_file_location("tracking", tracking)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_id = mod.log_run({"lr": 0.001}, {"loss": 1.23}, note="smoke")
    lines = (tmp_path / "experiments" / "runs.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    assert record["run_id"] == run_id
    assert record["params"] == {"lr": 0.001}
    assert record["metrics"] == {"loss": 1.23}
    assert record["note"] == "smoke"


def test_static_dirs_cover_cookiecutter_layout():
    assert set(templates.STATIC_DIRS) == {
        "data/raw", "data/processed", "notebooks", "reports/figures", "logs",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_templates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.templates'`

- [ ] **Step 3: Implement** — create `paperdigest/templates.py` exactly:

````python
"""Static file templates for generated projects — deterministic, no LLM."""

STATIC_DIRS = ("data/raw", "data/processed", "notebooks", "reports/figures", "logs")

GITIGNORE = """\
__pycache__/
*.pyc
*.egg-info/
.venv/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
logs/*
!logs/.gitkeep
"""

PYPROJECT = """\
[project]
name = "{name}"
version = "0.1.0"
description = "Reproduction scaffold generated by paperdigest"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

TRACKING_PY = '''\
"""Zero-dependency experiment tracking: JSONL run records + per-run log files."""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_FILE = ROOT / "experiments" / "runs.jsonl"
LOG_DIR = ROOT / "logs"


def log_run(params: dict, metrics: dict, **meta) -> str:
    """Append one run record to experiments/runs.jsonl and return its run id."""
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    record = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "metrics": metrics,
        **meta,
    }
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\\n")
    return run_id


def get_logger(name: str, run_id: str = "adhoc") -> logging.Logger:
    """Logger writing to stderr and logs/<run_id>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler(LOG_DIR / f"{run_id}.log")):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger
'''
````

(Note: `"\\n"` inside `TRACKING_PY` is intentional — the *generated* file must
contain `"\n"`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_templates.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/templates.py tests/test_templates.py
git commit -m "feat: static project templates (layout dirs, pyproject, JSONL tracker)"
```

---

### Task 3: Scaffold stages — analyze, plan, module stubs, smoke test

**Files:**
- Create: `paperdigest/scaffold.py`
- Test: `tests/test_scaffold_stages.py`

**Interfaces:**
- Consumes: `llm.strip_fences`, `llm.complete_with_retry`, `llm.Backend`, `llm.LLMError` (Task 1); `digest._paper_body(paper, max_chars, progress) -> str`; `render.slugify(text) -> str`; `extract.Paper`.
- Produces (consumed by Tasks 4-6):
  - `scaffold.ScaffoldError(stage: str, message: str, raw: str = "")` — exception with `.stage` and `.raw` attributes
  - `scaffold.package_name(title: str) -> str`
  - `scaffold.project_folder(arxiv_id: str, title: str, dest: Path) -> Path`
  - `scaffold._stage_json(backend, stage, system, user, required) -> dict` and `scaffold._stage_python(backend, stage, system, user) -> str` (internal, reused by Task 4)
  - `scaffold.ScaffoldProject` dataclass: `arxiv_id: str, title: str, url: str, package: str, model: str, files: dict[str, str]`
  - partial `build_scaffold` (completed in Task 4)

- [ ] **Step 1: Write the failing tests** — create `tests/test_scaffold_stages.py`:

```python
import json

import pytest

from conftest import FakeBackend
from paperdigest import scaffold
from paperdigest.extract import Paper, Section


def make_paper():
    return Paper(
        arxiv_id="1706.03762",
        title="Tiny Transformers Explained",
        abstract="We explain tiny transformers.",
        sections=[Section(title="3 Method", text="The encoder uses attention.")],
        url="https://arxiv.org/abs/1706.03762",
    )


ANALYZE = json.dumps(
    {
        "components": [{"name": "Encoder", "description": "Stack of attention blocks.", "section": "3 Method"}],
        "datasets": ["toy-corpus"],
        "hyperparameters": {"lr": "0.001"},
        "experiments": [{"name": "main", "description": "Perplexity, Table 1."}],
    }
)
PLAN = json.dumps(
    {
        "modules": [
            {
                "filename": "model.py",
                "responsibility": "Define the encoder.",
                "api": ["def build_model(cfg: dict)"],
                "dependencies": [],
            }
        ]
    }
)
MODULE_PY = (
    'def build_model(cfg: dict):\n'
    '    """Build the encoder. See paper §3."""\n'
    '    # TODO(paper §3): implement the encoder\n'
    '    raise NotImplementedError\n'
)
SMOKE_TEST = "from tiny_transformers_explained import model\n\n\ndef test_api_exists():\n    assert callable(model.build_model)\n"


def test_package_name_is_valid_identifier():
    assert scaffold.package_name("Tiny Transformers Explained") == "tiny_transformers_explained"
    assert scaffold.package_name("3D Gaussian Splatting").isidentifier()


def test_project_folder_uses_year_and_slug(tmp_path):
    folder = scaffold.project_folder("1706.03762", "Tiny Transformers Explained", tmp_path)
    assert folder == tmp_path / "2017-tiny-transformers-explained"


def test_stage_json_rejects_bad_json():
    backend = FakeBackend(["this is not json"])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold._stage_json(backend, "analyze", "sys", "user", required=("components",))
    assert exc.value.stage == "analyze"
    assert exc.value.raw == "this is not json"


def test_stage_json_rejects_missing_fields():
    backend = FakeBackend([json.dumps({"components": []})])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold._stage_json(backend, "analyze", "sys", "user", required=("components", "datasets"))
    assert "datasets" in str(exc.value)


def test_stage_python_rejects_unparseable_code():
    backend = FakeBackend(["def broken(:\n    pass"])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold._stage_python(backend, "module:model.py", "sys", "user")
    assert exc.value.stage == "module:model.py"


def test_stage_python_strips_fences_and_parses():
    backend = FakeBackend(["```python\n" + MODULE_PY + "```"])
    code = scaffold._stage_python(backend, "module:model.py", "sys", "user")
    assert code.startswith("def build_model")


def test_plan_stage_rejects_empty_module_list():
    backend = FakeBackend([ANALYZE, json.dumps({"modules": []})])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold.build_scaffold(make_paper(), backend, max_chars=100_000, progress=lambda m: None)
    assert exc.value.stage == "plan"


def test_module_filenames_are_stripped_to_basenames():
    evil_plan = json.dumps(
        {
            "modules": [
                {
                    "filename": "../../etc/model.py",
                    "responsibility": "Define the encoder.",
                    "api": ["def build_model(cfg: dict)"],
                    "dependencies": [],
                }
            ]
        }
    )
    backend = FakeBackend([ANALYZE, evil_plan, MODULE_PY, SMOKE_TEST])
    stubs = scaffold._build_stub_files(make_paper(), backend, "tiny_transformers_explained", max_chars=100_000, progress=lambda m: None)
    assert "src/tiny_transformers_explained/model.py" in stubs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scaffold_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperdigest.scaffold'`

- [ ] **Step 3: Implement** — create `paperdigest/scaffold.py`. Full content
(Task 4 will extend `build_scaffold`; write it now exactly as shown, with
stubs/smoke wiring factored into `_build_stub_files` so it is testable alone):

```python
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .digest import _paper_body
from .extract import Paper
from .llm import Backend, LLMError, complete_with_retry, strip_fences
from .render import slugify


class ScaffoldError(Exception):
    """A scaffold stage produced unusable output. Aborts the whole run."""

    def __init__(self, stage: str, message: str, raw: str = ""):
        super().__init__(message)
        self.stage = stage
        self.raw = raw


@dataclass
class ScaffoldProject:
    arxiv_id: str
    title: str
    url: str
    package: str
    model: str
    files: dict[str, str] = field(default_factory=dict)  # relative path -> content


_ANALYZE_SYSTEM = """\
You analyze an AI/ML research paper so its method can be reimplemented.
Respond with ONLY valid JSON (no markdown fences, no commentary) in exactly this shape:
{"components": [{"name": "<method component>", "description": "<1-2 sentences>", "section": "<paper section heading>"}],
 "datasets": ["<dataset name>", ...],
 "hyperparameters": {"<name>": "<value as stated in the paper>"},
 "experiments": [{"name": "<short name>", "description": "<what it measures and which table/figure it reproduces>"}]}
Rules: pick 3-8 method components. List every dataset the paper uses.
Include only hyperparameters the paper actually states."""

_PLAN_SYSTEM = """\
You plan the module layout of a Python project that will reimplement a research paper's method.
Respond with ONLY valid JSON (no markdown fences, no commentary):
{"modules": [{"filename": "<snake_case>.py",
              "responsibility": "<one sentence>",
              "api": ["<python signature, e.g. def build_model(cfg: dict) -> object>"],
              "dependencies": ["<other module filename>", ...]}]}
Rules: 2-6 modules, one clear responsibility each. Plain filenames only, no directories.
Do not include tracking.py, train.py, evaluate.py, or test files — those are handled separately."""

_MODULE_SYSTEM = """\
You write ONE Python skeleton file for a project reimplementing a research paper.
Output ONLY the file content (no markdown fences, no commentary).
Rules:
- Implement the given public API as signatures with complete docstrings.
- Docstrings cite the paper section or equation each item comes from (e.g. "See paper §3.2, Eq. 4").
- Function/method bodies are stubs: a `# TODO(paper §x.y): <what to implement>` comment, then `raise NotImplementedError`.
- Simple glue (dataclasses, config parsing, obvious helpers) may be fully implemented.
- The file must parse as valid Python. Standard-library imports only unless the API requires otherwise."""

_SMOKE_TEST_SYSTEM = """\
You write tests/test_smoke.py for a skeleton research-code project.
Output ONLY the file content (no markdown fences, no commentary).
Rules:
- Import each project module and assert its public API exists (functions/classes present, callable).
- Do NOT call stub functions — they raise NotImplementedError. Only check existence.
- pytest style, no fixtures. The file must parse as valid Python."""

_HARNESS_SYSTEM = """\
You write the experiment harness for a skeleton research-code project.
Respond with ONLY valid JSON (no markdown fences, no commentary) with exactly these string fields:
{"train_py": "<content of train.py>",
 "evaluate_py": "<content of evaluate.py>",
 "base_yaml": "<content of configs/base.yaml>",
 "smoke_yaml": "<content of configs/smoke.yaml>",
 "smoke_readme": "<content of experiments/exp001_smoke/README.md>"}
Rules:
- train.py and evaluate.py are COMPLETE runnable code (not stubs): argparse with a --config option
  taking a YAML path, calls into the project modules' public API, and logs results via
  `from <package>.tracking import log_run`. Parse the config with a minimal `key: value` line parser
  (standard library only — do not import yaml).
- Both files must parse as valid Python.
- base_yaml holds the paper's stated hyperparameters; smoke_yaml overrides them for a seconds-long toy run.
- smoke_readme explains what the smoke run checks and gives the exact command to run it."""

_DOCS_SYSTEM = """\
You write documentation for a skeleton research-code project.
Respond with ONLY valid JSON (no markdown fences, no commentary) with exactly these string fields:
{"readme": "<content of README.md>", "experiments_md": "<content of EXPERIMENTS.md>"}
Rules:
- readme: paper title + arXiv link, 3-5 sentence method summary, a repo map explaining every
  top-level file/folder, setup instructions (pip install -e '.[dev]'), and how to run the smoke experiment.
- experiments_md: one section per planned experiment mirroring the paper's tables/figures, each with
  a `- [ ]` status checkbox, which config to use, and the paper's reported numbers to compare against."""


def _default_progress(msg: str) -> None:
    print(msg, file=sys.stderr)


def package_name(title: str) -> str:
    name = slugify(title).replace("-", "_")
    if not name.isidentifier():
        name = "p_" + name
    return name


def project_folder(arxiv_id: str, title: str, dest: Path) -> Path:
    year = "20" + arxiv_id[:2]
    return dest / f"{year}-{slugify(title)}"


def _stage_json(backend: Backend, stage: str, system: str, user: str, required: tuple[str, ...]) -> dict:
    try:
        raw = complete_with_retry(backend, system, user)
    except LLMError as e:
        raise ScaffoldError(stage, str(e)) from e
    try:
        data = json.loads(strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ScaffoldError(stage, f"model returned invalid JSON: {e}", raw=raw) from e
    if not isinstance(data, dict):
        raise ScaffoldError(stage, f"expected a JSON object, got {type(data).__name__}", raw=raw)
    missing = [k for k in required if k not in data]
    if missing:
        raise ScaffoldError(stage, f"response is missing fields: {missing}", raw=raw)
    return data


def _stage_python(backend: Backend, stage: str, system: str, user: str) -> str:
    try:
        raw = complete_with_retry(backend, system, user)
    except LLMError as e:
        raise ScaffoldError(stage, str(e)) from e
    code = strip_fences(raw)
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ScaffoldError(stage, f"generated Python does not parse: {e}", raw=raw) from e
    return code


def _paper_ctx(paper: Paper, body: str) -> str:
    return f"PAPER TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}\n\nFULL TEXT:\n{body}"


def _analyze(paper: Paper, backend: Backend, body: str, progress: Callable[[str], None]) -> dict:
    progress("Stage 1/5: analyzing the paper...")
    return _stage_json(
        backend, "analyze", _ANALYZE_SYSTEM, _paper_ctx(paper, body),
        required=("components", "datasets", "hyperparameters", "experiments"),
    )


def _plan_modules(paper: Paper, backend: Backend, body: str, analysis: dict,
                  progress: Callable[[str], None]) -> list[dict]:
    progress("Stage 2/5: planning modules...")
    plan = _stage_json(
        backend, "plan", _PLAN_SYSTEM,
        f"{_paper_ctx(paper, body)}\n\nPAPER ANALYSIS:\n{json.dumps(analysis, indent=2)}",
        required=("modules",),
    )
    modules = plan["modules"]
    if not isinstance(modules, list) or not modules:
        raise ScaffoldError("plan", "'modules' must be a non-empty list", raw=json.dumps(plan))
    for m in modules:
        if not isinstance(m, dict) or not str(m.get("filename", "")).endswith(".py"):
            raise ScaffoldError("plan", f"bad module entry: {m!r}", raw=json.dumps(plan))
        m["filename"] = Path(str(m["filename"])).name  # never allow directory components
    return modules


def _build_stub_files(paper: Paper, backend: Backend, pkg: str, max_chars: int,
                      progress: Callable[[str], None]) -> dict[str, str]:
    """Stages 1-3: analyze, plan, module stubs + smoke test. Returns files + stashes context."""
    body = _paper_body(paper, max_chars, progress)
    analysis = _analyze(paper, backend, body, progress)
    modules = _plan_modules(paper, backend, body, analysis, progress)

    analysis_ctx = (
        f"PAPER ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
        f"MODULE PLAN:\n{json.dumps(modules, indent=2)}"
    )
    files: dict[str, str] = {}
    total = len(modules)
    for i, m in enumerate(modules, 1):
        progress(f"Stage 3/5: writing stub {i}/{total}: {m['filename']}")
        code = _stage_python(
            backend, f"module:{m['filename']}", _MODULE_SYSTEM,
            f"FILE TO WRITE: src/{pkg}/{m['filename']}\n"
            f"RESPONSIBILITY: {m.get('responsibility', '')}\n"
            "PUBLIC API:\n" + "\n".join(f"- {sig}" for sig in m.get("api", []))
            + f"\n\n{analysis_ctx}\n\nPAPER TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}",
        )
        files[f"src/{pkg}/{m['filename']}"] = code

    progress("Stage 3/5: writing tests/test_smoke.py")
    files["tests/test_smoke.py"] = _stage_python(
        backend, "smoke-test", _SMOKE_TEST_SYSTEM,
        f"PACKAGE: {pkg} (import as `from {pkg} import <module>`)\n\n{analysis_ctx}",
    )
    files["__analysis_ctx__"] = analysis_ctx  # consumed and removed by build_scaffold
    return files


def build_scaffold(paper: Paper, backend: Backend, max_chars: int,
                   progress: Callable[[str], None] = _default_progress) -> ScaffoldProject:
    pkg = package_name(paper.title)
    files = _build_stub_files(paper, backend, pkg, max_chars, progress)
    analysis_ctx = files.pop("__analysis_ctx__")

    progress("Stage 4/5: writing train/evaluate harness and configs...")
    harness = _stage_json(
        backend, "harness", _HARNESS_SYSTEM,
        f"PACKAGE: {pkg}\n\n{analysis_ctx}",
        required=("train_py", "evaluate_py", "base_yaml", "smoke_yaml", "smoke_readme"),
    )
    for key in ("train_py", "evaluate_py"):
        try:
            ast.parse(str(harness[key]))
        except SyntaxError as e:
            raise ScaffoldError("harness", f"{key} does not parse as Python: {e}", raw=str(harness[key])) from e
    files["train.py"] = str(harness["train_py"])
    files["evaluate.py"] = str(harness["evaluate_py"])
    files["configs/base.yaml"] = str(harness["base_yaml"])
    files["configs/smoke.yaml"] = str(harness["smoke_yaml"])
    files["experiments/exp001_smoke/README.md"] = str(harness["smoke_readme"])

    progress("Stage 5/5: writing README and EXPERIMENTS...")
    docs = _stage_json(
        backend, "docs", _DOCS_SYSTEM,
        f"ARXIV: {paper.url}\nPACKAGE: {pkg}\n\n{analysis_ctx}",
        required=("readme", "experiments_md"),
    )
    files["README.md"] = str(docs["readme"])
    files["EXPERIMENTS.md"] = str(docs["experiments_md"])

    return ScaffoldProject(
        arxiv_id=paper.arxiv_id, title=paper.title, url=paper.url,
        package=pkg, model=backend.model, files=files,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scaffold_stages.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/scaffold.py tests/test_scaffold_stages.py
git commit -m "feat: scaffold LLM stages — analyze, plan, module stubs, smoke test"
```

---

### Task 4: Full `build_scaffold` pipeline test (harness + docs stages)

The harness/docs code already exists from Task 3's file; this task locks it in
with an end-to-end fake-backend test of the whole in-memory pipeline.

**Files:**
- Test: `tests/test_scaffold_stages.py` (append)

**Interfaces:**
- Consumes: everything Task 3 produced.
- Produces: canonical fake response constants (`HARNESS`, `DOCS`) reused verbatim by Task 6's CLI tests.

- [ ] **Step 1: Write the tests** — append to `tests/test_scaffold_stages.py`:

```python
TRAIN_PY = (
    "import argparse\n"
    "from tiny_transformers_explained.tracking import log_run\n\n\n"
    "def main():\n"
    "    parser = argparse.ArgumentParser()\n"
    "    parser.add_argument('--config', required=True)\n"
    "    args = parser.parse_args()\n"
    "    log_run({'config': args.config}, {'loss': 0.0})\n\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)
HARNESS = json.dumps(
    {
        "train_py": TRAIN_PY,
        "evaluate_py": TRAIN_PY,
        "base_yaml": "lr: 0.001",
        "smoke_yaml": "epochs: 1",
        "smoke_readme": "# Smoke run\nRun: python train.py --config configs/smoke.yaml",
    }
)
DOCS = json.dumps({"readme": "# Tiny Transformers", "experiments_md": "# Experiments\n- [ ] main"})


def full_responses():
    return [ANALYZE, PLAN, MODULE_PY, SMOKE_TEST, HARNESS, DOCS]


def test_build_scaffold_produces_all_files():
    backend = FakeBackend(full_responses())
    project = scaffold.build_scaffold(make_paper(), backend, max_chars=100_000, progress=lambda m: None)
    assert project.package == "tiny_transformers_explained"
    assert project.model == "fake-model"
    assert set(project.files) == {
        "src/tiny_transformers_explained/model.py",
        "tests/test_smoke.py",
        "train.py",
        "evaluate.py",
        "configs/base.yaml",
        "configs/smoke.yaml",
        "experiments/exp001_smoke/README.md",
        "README.md",
        "EXPERIMENTS.md",
    }
    assert "TODO(paper" in project.files["src/tiny_transformers_explained/model.py"]


def test_build_scaffold_rejects_unparseable_harness():
    bad_harness = json.dumps(
        {
            "train_py": "def broken(:",
            "evaluate_py": TRAIN_PY,
            "base_yaml": "lr: 0.001",
            "smoke_yaml": "epochs: 1",
            "smoke_readme": "# Smoke",
        }
    )
    backend = FakeBackend([ANALYZE, PLAN, MODULE_PY, SMOKE_TEST, bad_harness])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold.build_scaffold(make_paper(), backend, max_chars=100_000, progress=lambda m: None)
    assert exc.value.stage == "harness"
```

- [ ] **Step 2: Run tests to verify they pass** (implementation exists from Task 3)

Run: `.venv/bin/python -m pytest tests/test_scaffold_stages.py -v`
Expected: 10 PASS. If either new test fails, fix `paperdigest/scaffold.py` (not the test) until it matches the interface above.

- [ ] **Step 3: Commit**

```bash
git add tests/test_scaffold_stages.py
git commit -m "test: end-to-end build_scaffold pipeline with fake backend"
```

---

### Task 5: `write_project` — static tree, LLM files, git init, cleanup on failure

**Files:**
- Modify: `paperdigest/scaffold.py` (append)
- Test: `tests/test_scaffold_write.py`

**Interfaces:**
- Consumes: `templates.*` (Task 2), `render.OutputExistsError`, `ScaffoldProject` (Task 3).
- Produces: `scaffold.write_project(project: ScaffoldProject, folder: Path, force: bool, progress=...) -> Path` — consumed by Task 6's CLI.

- [ ] **Step 1: Write the failing tests** — create `tests/test_scaffold_write.py`:

```python
import subprocess

import pytest

from paperdigest import scaffold
from paperdigest.render import OutputExistsError


def make_project():
    return scaffold.ScaffoldProject(
        arxiv_id="1706.03762",
        title="Tiny Transformers Explained",
        url="https://arxiv.org/abs/1706.03762",
        package="tiny_transformers_explained",
        model="fake-model",
        files={
            "src/tiny_transformers_explained/model.py": "def build_model(cfg): ...\n",
            "tests/test_smoke.py": "def test_ok():\n    assert True\n",
            "train.py": "print('train')\n",
            "evaluate.py": "print('eval')\n",
            "configs/base.yaml": "lr: 0.001\n",
            "configs/smoke.yaml": "epochs: 1\n",
            "experiments/exp001_smoke/README.md": "# Smoke\n",
            "README.md": "# Tiny Transformers\n",
            "EXPERIMENTS.md": "# Experiments\n",
        },
    )


def test_write_project_creates_full_tree(tmp_path):
    folder = tmp_path / "2017-tiny-transformers-explained"
    out = scaffold.write_project(make_project(), folder, force=False, progress=lambda m: None)
    assert out == folder
    for rel in (
        "README.md",
        "EXPERIMENTS.md",
        "pyproject.toml",
        ".gitignore",
        "train.py",
        "configs/base.yaml",
        "src/tiny_transformers_explained/__init__.py",
        "src/tiny_transformers_explained/tracking.py",
        "src/tiny_transformers_explained/model.py",
        "tests/test_smoke.py",
        "experiments/runs.jsonl",
        "experiments/exp001_smoke/README.md",
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "notebooks/.gitkeep",
        "reports/figures/.gitkeep",
        "logs/.gitkeep",
    ):
        assert (folder / rel).exists(), rel
    assert 'name = "tiny_transformers_explained"' in (folder / "pyproject.toml").read_text()


def test_write_project_git_inits_with_initial_commit(tmp_path):
    folder = tmp_path / "proj"
    scaffold.write_project(make_project(), folder, force=False, progress=lambda m: None)
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=folder, capture_output=True, text=True, check=True
    ).stdout
    assert "arXiv:1706.03762" in log


def test_write_project_refuses_existing_without_force(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    with pytest.raises(OutputExistsError):
        scaffold.write_project(make_project(), folder, force=False, progress=lambda m: None)


def test_write_project_force_replaces_existing(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "old.txt").write_text("stale")
    scaffold.write_project(make_project(), folder, force=True, progress=lambda m: None)
    assert not (folder / "old.txt").exists()
    assert (folder / "README.md").exists()


def test_write_failure_leaves_no_partial_dir(tmp_path, monkeypatch):
    folder = tmp_path / "proj"
    monkeypatch.setattr(scaffold.templates, "PYPROJECT", None)  # None.format(...) raises mid-write
    with pytest.raises(AttributeError):
        scaffold.write_project(make_project(), folder, force=False, progress=lambda m: None)
    assert not folder.exists()


def test_write_project_survives_missing_git(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold.shutil, "which", lambda cmd: None)
    messages = []
    folder = tmp_path / "proj"
    scaffold.write_project(make_project(), folder, force=False, progress=messages.append)
    assert (folder / "README.md").exists()
    assert any("git" in m.lower() for m in messages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scaffold_write.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'write_project'`

- [ ] **Step 3: Implement** — in `paperdigest/scaffold.py`: add `import shutil`
and `import subprocess` to the imports, `from . import templates`, and
`from .render import OutputExistsError`; append:

```python
def _git_init(folder: Path, arxiv_id: str, progress: Callable[[str], None]) -> None:
    git = shutil.which("git")
    if git is None:
        progress("WARNING: git not found — skipping git init (project kept without version control)")
        return
    try:
        subprocess.run([git, "init", "-q"], cwd=folder, check=True)
        subprocess.run([git, "add", "-A"], cwd=folder, check=True)
        subprocess.run(
            [git, "-c", "user.name=paperdigest", "-c", "user.email=paperdigest@localhost",
             "commit", "-q", "-m", f"Scaffold generated from arXiv:{arxiv_id} by paperdigest"],
            cwd=folder, check=True,
        )
    except subprocess.CalledProcessError as e:
        progress(f"WARNING: git init failed ({e}) — project kept without version control")


def write_project(project: ScaffoldProject, folder: Path, force: bool,
                  progress: Callable[[str], None] = _default_progress) -> Path:
    if folder.exists():
        if not force:
            raise OutputExistsError(
                f"{folder} already exists — re-run with --force to overwrite (your edits there will be lost)"
            )
        shutil.rmtree(folder)
    try:
        for d in templates.STATIC_DIRS:
            (folder / d).mkdir(parents=True)
            (folder / d / ".gitkeep").touch()
        (folder / "experiments").mkdir()
        (folder / "experiments" / "runs.jsonl").touch()
        pkg_dir = folder / "src" / project.package
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "tracking.py").write_text(templates.TRACKING_PY)
        (folder / ".gitignore").write_text(templates.GITIGNORE)
        (folder / "pyproject.toml").write_text(templates.PYPROJECT.format(name=project.package))
        for rel, content in project.files.items():
            path = folder / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content if content.endswith("\n") else content + "\n")
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)  # never leave a half-project
        raise
    _git_init(folder, project.arxiv_id, progress)
    return folder
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scaffold_write.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add paperdigest/scaffold.py tests/test_scaffold_write.py
git commit -m "feat: write_project — static tree, LLM files, git init, cleanup on failure"
```

---

### Task 6: CLI — subcommand group with digest fallback, scaffold command, loud failure

**Files:**
- Modify: `paperdigest/cli.py` (restructure into a command group)
- Test: `tests/test_scaffold_cli.py`
- Existing `tests/test_cli.py` must pass **unmodified** — it is the backward-compatibility oracle.

**Interfaces:**
- Consumes: `scaffold.build_scaffold`, `scaffold.write_project`, `scaffold.project_folder`, `scaffold.ScaffoldError` (Tasks 3-5); everything digest already used.
- Produces: `paperdigest digest <ref> ...` (old behavior, also reachable as bare `paperdigest <ref> ...`), `paperdigest scaffold <ref> --dest <dir> [--force] [--backend] [--model] [--base-url] [--config]`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_scaffold_cli.py`:

```python
import json
import subprocess

from typer.testing import CliRunner

from conftest import FakeBackend
from paperdigest import cli

from test_scaffold_stages import full_responses  # tests/ is on sys.path under pytest

runner = CliRunner()


def _patch(monkeypatch, fixture_html, responses):
    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", lambda *a, **k: FakeBackend(responses))


def test_scaffold_end_to_end(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html, full_responses())
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "2017-tiny-transformers-explained"
    assert (folder / "README.md").exists()
    assert (folder / "train.py").exists()
    assert (folder / ".git").is_dir()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=folder, capture_output=True, text=True, check=True
    ).stdout
    assert "paperdigest" in log


def test_scaffold_stage_failure_is_loud(tmp_path, monkeypatch, fixture_html):
    monkeypatch.chdir(tmp_path)
    _patch(monkeypatch, fixture_html, ["this is not json"])
    dest = tmp_path / "projects"
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(dest)])
    assert result.exit_code == 1
    assert "ERROR" in result.output
    assert "stage:" in result.output and "analyze" in result.output
    debug = tmp_path / "scaffold-debug-analyze.txt"
    assert debug.exists()
    assert debug.read_text() == "this is not json"
    assert not (dest / "2017-tiny-transformers-explained").exists()


def test_scaffold_refuses_existing_before_llm(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html, full_responses())
    assert runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)]).exit_code == 0

    def boom(*a, **k):
        raise AssertionError("LLM backend must not be constructed when output exists")

    monkeypatch.setattr(cli.fetch, "fetch_html", lambda arxiv_id, cache_dir, client=None: fixture_html)
    monkeypatch.setattr(cli, "make_backend", boom)
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)])
    assert result.exit_code == 1
    assert "--force" in result.output


def test_scaffold_force_overwrites(tmp_path, monkeypatch, fixture_html):
    _patch(monkeypatch, fixture_html, full_responses())
    assert runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path)]).exit_code == 0
    _patch(monkeypatch, fixture_html, full_responses())
    result = runner.invoke(cli.app, ["scaffold", "1706.03762", "--dest", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output


def test_explicit_digest_subcommand_works(tmp_path, monkeypatch, fixture_html):
    outline = json.dumps(
        {
            "tldr": "Tiny transformers work.",
            "why_it_matters": "Cheap models are useful.",
            "concepts": [{"title": "Self-Attention", "section": "1 Introduction"}],
            "jargon": ["attention"],
            "self_test": ["What is attention?"],
        }
    )
    glossary = json.dumps({"terms": {"attention": "A weighting scheme."}})
    _patch(monkeypatch, fixture_html, [outline, "Attention body.", glossary])
    result = runner.invoke(cli.app, ["digest", "1706.03762", "--vault", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "Papers" / "2017-tiny-transformers-explained" / "00 Overview.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scaffold_cli.py -v`
Expected: FAIL — `paperdigest.cli` has no `scaffold` command (typer reports usage error / exit code 2).

- [ ] **Step 3: Implement** — replace `paperdigest/cli.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

import click
import typer
from typer.core import TyperGroup

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
        except click.UsageError:
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


@app.command()
def scaffold(
    ref: str = typer.Argument(..., help="arXiv URL or ID, e.g. https://arxiv.org/abs/1706.03762"),
    dest: Path = typer.Option(..., "--dest", help="directory the project folder is created under"),
    backend: str = typer.Option(None, "--backend", help="local (default) | anthropic | openai"),
    model: str = typer.Option(None, "--model"),
    base_url: str = typer.Option(None, "--base-url", help="local server URL (default http://localhost:8080/v1)"),
    config: Path = typer.Option(Path("config.toml"), "--config"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing project folder"),
):
    """Scaffold a git-initialized research project (structure, docs, stubs) from a paper."""
    try:
        cfg = load_config(config, backend=backend, model=model, base_url=base_url)
        arxiv_id = fetch.parse_arxiv_id(ref)
        _progress(f"Fetching arXiv {arxiv_id}...")
        html = fetch.fetch_html(arxiv_id, cfg.cache_dir)
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
        debug = Path(f"scaffold-debug-{safe_stage}.txt")
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
    except (fetch.FetchError, extract.ExtractError, LLMError, render.OutputExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        raise typer.Exit(1)
```

(The `digest` command body is byte-identical to the old `main` — only the
function name changes. `click` is a typer dependency, no new install.)

- [ ] **Step 4: Run the FULL suite — old CLI tests are the compat oracle**

Run: `.venv/bin/python -m pytest -v`
Expected: all PASS, including every test in `tests/test_cli.py` unmodified
(they invoke `cli.app` with a bare ref, exercising the `_DefaultDigestGroup`
fallback). If a `test_cli.py` test fails, fix `cli.py` — do not touch the test.

- [ ] **Step 5: Manual smoke of the help surface**

Run: `.venv/bin/paperdigest --help && .venv/bin/paperdigest scaffold --help`
Expected: both exit 0; top-level help lists `digest` and `scaffold`.

- [ ] **Step 6: Commit**

```bash
git add paperdigest/cli.py tests/test_scaffold_cli.py
git commit -m "feat: scaffold subcommand with loud failure; digest stays the default command"
```

---

### Task 7: README + live smoke test against the local 9B

**Files:**
- Modify: `README.md` (add a "Scaffold a project" section after "Quick start")
- Create: `docs/smoke-test-scaffold.md` (live results)

- [ ] **Step 1: Add README section** — insert after the Quick start section:

````markdown
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
````

- [ ] **Step 2: Verify the local server is up before the live test**

Run: `curl -s http://localhost:8001/v1/models`
Expected: JSON listing the served model. If this fails, STOP the live test,
record "server not reachable" in the notes below, and skip to Step 5 —
do not start a model server yourself.

- [ ] **Step 3: Run the live smoke test (9B on :8001)**

Run:
```bash
.venv/bin/paperdigest scaffold 1706.03762 --dest /tmp/claude-scaffold-smoke --base-url http://localhost:8001/v1
```
Expected: exit 0 and a printed project path — OR a loud ERROR banner naming the
failed stage (also a valid, recordable result; per the design there is no
auto-fallback). This may take several minutes; do not kill it early.

- [ ] **Step 4: Inspect the result**

Run: `ls -R /tmp/claude-scaffold-smoke/2017-*/ | head -40` and
`git -C /tmp/claude-scaffold-smoke/2017-*/ log --oneline`
Expected: full tree from the spec; one commit.

- [ ] **Step 5: Record results** — create `docs/smoke-test-scaffold.md` in the
style of the existing smoke-test note in `docs/` (date, model, server, paper,
outcome per stage, verbatim error banner if any, wall-clock time if observed —
"not measured" is fine). Record what actually happened; if the 9B failed a
stage, say which and include the banner.

- [ ] **Step 6: Run the full suite one last time**

Run: `.venv/bin/python -m pytest`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add README.md docs/smoke-test-scaffold.md
git commit -m "docs: scaffold usage + live smoke test results"
```
