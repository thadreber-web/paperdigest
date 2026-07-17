from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import templates
from .digest import _paper_body
from .extract import Paper
from .llm import Backend, LLMError, complete_with_retry, repair_json, run_tasks, strip_fences
from .render import OutputExistsError, folder_name, paper_folder, slugify


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
    analysis: dict = field(default_factory=dict)
    modules: list[dict] = field(default_factory=list)


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
    return dest / folder_name(arxiv_id, title)


def _stage_json(backend: Backend, stage: str, system: str, user: str, required: tuple[str, ...]) -> dict:
    try:
        # complete_with_retry only retries transport/SDK errors; bad *content* is handled
        # below via a one-shot repair round-trip (mirrors digest._call_json).
        raw = complete_with_retry(backend, system, user, json_mode=True)
    except LLMError as e:
        raise ScaffoldError(stage, str(e)) from e
    try:
        # raw_decode reads the first complete JSON value and ignores trailing junk
        # (small local models sometimes leak a stray fence character after the JSON)
        data, _ = json.JSONDecoder().raw_decode(strip_fences(raw))
    except json.JSONDecodeError:
        try:
            repaired = repair_json(backend, raw)
            data, _ = json.JSONDecoder().raw_decode(strip_fences(repaired))
        except (LLMError, json.JSONDecodeError) as e:
            raise ScaffoldError(
                stage, f"model returned unparseable JSON even after a repair attempt: {e}", raw=raw
            ) from e
    if not isinstance(data, dict):
        raise ScaffoldError(stage, f"expected a JSON object, got {type(data).__name__}", raw=raw)
    missing = [k for k in required if k not in data]
    if missing:
        raise ScaffoldError(stage, f"response is missing fields: {missing}", raw=raw)
    return data


_BANNED_IMPORTS = (
    "subprocess", "socket", "ctypes", "urllib", "http.client",
    "requests", "ftplib", "telnetlib", "smtplib",
)
_BANNED_CALL_NAMES = ("eval", "exec", "compile", "__import__")
_BANNED_OS_EXACT = ("system", "popen", "fork")
_BANNED_OS_PREFIXES = ("exec", "spawn")
_BANNED_PICKLE = ("load", "loads")
_BANNED_SHUTIL = ("rmtree",)


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def _flag(self, node: ast.AST, message: str) -> None:
        self.violations.append(f"line {node.lineno}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if any(alias.name == b or alias.name.startswith(b + ".") for b in _BANNED_IMPORTS):
                self._flag(node, f"import of banned module {alias.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if any(module == b or module.startswith(b + ".") for b in _BANNED_IMPORTS):
            self._flag(node, f"import from banned module {module!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _BANNED_CALL_NAMES:
            self._flag(node, f"call to banned builtin {func.id}()")
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            root, attr = func.value.id, func.attr
            if root == "os" and (attr in _BANNED_OS_EXACT or attr.startswith(_BANNED_OS_PREFIXES)):
                self._flag(node, f"call to banned function os.{attr}()")
            elif root == "pickle" and attr in _BANNED_PICKLE:
                self._flag(node, f"call to banned function pickle.{attr}()")
            elif root == "shutil" and attr in _BANNED_SHUTIL:
                self._flag(node, f"call to banned function shutil.{attr}()")
        self.generic_visit(node)


def check_python_safety(code: str, filename: str) -> list[str]:
    """Scan LLM-generated Python for dangerous imports/calls (subprocess, network, eval, etc).

    Returns a list of human-readable violation descriptions (empty if none). Callers
    should abort the run on any violation — this is a coarse AST scan, not a sandbox.
    """
    tree = ast.parse(code, filename=filename)
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    return visitor.violations


def _validate_python(stage: str, code: str, raw: str, filename: str = "<generated>") -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ScaffoldError(stage, f"generated Python does not parse: {e}", raw=raw) from e
    # Prompt echoes like `src/pkg/encoder.py` parse as name-division expressions,
    # so ast.parse alone misses them. Bare top-level expressions other than
    # docstrings and calls are junk in generated files.
    for node in tree.body:
        if isinstance(node, ast.Expr):
            value = node.value
            if isinstance(value, ast.Call):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue
            raise ScaffoldError(
                stage,
                f"generated Python has a junk top-level expression at line {node.lineno} "
                f"(likely a prompt echo): {ast.unparse(node).strip()[:80]!r}",
                raw=raw,
            )
    violations = check_python_safety(code, filename)
    if violations:
        raise ScaffoldError(
            stage,
            f"generated file {filename} failed the safety scan:\n" + "\n".join(violations),
            raw=raw,
        )


def _stage_python(backend: Backend, stage: str, system: str, user: str, filename: str = "<generated>") -> str:
    try:
        raw = complete_with_retry(backend, system, user)
    except LLMError as e:
        raise ScaffoldError(stage, str(e)) from e
    code = strip_fences(raw)
    _validate_python(stage, code, raw, filename)
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
    # Directory stripping above can make two distinct planned paths collide on the same
    # filename, silently overwriting one in _build_stub_files. Also guard against
    # collisions with the fixed files write_project always creates alongside modules.
    seen: dict[str, int] = {}
    for m in modules:
        seen[m["filename"]] = seen.get(m["filename"], 0) + 1
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    reserved = sorted(name for name in seen if name in ("__init__.py", "tracking.py"))
    if duplicates or reserved:
        problems = [f"{name!r} (used {seen[name]}x)" for name in duplicates]
        problems += [f"{name!r} (reserved, always created by paperdigest)" for name in reserved]
        raise ScaffoldError(
            "plan", f"module plan has filename collisions: {', '.join(problems)}", raw=json.dumps(plan)
        )
    return modules


def _build_stub_files(paper: Paper, backend: Backend, pkg: str, max_chars: int,
                      progress: Callable[[str], None],
                      workers: int = 1) -> tuple[dict[str, str], dict, list[dict]]:
    """Stages 1-3: analyze, plan, module stubs + smoke test. Returns (files, analysis, modules)."""
    body = _paper_body(paper, max_chars, progress)
    analysis = _analyze(paper, backend, body, progress)
    modules = _plan_modules(paper, backend, body, analysis, progress)

    analysis_ctx = (
        f"PAPER ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
        f"MODULE PLAN:\n{json.dumps(modules, indent=2)}"
    )
    total = len(modules)

    def _write_stub(numbered: tuple[int, dict]) -> str:
        i, m = numbered
        progress(f"Stage 3/5: writing stub {i}/{total}: {m['filename']}")
        return _stage_python(
            backend, f"module:{m['filename']}", _MODULE_SYSTEM,
            f"FILE TO WRITE: src/{pkg}/{m['filename']}\n"
            f"RESPONSIBILITY: {m.get('responsibility', '')}\n"
            "PUBLIC API:\n" + "\n".join(f"- {sig}" for sig in m.get("api", []))
            + f"\n\n{analysis_ctx}\n\nPAPER TITLE: {paper.title}\n\nABSTRACT: {paper.abstract}",
            filename=f"src/{pkg}/{m['filename']}",
        )

    codes = run_tasks(list(enumerate(modules, 1)), _write_stub, workers=workers)
    files: dict[str, str] = {f"src/{pkg}/{m['filename']}": code for m, code in zip(modules, codes)}

    progress("Stage 3/5: writing tests/test_smoke.py")
    files["tests/test_smoke.py"] = _stage_python(
        backend, "smoke-test", _SMOKE_TEST_SYSTEM,
        f"PACKAGE: {pkg} (import as `from {pkg} import <module>`)\n\n{analysis_ctx}",
        filename="tests/test_smoke.py",
    )
    return files, analysis, modules


_SECTION_REF_RE = re.compile(r"§(\d+(?:\.\d+)*)")
_TODO_RE = re.compile(r"#\s*TODO\(paper §")


def _module_order(modules: list[dict]) -> list[dict]:
    """Dependency order (Kahn's algorithm); unknown deps ignored, cycles fall back to plan order."""
    names = [m["filename"] for m in modules]
    by_name = {m["filename"]: m for m in modules}
    deps = {n: [d for d in by_name[n].get("dependencies", []) if d in by_name and d != n] for n in names}
    ordered: list[str] = []
    while len(ordered) < len(names):
        ready = [n for n in names if n not in ordered and all(d in ordered for d in deps[n])]
        if not ready:  # cycle: append the rest in plan order
            ordered.extend(n for n in names if n not in ordered)
            break
        ordered.extend(ready)
    return [by_name[n] for n in ordered]


def build_agents_md(project: ScaffoldProject, vault: Path | None) -> str:
    """Deterministic coding-agent brief assembled from stage data. Pure: no disk I/O, no LLM."""
    pkg = project.package
    lines = [
        "# AGENTS.md — implementation brief",
        "",
        f'Generated by paperdigest from arXiv:{project.arxiv_id} — "{project.title}"',
        f"({project.url}). This project is a scaffold: stub modules plus a",
        "complete train/evaluate harness. Your job is to implement the stubs.",
        "",
        "## Mission",
        "",
        f"Implement every `TODO(paper §…)` stub under `src/{pkg}/` so the",
        "smoke experiment runs end-to-end. Work module by module in the order below. The paper",
        "is the spec; docstrings cite the governing section as `§x.y`.",
        "",
        "## Implementation order",
        "",
        "Dependency-ordered. Each entry lists the module's responsibility, its TODO count, and",
        "the paper sections its code cites.",
        "",
    ]
    for i, mod in enumerate(_module_order(project.modules), start=1):
        code = project.files.get(f"src/{pkg}/{mod['filename']}", "")
        todos = len(_TODO_RE.findall(code))
        sections = list(dict.fromkeys(_SECTION_REF_RE.findall(code)))
        entry = f"{i}. `{mod['filename']}` — {mod['responsibility']}"
        entry += f" {todos} TODO{'s' if todos != 1 else ''}"
        if sections:
            entry += " (" + ", ".join(f"§{s}" for s in sections) + ")"
        deps = [d for d in mod.get("dependencies", []) if d != mod["filename"]]
        if deps:
            entry += " — depends on: " + ", ".join(deps)
        lines.append(entry)
    lines += [
        "",
        "## Definition of done",
        "",
        "1. `python -m pytest tests/test_smoke.py` passes.",
        "2. `python train.py configs/smoke.yaml` exits 0 and appends a line to `experiments/runs.jsonl`.",
        "",
        "Do not weaken either check to make it pass.",
        "",
        "## Ground rules",
        "",
        f"- Do not modify `train.py`, `evaluate.py`, `src/{pkg}/tracking.py`,",
        "  or `tests/test_smoke.py` — implement the stubs to fit the harness, not the reverse.",
        "- Hyperparameters in `configs/base.yaml` are the paper's real values; do not change them.",
        "  `configs/smoke.yaml` is a toy run and must stay tiny.",
        "- Log every run through `tracking.log_run()` (the harness already does).",
        "- `EXPERIMENTS.md` lists the planned experiments; tick its checkboxes as they become runnable.",
        "",
        "## Paper notes",
        "",
    ]
    if vault is not None:
        notes = paper_folder(project.arxiv_id, project.title, vault)
        lines += [
            "Plain-English notes for this paper live in the Obsidian vault:",
            f"[`{notes}`]({notes.as_uri()})",
            "If that folder doesn't exist yet, generate it:",
            f"`paperdigest {project.arxiv_id} --vault {vault}`",
        ]
    else:
        lines += [
            "Generate plain-English Obsidian notes for this paper with:",
            f"`paperdigest {project.arxiv_id} --vault <your-vault>`",
        ]
    return "\n".join(lines) + "\n"


def build_scaffold(paper: Paper, backend: Backend, max_chars: int,
                   progress: Callable[[str], None] = _default_progress, workers: int = 1,
                   vault: Path | None = None) -> ScaffoldProject:
    pkg = package_name(paper.title)
    files, analysis, modules = _build_stub_files(paper, backend, pkg, max_chars, progress, workers=workers)
    analysis_ctx = (
        f"PAPER ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
        f"MODULE PLAN:\n{json.dumps(modules, indent=2)}"
    )

    progress("Stage 4/5: writing train/evaluate harness and configs...")
    harness = _stage_json(
        backend, "harness", _HARNESS_SYSTEM,
        f"PACKAGE: {pkg}\n\n{analysis_ctx}",
        required=("train_py", "evaluate_py", "base_yaml", "smoke_yaml", "smoke_readme"),
    )
    for key, filename in (("train_py", "train.py"), ("evaluate_py", "evaluate.py")):
        _validate_python("harness", str(harness[key]), raw=str(harness[key]), filename=filename)
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

    project = ScaffoldProject(
        arxiv_id=paper.arxiv_id, title=paper.title, url=paper.url,
        package=pkg, model=backend.model, files=files,
        analysis=analysis, modules=modules,
    )
    project.files["AGENTS.md"] = build_agents_md(project, vault)
    return project


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
