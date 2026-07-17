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
            "AGENTS.md": "# AGENTS.md — implementation brief\n",
        },
    )


def test_write_project_creates_full_tree(tmp_path):
    folder = tmp_path / "2017-tiny-transformers-explained"
    out = scaffold.write_project(make_project(), folder, force=False, progress=lambda m: None)
    assert out == folder
    for rel in (
        "README.md",
        "EXPERIMENTS.md",
        "AGENTS.md",
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
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=folder, capture_output=True, text=True, check=True
    ).stdout
    assert "AGENTS.md" in tracked.splitlines()


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
