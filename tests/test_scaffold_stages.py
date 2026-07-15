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
SMOKE_TEST = (
    "from tiny_transformers_explained import model\n\n\n"
    "def test_api_exists():\n    assert callable(model.build_model)\n"
)


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
    stubs = scaffold._build_stub_files(
        make_paper(), backend, "tiny_transformers_explained", max_chars=100_000, progress=lambda m: None,
    )
    assert "src/tiny_transformers_explained/model.py" in stubs


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


def test_stage_json_tolerates_trailing_junk():
    backend = FakeBackend([ANALYZE + "`"])
    data = scaffold._stage_json(backend, "analyze", "sys", "user", required=("components",))
    assert data["datasets"] == ["toy-corpus"]


def test_stage_json_requests_json_mode():
    backend = FakeBackend([ANALYZE])
    scaffold._stage_json(backend, "analyze", "sys", "user", required=("components",))
    assert backend.json_modes == [True]


def test_stage_python_does_not_use_json_mode():
    backend = FakeBackend([MODULE_PY])
    scaffold._stage_python(backend, "module:model.py", "sys", "user")
    assert backend.json_modes == [False]


PROMPT_ECHO_PY = (
    "src/attention_is_all_you_need/encoder.py\n"  # parses as a division expression
    "import torch\n\n"
    "def build_encoder(cfg: dict):\n"
    '    """Build the encoder. See paper §3.1."""\n'
    "    raise NotImplementedError\n"
)


def test_stage_python_rejects_prompt_echo_line():
    backend = FakeBackend([PROMPT_ECHO_PY])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold._stage_python(backend, "module:encoder.py", "sys", "user")
    assert exc.value.stage == "module:encoder.py"
    assert "line 1" in str(exc.value)
    assert exc.value.raw == PROMPT_ECHO_PY


def test_stage_python_allows_docstrings_and_toplevel_calls():
    code = '"""Module docstring."""\nimport sys\n\n\ndef main():\n    pass\n\n\nmain()\n'
    backend = FakeBackend([code])
    assert scaffold._stage_python(backend, "module:x.py", "sys", "user") == code.strip()


CLEAN_STUB_PY = (
    "import numpy as np\n"
    "import torch\n\n\n"
    "def build_model(cfg: dict):\n"
    '    """Build the encoder. See paper §3."""\n'
    "    # TODO(paper §3): implement the encoder\n"
    "    raise NotImplementedError\n"
)


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\n",
        "from socket import socket\n",
        'import os\nos.system("x")\n',
        'eval("x")\n',
    ],
)
def test_check_python_safety_flags_dangerous_code(code):
    violations = scaffold.check_python_safety(code, "bad.py")
    assert violations


def test_check_python_safety_passes_clean_stub():
    assert scaffold.check_python_safety(CLEAN_STUB_PY, "model.py") == []


def test_stage_python_rejects_unsafe_module():
    backend = FakeBackend(['import os\nos.system("rm -rf /")\n'])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold._stage_python(backend, "module:model.py", "sys", "user", filename="model.py")
    assert "safety scan" in str(exc.value)


def test_build_scaffold_aborts_when_stub_contains_os_system():
    unsafe_module = 'import os\n\n\ndef build_model(cfg: dict):\n    os.system("whoami")\n'
    backend = FakeBackend([ANALYZE, PLAN, unsafe_module])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold.build_scaffold(make_paper(), backend, max_chars=100_000, progress=lambda m: None)
    assert exc.value.stage == "module:model.py"
    assert "safety scan" in str(exc.value)


def test_plan_stage_rejects_duplicate_module_filenames():
    dup_plan = json.dumps(
        {
            "modules": [
                {"filename": "utils/model.py", "responsibility": "a", "api": [], "dependencies": []},
                {"filename": "model.py", "responsibility": "b", "api": [], "dependencies": []},
            ]
        }
    )
    backend = FakeBackend([ANALYZE, dup_plan])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold.build_scaffold(make_paper(), backend, max_chars=100_000, progress=lambda m: None)
    assert exc.value.stage == "plan"
    assert "model.py" in str(exc.value)


def test_harness_rejects_prompt_echo_in_train_py():
    echo_harness = json.dumps(
        {
            "train_py": PROMPT_ECHO_PY,
            "evaluate_py": TRAIN_PY,
            "base_yaml": "lr: 0.001",
            "smoke_yaml": "epochs: 1",
            "smoke_readme": "# Smoke",
        }
    )
    backend = FakeBackend([ANALYZE, PLAN, MODULE_PY, SMOKE_TEST, echo_harness])
    with pytest.raises(scaffold.ScaffoldError) as exc:
        scaffold.build_scaffold(make_paper(), backend, max_chars=100_000, progress=lambda m: None)
    assert exc.value.stage == "harness"
