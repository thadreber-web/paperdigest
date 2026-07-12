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
