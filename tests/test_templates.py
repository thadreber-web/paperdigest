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
