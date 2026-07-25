import ast
import importlib.util
import json

from paperdigest import templates


def test_tracking_template_is_valid_python():
    ast.parse(templates.TRACKING_PY)


def test_pyproject_template_formats_with_package_name():
    text = templates.PYPROJECT.format(name="my_paper", dependencies='"torch", "numpy"')
    assert 'name = "my_paper"' in text
    assert 'packages = ["src/my_paper"]' in text
    assert 'dependencies = ["torch", "numpy"]' in text


def test_config_loader_parses_typed_values(tmp_path):
    ns = {}
    exec(templates.CONFIG_PY, ns)
    cfg_file = tmp_path / "base.yaml"
    cfg_file.write_text(
        "num_layers: 6\n"
        "dropout: 0.1\n"
        "epsilon: 1e-9\n"          # scientific notation -> float (the hand-rolled parser got this wrong)
        "betas: [0.9, 0.98]\n"
        "data_path: data/en_de.txt\n"
        "flag: true\n"
        "# a comment\n"
        "\n"
    )
    cfg = ns["load_config"](str(cfg_file))
    assert cfg["num_layers"] == 6 and isinstance(cfg["num_layers"], int)
    assert cfg["dropout"] == 0.1
    assert cfg["epsilon"] == 1e-9 and isinstance(cfg["epsilon"], float)
    assert cfg["betas"] == [0.9, 0.98]
    assert cfg["data_path"] == "data/en_de.txt"   # bare string stays a string
    assert cfg["flag"] is True
    assert not any(k.startswith("#") for k in cfg)


def test_config_loader_accepts_equals_separator(tmp_path):
    # The 9B sometimes emits `.env`-style `key = value` instead of YAML `key: value`.
    ns = {}
    exec(templates.CONFIG_PY, ns)
    cfg_file = tmp_path / "base.yaml"
    cfg_file.write_text("N=6\nd_model = 512\nepsilon=1e-9\n")
    cfg = ns["load_config"](str(cfg_file))
    assert cfg["N"] == 6
    assert cfg["d_model"] == 512
    assert cfg["epsilon"] == 1e-9 and isinstance(cfg["epsilon"], float)


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
