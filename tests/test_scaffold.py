import tomllib
from pathlib import Path


def test_package_importable():
    import paperdigest

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    )
    assert paperdigest.__version__ == pyproject["project"]["version"]
