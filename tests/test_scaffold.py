import tomllib
from importlib.metadata import version as dist_version
from pathlib import Path


def test_package_importable():
    import paperdigest

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    )
    assert paperdigest.__version__ == pyproject["project"]["version"]
    # Guard the name<->metadata link across renames: the version must resolve
    # from the distribution named in pyproject.toml, whatever that name is.
    assert dist_version(pyproject["project"]["name"]) == pyproject["project"]["version"]
