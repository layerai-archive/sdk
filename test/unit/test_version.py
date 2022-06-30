import pathlib

import toml

import layer


def test_package_version():
    pyproject_toml = toml.load(
        pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    )
    pyproject_toml_version = pyproject_toml["tool"]["poetry"]["version"]

    assert pyproject_toml_version == layer.__version__
