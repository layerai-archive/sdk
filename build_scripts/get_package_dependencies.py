import pathlib
from typing import Dict

import toml


default_dependencies = {"pyproject.toml", "DESCRIPTION.md"}


def get_path_from_package(package: Dict[str, str]) -> str:
    include = package["include"]
    dir = package.get("from")

    if dir:
        return f"{dir}/{include}/**"
    else:
        return f"{include}/**"


if __name__ == "__main__":
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"

    pyproject = toml.load(pyproject_path)
    packages = pyproject["tool"]["poetry"]["packages"]
    includes_paths = {get_path_from_package(package) for package in packages}

    all_paths = default_dependencies.union(includes_paths)

    print(",".join(all_paths))
