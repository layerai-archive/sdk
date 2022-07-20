import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

import layer
from layer.decorators.assertions import assert_unique
from layer.decorators.dataset_decorator import dataset
from layer.decorators.fabric_decorator import fabric
from layer.decorators.model_decorator import model
from layer.decorators.pip_requirements_decorator import pip_requirements
from layer.decorators.resources_decorator import resources
from layer.executables.function import (
    DatasetOutput,
    Function,
    FunctionError,
    ModelOutput,
    _undecorate_function,
)


def test_from_decorated_dataset():
    @dataset("test_dataset")
    def dataset_function():
        return [
            1,
            2,
            3,
        ]

    function = Function.from_decorated(dataset_function)
    assert function.func == dataset_function
    assert function.output == DatasetOutput("test_dataset")


def test_from_decorated_model():
    @model("test_model")
    def model_function():
        return 42

    function = Function.from_decorated(model_function)
    assert function.func == model_function
    assert function.output == ModelOutput("test_model")


def test_from_decorated_raises_when_required_decorators_missing():
    def function():
        return 42

    with pytest.raises(
        FunctionError,
        match=r"either @dataset\(name=\"\.\.\.\"\) or @model\(name=\"\.\.\.\"\) top level decorator is required for each function\. Add @dataset or @model decorator on top of existing decorators to run functions in Layer",
    ):
        Function.from_decorated(function)


def test_from_decorated_resources():
    @dataset("test_dataset")
    @resources("path/to/resource", "path/to/other/resource")
    def dataset_function():
        return [42]

    function = Function.from_decorated(dataset_function)
    assert function.resources == (
        Path("path/to/resource"),
        Path("path/to/other/resource"),
    )


def test_from_decorated_pip_dependencies_packages():
    @dataset("test_dataset")
    @pip_requirements(packages=["package1", "package2==0.0.42"])
    def dataset_function():
        return [42]

    function = Function.from_decorated(dataset_function)
    assert function.pip_dependencies == (
        "package1",
        "package2==0.0.42",
    )


def test_from_decorated_pip_dependencies_requirements(tmp_path):
    pip_packages = ["package1", "package2==0.0.42"]
    requirements_path = tmp_path / "requirements.txt"

    with open(requirements_path, "w") as f:
        f.write("\n".join(pip_packages))

    @dataset("test_dataset")
    @pip_requirements(file=str(requirements_path))
    def dataset_function():
        return [42]

    function = Function.from_decorated(dataset_function)
    assert function.pip_dependencies == (
        "package1",
        "package2==0.0.42",
    )


def test_package_function():
    package_dir = Path("package/dir")
    package_path = Path("path/to/package")

    with patch("layer.executables.function.package_function") as package_function:
        package_function.return_value = package_path

        @dataset("test_dataset")
        @resources("path/to/resource", "path/to/other/resource")
        @pip_requirements(packages=["package1", "package2==0.0.42"])
        def dataset_function():
            return [42]

        function = Function.from_decorated(dataset_function)

        assert function.package(output_dir=package_dir) == package_path
        package_function.assert_called_once_with(
            dataset_function,
            output_dir=package_dir,
            resources=(Path("path/to/resource"), Path("path/to/other/resource")),
            pip_dependencies=("package1", "package2==0.0.42"),
            metadata=_default_function_metadata(
                output={"name": "test_dataset", "type": "dataset"}
            ),
        )


def test_undecorate_function_dataset_decorator():
    @dataset("x")
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_model_decorator():
    @model("x")
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_pip_requirements_decorator():
    @pip_requirements(packages=["package1", "package2==0.0.42"])
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_assert_unique_decorator():
    @assert_unique(["x"])
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_fabric_decorator():
    @fabric("f-medium")
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_resources_decorator():
    @resources("path/to/resource", "path/to/other/resource")
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_from_multiple_decorators():
    @dataset("x")
    @pip_requirements(packages=["package1", "package2==0.0.42"])
    def f1():
        return [42]

    assert type(_undecorate_function(f1)).__name__ == "function"


def test_undecorate_function_custom_decorator():
    def custom_decorator(func):
        def wrapper():
            return func() + 1

        return wrapper

    @custom_decorator
    def f1():
        return 42

    undecorated = _undecorate_function(f1)

    assert type(undecorated).__name__ == "function"
    assert undecorated() == 43


def test_undecorate_function_no_decorator():
    def f1():
        return 42

    undecorated = _undecorate_function(f1)

    assert type(undecorated).__name__ == "function"
    assert undecorated() == 42


def test_packaged_function_is_unwrapped_from_all_the_decorators(tmpdir: Path):
    @dataset("test_dataset")
    @resources("path/to/resource", "path/to/other/resource")
    @pip_requirements(packages=["package1", "package2==0.0.42"])
    def dataset_function():
        return [42]

    function = Function.from_decorated(dataset_function)
    executable = function.package(output_dir=tmpdir)

    # the packaged function should be an instance of a built-in function class
    assert type(function.func).__name__ == "function"
    # the decorators should have no effect on the function execution (fails otherwise)
    assert subprocess.check_call([sys.executable, str(executable)]) == 0


def test_dataset_asset_metadata():
    @dataset("d")
    def f1():
        return [42]

    function = Function.from_decorated(f1)

    assert function.metadata == _default_function_metadata(
        output={"name": "d", "type": "dataset"},
    )


def test_model_asset_metadata():
    @model("m")
    def f1():
        return [42]

    function = Function.from_decorated(f1)

    assert function.metadata == _default_function_metadata(
        output={"name": "m", "type": "model"},
    )


def _default_function_metadata(output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sdk": {"version": layer.__version__},
        "function": {
            "serializer": {"name": "layer.cloudpickle", "version": "2.1.0"},
            "output": output,
        },
    }
