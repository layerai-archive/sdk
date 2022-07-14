from pathlib import Path
from unittest.mock import patch

import pytest

from layer.decorators.dataset_decorator import dataset
from layer.decorators.model_decorator import model
from layer.decorators.pip_requirements_decorator import pip_requirements
from layer.decorators.resources_decorator import resources
from layer.executables.function import (
    DatasetOutput,
    Function,
    FunctionError,
    ModelOutput,
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
    assert function.resources == [
        Path("path/to/resource"),
        Path("path/to/other/resource"),
    ]


def test_from_decorated_pip_dependencies_packages():
    @dataset("test_dataset")
    @pip_requirements(packages=["package1", "package2==0.0.42"])
    def dataset_function():
        return [42]

    function = Function.from_decorated(dataset_function)
    assert function.pip_dependencies == [
        "package1",
        "package2==0.0.42",
    ]


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
    assert function.pip_dependencies == [
        "package1",
        "package2==0.0.42",
    ]


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
            resources=[Path("path/to/resource"), Path("path/to/other/resource")],
            pip_dependencies=["package1", "package2==0.0.42"],
        )
