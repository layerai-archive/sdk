import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePath

import pytest

from layer.executables.packager import (
    FunctionPackageInfo,
    get_function_package_info,
    package_function,
)


def func_simple() -> None:
    print("running simple function")


def test_executable_has_execute_permissions(tmpdir: Path):
    executable = package_function(func_simple, output_dir=tmpdir)

    assert os.access(executable, os.X_OK)


def test_execute_func_simple_as_python_script(tmpdir: Path):
    executable = package_function(func_simple, output_dir=tmpdir)

    subprocess.check_call([sys.executable, executable])


def test_package_contents(tmpdir: Path):
    resources_parent = Path("test") / "unit" / "executables" / "data"
    resource_paths = [
        resources_parent / "1",
        resources_parent / "dir" / "a",
        resources_parent / "dir" / "2",
    ]

    def func():
        pass

    executable = package_function(func, resources=resource_paths, output_dir=tmpdir)

    with zipfile.ZipFile(executable) as exec:
        exec_entries = {entry.filename for entry in exec.infolist()}
        assert exec_entries == {
            "requirements.txt",
            "function",
            "metadata.json",
            "__main__.py",
            "resources/",
            "resources/test/",
            "resources/test/unit/",
            "resources/test/unit/executables/",
            "resources/test/unit/executables/data/",
            "resources/test/unit/executables/data/1",
            "resources/test/unit/executables/data/dir/",
            "resources/test/unit/executables/data/dir/2",
            "resources/test/unit/executables/data/dir/a/",
            "resources/test/unit/executables/data/dir/a/3",
            "resources/test/unit/executables/data/dir/a/b/",
            "resources/test/unit/executables/data/dir/a/b/4",
            "cloudpickle/",
            "cloudpickle/LICENSE",
            "cloudpickle/__init__.py",
            "cloudpickle/cloudpickle.py",
            "cloudpickle/cloudpickle_fast.py",
            "cloudpickle/compat.py",
        }


def test_get_function_package_info_with_pip_dependencies(tmpdir: Path):
    def func_with_pip_dependencies():
        pass

    executable = package_function(
        func_with_pip_dependencies,
        pip_dependencies=["package1=0.0.1", "package2"],
        output_dir=tmpdir,
    )
    package_info = get_function_package_info(executable)

    assert package_info == FunctionPackageInfo(
        pip_dependencies=(
            "package1=0.0.1",
            "package2",
        ),
    )


def test_get_function_package_info_without_pip_dependecies(tmpdir: Path):
    def func_without_pip_dependencies():
        pass

    executable = package_function(
        func_without_pip_dependencies,
        pip_dependencies=[],
        output_dir=tmpdir,
    )
    package_info = get_function_package_info(executable)

    assert package_info == FunctionPackageInfo(pip_dependencies=())


def test_get_function_package_info_metadata(tmpdir: Path):
    def func_with_metadata():
        pass

    executable = package_function(
        func_with_metadata,
        metadata={"a": 1, "b": "2", "c": [3, 4], "d": {"e": 5}},
        output_dir=tmpdir,
    )
    package_info = get_function_package_info(executable)

    assert package_info == FunctionPackageInfo(
        metadata={"a": 1, "b": "2", "c": [3, 4], "d": {"e": 5}}
    )


def test_get_function_package_info_empty(tmpdir: Path):
    def func_empty():
        pass

    executable = package_function(func_empty, output_dir=tmpdir)
    package_info = get_function_package_info(executable)

    assert package_info == FunctionPackageInfo()


class CallableClass:
    def __call__(self):
        return 42


class CallableMethod:
    def x(self):
        return 42


@pytest.mark.parametrize(
    "callable",
    [(lambda: 42), (CallableClass(),), (CallableMethod().x,), ("e",), (42,)],
)
def test_only_funcions_could_be_packaged(callable):
    with pytest.raises(ValueError, match=r"function must be a function"):
        package_function(callable)


def test_package_same_function_to_the_same_output_dir(tmpdir: Path):
    exec1 = package_function(func_simple, output_dir=tmpdir)
    exec2 = package_function(func_simple, output_dir=tmpdir)

    assert exec1 == exec2
    assert PurePath(exec1).name == "func_simple"
