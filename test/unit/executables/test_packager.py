import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePath

import pytest

from layer.executables.packager import package_function


def test_executable_has_execute_permissions(tmpdir: Path):
    executable = package_function(func_simple, output_dir=tmpdir)

    assert os.access(executable, os.X_OK)


def test_execute_func_simple(tmpdir: Path):
    executable = package_function(func_simple, output_dir=tmpdir)
    result = _subprocess_run(executable)

    assert result.returncode == 0
    assert "running simple function" in result.stdout


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
            "resources/",
            "function",
            "__main__.py",
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
        }


def test_execute_func_with_resources(tmpdir: Path):
    resources_parent = Path("test") / "unit" / "executables" / "data"
    resource_paths = [
        resources_parent / "1",
        resources_parent / "dir" / "a",
        resources_parent / "dir" / "2",
    ]

    def func_with_resources():
        content = ""
        for resource_path in resource_paths:
            resource = Path(resource_path)
            if resource.is_file():
                with open(resource_path, "r") as f:
                    content += f"file:{resource}={f.read()},"
            elif resource.is_dir():
                content += f"dir:{resource},"
        print(f"content is '{content}'")

    executable = package_function(
        func_with_resources, resources=resource_paths, output_dir=tmpdir
    )
    result = _subprocess_run(executable, cwd=tmpdir)

    assert result.returncode == 0
    assert (
        "content is 'file:test/unit/executables/data/1=1,dir:test/unit/executables/data/dir/a,file:test/unit/executables/data/dir/2=2,'"
        in result.stdout
    )


def test_executable_with_dependencies(tmpdir: Path):
    executable = package_function(
        func_with_dependencies,
        pip_dependencies=["requests==2.28.0", "click==8.1.3"],
        output_dir=tmpdir,
    )
    result = _subprocess_run(executable)

    assert result.returncode == 0
    assert "running requests version 2.28.0" in result.stdout
    assert "running click version 8.1.3" in result.stdout


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


@pytest.mark.skip(
    reason="TODO: https://linear.app/layer/issue/LAY-3451/handle-package-conflicts-between-user-and-runtime-dependencies"
)
def test_cloudpickle_already_defined_in_pip_requirements(tmpdir: Path):
    def use_cloudpickle():
        import cloudpickle  # type: ignore

        print(f"running cloudpickle version {cloudpickle.__version__}")

    executable = package_function(
        use_cloudpickle,
        pip_dependencies=["cloudpickle==1.3.0"],
        output_dir=tmpdir,
    )
    result = _subprocess_run(executable)

    assert result.returncode == 0
    assert "running cloudpickle version 2.1.0" in result.stdout


def test_package_same_function_to_the_same_output_dir(tmpdir: Path):
    exec1 = package_function(func_simple, output_dir=tmpdir)
    exec2 = package_function(func_simple, output_dir=tmpdir)

    assert exec1 == exec2
    assert PurePath(exec1).name == "func_simple"


def func_simple() -> None:
    print("running simple function")


def func_with_dependencies() -> None:
    import click  # type: ignore
    import requests  # type: ignore

    print(f"running requests version {requests.__version__}")
    print(f"running click version {click.__version__}")


def _subprocess_run(executable, cwd=None):
    return subprocess.run(
        [executable], capture_output=True, text=True, shell=True, cwd=cwd
    )
