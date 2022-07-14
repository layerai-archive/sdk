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
            "cloudpickle/",
            "cloudpickle/__init__.py",
            "cloudpickle/cloudpickle.py",
            "cloudpickle/cloudpickle_fast.py",
            "cloudpickle/compat.py",
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


def func_simple() -> None:
    print("running simple function")


def _subprocess_run(executable, cwd=None):
    return subprocess.run(
        [executable], capture_output=True, text=True, shell=True, cwd=cwd
    )
