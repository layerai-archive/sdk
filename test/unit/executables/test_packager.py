import os
import subprocess
import sys
import zipfile
from pathlib import Path

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