import contextlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from layer.executables.packager import package_function
from layer.executables.runtime import BaseFunctionRuntime, FunctionRuntimeError


def test_executable_path_does_not_exist():
    executable_path = Path("does_not_exist")
    with pytest.raises(
        FunctionRuntimeError, match="executable path does not exist: does_not_exist"
    ):
        BaseFunctionRuntime.execute(executable_path)


def test_executable_path_is_not_a_file(tmpdir: Path):
    with pytest.raises(FunctionRuntimeError, match="executable path is not a file: "):
        BaseFunctionRuntime.execute(tmpdir)


def test_execute_func_simple(tmpdir: Path, capsys: Any):
    executable = package_function(func_simple, output_dir=tmpdir)
    BaseFunctionRuntime.execute(executable)

    assert "running simple function" in capsys.readouterr().out


def test_execute_func_with_resources(tmpdir: Path, capsys: Any):
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
    BaseFunctionRuntime.execute(executable)

    assert (
        "content is 'file:test/unit/executables/data/1=1,dir:test/unit/executables/data/dir/a,file:test/unit/executables/data/dir/2=2,'"
        in capsys.readouterr().out
    )


def test_execute_func_with_packages(tmpdir: Path):
    def function_with_packages():
        import marshmallow  # type: ignore

        print(f"running marshmallow version {marshmallow.__version__}")

    executable = package_function(
        function_with_packages,
        pip_dependencies=["marshmallow==3.17.0"],
        output_dir=tmpdir,
    )
    result = _execute_runtime_module(executable)

    assert "running marshmallow version 3.17.0" in result.stdout


def func_simple() -> None:
    print("running simple function")


@contextlib.contextmanager
def _virtual_env_python() -> Path:
    import venv

    with tempfile.TemporaryDirectory() as venv_dir:
        python_bin = Path(venv_dir) / "bin" / "python"
        venv.create(venv_dir, with_pip=True, system_site_packages=True)
        yield python_bin


def _execute_runtime_module(executable: Path) -> subprocess.CompletedProcess:
    with _virtual_env_python() as python_bin:
        # run in a virtual environment, not to mess the current one
        return subprocess.run(
            [python_bin, "-m", "layer.executables.runtime", str(executable)],
            check=True,
            capture_output=True,
            text=True,
        )
