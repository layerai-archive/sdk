import subprocess
from pathlib import Path

from layer.executables.tar import build_executable_tar


def func_simple() -> None:
    print("running simple function")


def func_with_dependency() -> None:
    import click  # type: ignore
    import requests  # type: ignore

    print(f"running requests version {requests.__version__}")
    print(f"running click version {click.__version__}")


def test_executable_tar(tmpdir: Path) -> None:
    executable_path = tmpdir / "test.bsx"
    build_executable_tar(executable_path, func_simple)

    result = subprocess.run(["sh", executable_path], capture_output=True)
    assert b"running simple function" in result.stdout


def test_executable_tar_with_dependencies(tmpdir: Path) -> None:
    executable_path = tmpdir / "test.bsx"
    build_executable_tar(
        executable_path,
        func_with_dependency,
        pip_dependencies=["requests==2.28.0", "click==8.1.3"],
    )

    result = subprocess.run(["sh", executable_path], capture_output=True)
    assert b"running requests version 2.28.0" in result.stdout
    assert b"running click version 8.1.3" in result.stdout
