import os
import subprocess
import sys
from http.server import executable
from pathlib import Path

import pytest

from layer.executables.tar import build_executable_tar


TEST_ENTRYPOINT_PATH = Path(__file__).parent / "assets" / "entrypoint.py"


@pytest.fixture(autouse=True)
def python_executable_path():
    os.environ["PYTHON_EXECUTABLE_PATH"] = sys.executable


def func_simple() -> None:
    print("running simple function")


def func_with_resources() -> None:
    with open("test/e2e/assets/data/test.csv", "r") as file:
        contents = file.read()
    print(f"reading file of size {len(contents)}")


def func_with_dependencies() -> None:
    import click  # type: ignore
    import requests  # type: ignore

    print(f"running requests version {requests.__version__}")
    print(f"running click version {click.__version__}")


def test_executable_tar(tmpdir: Path) -> None:
    executable_path = tmpdir / "test.bsx"
    build_executable_tar(executable_path, func_simple, TEST_ENTRYPOINT_PATH)

    result = subprocess.run(["sh", executable_path], capture_output=True)
    assert b"running simple function" in result.stdout


def test_executable_tar_with_resources(tmpdir: Path) -> None:
    executable_path = tmpdir / "test.bsx"

    build_executable_tar(
        executable_path,
        func_with_resources,
        TEST_ENTRYPOINT_PATH,
        resources=[Path("test/e2e/assets/data/test.csv")],
    )

    result = subprocess.run(["sh", executable_path], capture_output=True)
    assert b"reading file of size 67" in result.stdout


def test_executable_tar_with_dependencies(tmpdir: Path) -> None:
    executable_path = tmpdir / "test.bsx"
    build_executable_tar(
        executable_path,
        func_with_dependencies,
        TEST_ENTRYPOINT_PATH,
        pip_dependencies=["requests==2.28.0", "click==8.1.3"],
    )

    result = subprocess.run(["sh", executable_path], capture_output=True)
    assert b"running requests version 2.28.0" in result.stdout
    assert b"running click version 8.1.3" in result.stdout
