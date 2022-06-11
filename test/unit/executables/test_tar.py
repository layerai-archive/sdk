import subprocess
from pathlib import Path

from layer.executables.tar import build_executable_tar


def test_pickled_func() -> None:
    print("training test")


def test_executable_tar(tmpdir: Path) -> None:
    executable_path = tmpdir / "test.tar.gz"
    build_executable_tar(executable_path, test_pickled_func)

    result = subprocess.run(["sh", executable_path], capture_output=True)
    assert b"training test" in result.stdout
