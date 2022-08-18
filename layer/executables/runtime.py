import os
import runpy
import site
import subprocess  # nosec
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from layer.executables.packager import FunctionPackageInfo, get_function_package_info


ExecutablePath = Union[str, Path]


class BaseFunctionRuntime:
    """Base skeleton of a function runtime."""

    def __init__(self, executable_path: Path) -> None:
        self._executable_path = executable_path

    def initialise(self, package_info: FunctionPackageInfo) -> None:
        """Any initialisation required to run the function."""

    @property
    def executable_path(self) -> Path:
        return self._executable_path

    def install_packages(self, packages: Sequence[str]) -> None:
        """Installs packages required to run the function."""
        _run_pip_install(packages)

    def __call__(self, func: Callable[..., Any]) -> Any:
        """Called from the executable to run the function."""
        return func()

    def run_executable(self) -> Any:
        """Runs the packaged function."""
        return runpy.run_path(
            str(self.executable_path),
            run_name="__main__",
            init_globals={"__function_runtime": self},
        ).get("__function_return_result", None)

    @classmethod
    def execute(cls, executable_path: ExecutablePath, *args: Any, **kwargs: Any) -> Any:
        """Initialises the environment, installs packages and runs the executable."""
        local_executable_path = _get_local_executable_path(executable_path)
        _validate_executable_path(local_executable_path)
        package_info = get_function_package_info(local_executable_path)
        runtime = cls(local_executable_path, *args, **kwargs)
        runtime.initialise(package_info)
        runtime.install_packages(packages=package_info.pip_dependencies)
        return runtime.run_executable()

    @classmethod
    def main(
        cls, add_cli_args: Optional[Callable[[ArgumentParser], None]] = None
    ) -> None:
        parser = ArgumentParser(description="Function runtime")

        parser.add_argument(
            "executable_path",
            type=str,
            help="the local file path of the executable",
        )

        if add_cli_args is not None:
            add_cli_args(parser)

        args = parser.parse_args()

        cls.execute(**vars(args))


def _get_local_executable_path(executable_path: ExecutablePath) -> Path:
    if not isinstance(executable_path, str):
        return executable_path

    from urllib import parse, request

    uri = parse.urlparse(executable_path)
    if uri.scheme == "":
        return Path(executable_path)
    if uri.scheme == "file":
        return Path(uri.path)
    elif uri.scheme == "http" or uri.scheme == "https":
        local_path, _ = request.urlretrieve(executable_path)  # nosec
        return Path(local_path)
    else:
        raise ValueError(f"unsupported scheme: {uri.scheme}")


def _validate_executable_path(executable_path: Path) -> None:
    if not executable_path:
        raise FunctionRuntimeError("executable path is required")
    if not executable_path.exists():
        raise FunctionRuntimeError(f"executable path does not exist: {executable_path}")
    if not os.path.isfile(str(executable_path)):
        raise FunctionRuntimeError(f"executable path is not a file: {executable_path}")


class FunctionRuntimeError(Exception):
    pass


def _run_pip_install(packages: Sequence[str]) -> None:
    if len(packages) == 0:
        return

    pip_install = [
        sys.executable,
        "-m",
        "pip",
        "--quiet",
        "--disable-pip-version-check",
        "--no-color",
        "install",
    ] + list(packages)

    result = subprocess.run(  # nosec
        pip_install,
        text=True,
        check=False,
        capture_output=True,
    )

    if result.returncode != 0:
        raise FunctionRuntimeError(f"package instalation failed:\n{result.stderr}")

    # if site packages is not writeable, user site is used by pip to install dependencies
    # add it to the path if it's not there already
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)


if __name__ == "__main__":
    BaseFunctionRuntime.main()
