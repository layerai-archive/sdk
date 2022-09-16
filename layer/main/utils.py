import functools
import json
import os
import pathlib
import re
import sys
import urllib.request
from typing import Any, Callable


_HAS_SHOWN_PYPI_UPDATE_MESSAGE = False
_HAS_SHOWN_PYTHON_VERSION_MESSAGE = False


def sdk_function(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def inner(*args: Any, **kwargs: Any) -> Any:
        _check_os()
        check_latest_version()
        return func(*args, **kwargs)

    return inner


def _check_os() -> None:
    import platform

    def is_windows_os() -> bool:
        return platform.system() == "Windows"

    if is_windows_os():
        print(
            "Windows is not supported. Please, have a look at https://docs.layer.ai/docs/installation"
        )
        sys.exit()


def get_version() -> str:
    with open(
        pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    ) as pyproject:
        text = pyproject.read()
        # Use a simple regex to avoid a dependency on toml
        version_match = re.search(r'version = "(\d+\.\d+\.\d+.*)"', text)

    if version_match is None:
        raise RuntimeError("Failed to parse version")
    return version_match.group(1)


def get_latest_version() -> str:
    pypi_url = "https://pypi.org/pypi/layer/json"
    response = urllib.request.urlopen(pypi_url).read().decode()  # nosec urllib_urlopen
    data = json.loads(response)

    return data["info"]["version"]


def check_latest_version() -> None:
    global _HAS_SHOWN_PYPI_UPDATE_MESSAGE
    if _HAS_SHOWN_PYPI_UPDATE_MESSAGE or "LAYER_DISABLE_UI" in os.environ:
        return

    latest_version = get_latest_version()
    current_version = get_version()
    if current_version != latest_version:
        print(
            f"You are using the version {current_version} but the latest version is {latest_version}, please upgrade with 'pip install --upgrade layer'"
        )
    _HAS_SHOWN_PYPI_UPDATE_MESSAGE = True


def check_python_version() -> None:
    global _HAS_SHOWN_PYTHON_VERSION_MESSAGE
    if _HAS_SHOWN_PYTHON_VERSION_MESSAGE or "LAYER_DISABLE_UI" in os.environ:
        return

    import platform

    major, minor, _ = platform.python_version_tuple()

    if major != "3" or minor not in ["7", "8"]:
        print(
            f"You are using the Python version {platform.python_version()} but layer requires Python 3.7.x or 3.8.x"
        )
    _HAS_SHOWN_PYTHON_VERSION_MESSAGE = True
