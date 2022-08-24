import sys
from typing import Any, Callable

from .version import check_latest_version


def sdk_function(func: Callable[..., Any]) -> Callable[..., Any]:
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
            "Windows is not supported. Please, have a look at https://docs.app.layer.ai/docs/installation"
        )
        sys.exit()
