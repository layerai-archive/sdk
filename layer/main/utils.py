from typing import Any, Callable

from .version import check_latest_version


def sdk_function(func: Callable[..., Any]) -> Callable[..., Any]:
    def inner(*args: Any, **kwargs: Any) -> Any:
        check_latest_version()
        return func(*args, **kwargs)

    return inner
