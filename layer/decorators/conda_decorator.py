from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import wrapt  # type: ignore

from layer.contracts.conda import CondaEnv
from layer.decorators.layer_wrapper import LayerFunctionWrapper


def conda(
    environment_file: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
) -> Callable[..., Any]:
    if environment_file and environment:
        raise ValueError(
            "either environment_file or environment dictionary should be provided, not both."
        )
    if environment_file:
        conda_environment = CondaEnv.load_from_file(path=Path(environment_file))

    if environment:
        conda_environment = CondaEnv(environment=environment)

    @wrapt.decorator(proxy=_conda_wrapper(conda_environment))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _conda_wrapper(
    conda_environment: CondaEnv,
) -> Any:
    class CondaFunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(wrapped, wrapper, enabled)
            self.layer.set_conda_environment(conda_environment)

    return CondaFunctionWrapper
