import pathlib
from abc import ABC
from typing import Any, Callable, Dict, List, Sequence, Union

import wrapt  # type: ignore

from layer.contracts.runs import ResourcePath
from layer.decorators.layer_wrapper import LayerFunctionWrapper


def resources(
    path: Union[str, pathlib.Path], *paths: Union[str, pathlib.Path]
) -> Callable[..., Any]:
    """
    Syncs arbitrary files between local and fabric environments.
    ``@resources`` decorator can be applied to the functions that are wrapped by the ``@dataset`` or ``@model`` decorators.

    :param path: File or directory path, could be relative or absolute. If path is a directory, all directories and files are uploaded recursively.
    :param paths: Additional paths.
    :return: Function object being decorated.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset, resources

        @dataset("titanic")
        @resources("data/titanic.csv")
        def titanic():
            return pd.read_csv("data/titanic.csv")

    When you execute ``layer.run([titanic])``, the following happens:
    - Function ``titanic`` is submitted to Layer fabric, ``data/titanic.csv`` is uploaded to internal resource storage.
    - Before the execution, fabric runtime downloads the resource ``data/titanic.csv`` and makes it available to use within the function scope.

    If you have multiple resources, separate them with commas.

    .. code-block:: python

        @resources("path", "path1", "path2")

    **Note:** while absolute or relative paths are allowed as parameters to ``@resources``, they should always be relative when used within function body. For example:

    .. code-block:: python

        @dataset("titanic")
        @resources("/data/titanic.csv")  # it's ok to use absolute paths as they're resolved on the local machine
        def titanic():
            return pd.read_csv("data/titanic.csv")  # while run inside function fabric, always use the relative paths!

    #### Limitations:
    - You can upload a maximum of 1000 files
    - Total upload size is limited to 5GB

    """

    clean_paths = []

    for raw_path in [path, *paths]:
        if isinstance(raw_path, pathlib.Path):
            clean_paths.append(raw_path)
        elif isinstance(raw_path, str):
            clean_paths.append(pathlib.Path(raw_path))
        else:
            raise ValueError("resource paths must be a string or a pathlib.Path")

    @wrapt.decorator(proxy=_resources_wrapper(clean_paths))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _resources_wrapper(paths: Sequence[pathlib.Path]) -> Any:
    class ResourcesFunctionWrapper(LayerFunctionWrapper, ABC):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(wrapped, wrapper, enabled)

            self.layer.set_resource_paths([ResourcePath(path=p) for p in paths])

    return ResourcesFunctionWrapper
