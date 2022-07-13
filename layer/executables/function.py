from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from layer.contracts.asset import AssetType
from layer.executables.packager import package_function


FunctionOutput = Union["DatasetOutput", "ModelOutput"]


class Function:
    def __init__(
        self,
        func: Callable[..., Any],
        output: FunctionOutput,
        pip_dependencies: List[str],
        resources: List[Path],
    ) -> None:
        self._func = func
        self._output = output
        self._pip_dependencies = pip_dependencies
        self._resources = resources

    @staticmethod
    def from_decorated(func: Callable[..., Any]) -> "Function":
        output = _get_function_output(func)
        pip_dependencies = _get_function_pip_dependencies(func)
        resources = _get_function_resources(func)
        return Function(
            func, output=output, pip_dependencies=pip_dependencies, resources=resources
        )

    @property
    def func(self) -> Callable[..., Any]:
        return self._func

    @property
    def output(self) -> FunctionOutput:
        return self._output

    @property
    def pip_dependencies(self) -> List[str]:
        return self._pip_dependencies

    @property
    def resources(self) -> List[Path]:
        return self._resources

    def package(self, output_dir: Optional[Path] = None) -> Path:
        return package_function(
            self._func,
            pip_dependencies=self._pip_dependencies,
            resources=self._resources,
            output_dir=output_dir,
        )


def _get_function_output(func: Callable[..., Any]) -> FunctionOutput:
    asset_type = _get_decorator_attr(func, "asset_type")
    asset_name = _get_decorator_attr(func, "asset_name")
    if asset_type is None or asset_name is None:
        raise FunctionError(
            'either @dataset(name="...") or @model(name="...") top level decorator '
            "is required for each function. Add @dataset or @model decorator on top of existing "
            "decorators to run functions in Layer"
        )
    if asset_type == AssetType.DATASET:
        return DatasetOutput(asset_name)
    if asset_type == AssetType.MODEL:
        return ModelOutput(asset_name)

    raise FunctionError(f"unsupported asset type: '{asset_type}'")


def _get_function_pip_dependencies(func: Callable[..., Any]) -> List[str]:
    pip_packages = _get_decorator_attr(func, "pip_packages") or []
    requirements = _get_decorator_attr(func, "pip_requirements_file")
    if requirements is not None and len(requirements) > 0:
        with open(requirements, "r") as f:
            pip_packages += f.read().splitlines()
    return pip_packages


def _get_function_resources(func: Callable[..., Any]) -> List[Path]:
    resource_paths = _get_decorator_attr(func, "resource_paths") or []
    return [Path(resource_path.path) for resource_path in resource_paths]


def _get_decorator_attr(func: Callable[..., Any], attr: str) -> Optional[Any]:
    if hasattr(func, "layer") and hasattr(func.layer, attr):  # type: ignore
        return getattr(func.layer, attr)  # type: ignore
    return None


@dataclass(frozen=True)
class DatasetOutput:
    name: str


@dataclass(frozen=True)
class ModelOutput:
    name: str


class FunctionError(Exception):
    pass
