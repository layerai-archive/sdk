from typing import Any, List, Optional, Union

from layer import Dataset, Model
from layer.contracts.asset import AssetType
from layer.contracts.fabrics import Fabric
from layer.exceptions.exceptions import ConfigError
from layer.global_context import (
    default_fabric,
    get_pip_packages,
    get_pip_requirements_file,
)


def _resolve_settings(
    override: Optional[Any], default: Optional[Any], fallback: Optional[Any]
) -> Any:
    setting = override if override else default
    return setting if setting else fallback


class LayerSettings:
    _asset_type: Optional[AssetType] = None
    _name: Optional[str] = None
    _fabric: Optional[Fabric] = None
    _pip_requirements_file: Optional[str] = None
    _pip_packages: Optional[List[str]] = None
    _paths: Optional[List[str]] = None
    _dependencies: Optional[List[Union[Dataset, Model]]] = None
    _assertions: Optional[List[Any]] = None

    def get_fabric(self) -> Optional[Fabric]:
        return _resolve_settings(self._fabric, default_fabric(), None)

    def get_pip_packages(self) -> List[str]:
        return _resolve_settings(self._pip_packages, get_pip_packages(), [])

    def get_pip_requirements_file(self) -> str:
        return _resolve_settings(
            self._pip_requirements_file, get_pip_requirements_file(), ""
        )

    def get_paths(self) -> Optional[List[str]]:
        return self._paths

    def get_entity_name(self) -> Optional[str]:
        return self._name

    def get_asset_type(self) -> Optional[AssetType]:
        return self._asset_type

    def get_dependencies(self) -> List[Union[Dataset, Model]]:
        return self._dependencies if self._dependencies else []

    def get_assertions(self) -> List[Any]:
        return self._assertions if self._assertions else []

    def set_fabric(self, f: str) -> None:
        if Fabric.has_member_key(f):
            self._fabric = Fabric(f)
            return
        raise ValueError(
            'Fabric setting "{}" is not valid. You can check valid values in Fabric enum definition.'.format(
                f
            )
        )

    def set_pip_requirements_file(self, file: Optional[str]) -> None:
        self._pip_requirements_file = file

    def set_pip_packages(self, packages: Optional[List[str]]) -> None:
        self._pip_packages = packages

    def set_paths(self, paths: Optional[List[str]]) -> None:
        self._paths = paths

    def set_asset_type(self, asset_type: AssetType) -> None:
        self._asset_type = asset_type

    def set_entity_name(self, name: str) -> None:
        self._name = name

    def set_dependencies(self, dependencies: List[Union[Dataset, Model]]) -> None:
        self._dependencies = dependencies

    def append_assertions(self, assertions: List[Any]) -> None:
        existing_assertions = self.get_assertions()
        existing_assertions.append(assertions)
        self._assertions = existing_assertions

    def validate(self) -> None:
        if self.get_asset_type() is None:
            raise ConfigError(
                'Either @dataset(name="...") or @model(name="...") top level decorator '
                "is required for each function. Add @dataset or @model decorator on top of existing "
                "decorators to run functions in Layer."
            )
        if self.get_entity_name() is None or self.get_entity_name() == "":
            raise ConfigError(
                "Your @dataset and @model must be named. Pass an entity name as a first argument to your decorators."
            )
        fabric = self.get_fabric()
        if (
            fabric is not None
            and fabric.is_gpu()
            and self.get_asset_type() is AssetType.DATASET
        ):
            raise ConfigError(
                "GPU fabrics can only be used for model training. Use a different fabric for your dataset build."
            )
