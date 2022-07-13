from typing import Any, List, Optional

from layer.contracts.assertions import Assertion
from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.fabrics import Fabric
from layer.contracts.runs import ResourcePath
from layer.exceptions.exceptions import ConfigError, LayerClientException
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
    _resource_paths: Optional[List[ResourcePath]] = None
    _dependencies: Optional[List[AssetPath]] = None
    _assertions: Optional[List[Assertion]] = None

    def get_asset_type(self) -> AssetType:
        if self._asset_type is None:
            raise LayerClientException("Asset type cannot be empty")
        return self._asset_type

    @property
    def asset_type(self) -> AssetType:
        return self.get_asset_type()

    @property
    def asset_name(self) -> str:
        return self.get_asset_name()

    @property
    def pip_packages(self) -> List[str]:
        return self.get_pip_packages()

    @property
    def pip_requirements_file(self) -> str:
        return self.get_pip_requirements_file()

    @property
    def resource_paths(self) -> List[ResourcePath]:
        return self.get_resource_paths()

    def get_asset_name(self) -> str:
        if self._name is None:
            raise LayerClientException("Asset name cannot be empty")
        return self._name

    def get_fabric(self) -> Fabric:
        return _resolve_settings(self._fabric, default_fabric(), Fabric.default())

    def get_pip_requirements_file(self) -> str:
        return _resolve_settings(
            self._pip_requirements_file, get_pip_requirements_file(), ""
        )

    def get_pip_packages(self) -> List[str]:
        return _resolve_settings(self._pip_packages, get_pip_packages(), [])

    def get_resource_paths(self) -> List[ResourcePath]:
        return self._resource_paths or []

    def get_dependencies(self) -> List[AssetPath]:
        return self._dependencies or []

    def get_assertions(self) -> List[Assertion]:
        return self._assertions or []

    def set_asset_type(self, asset_type: AssetType) -> None:
        self._asset_type = asset_type

    def set_asset_name(self, name: str) -> None:
        self._name = name

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

    def set_resource_paths(self, paths: Optional[List[ResourcePath]]) -> None:
        self._resource_paths = paths

    def set_dependencies(self, dependencies: List[AssetPath]) -> None:
        self._dependencies = dependencies

    def append_assertion(self, assertion: Assertion) -> None:
        self._assertions = self._assertions or []
        self._assertions.append(assertion)

    def validate(self) -> None:
        if self._asset_type is None:
            raise ConfigError(
                'Either @dataset(name="...") or @model(name="...") top level decorator '
                "is required for each function. Add @dataset or @model decorator on top of existing "
                "decorators to run functions in Layer."
            )
        if self._name is None or self._name == "":
            raise ConfigError(
                "Your @dataset and @model must be named. Pass an asset name as a first argument to your decorators."
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
