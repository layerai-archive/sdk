from typing import Any, Dict, List, Optional, Tuple, Union

import wrapt  # type: ignore

from layer.contracts.assets import AssetPath, AssetType, BaseAsset
from layer.contracts.datasets import Dataset
from layer.contracts.models import Model
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import FunctionDefinition
from layer.projects.utils import get_current_project_full_name
from layer.settings import LayerSettings


# See https://wrapt.readthedocs.io/en/latest/wrappers.html#custom-function-wrappers for more.
class LayerFunctionWrapper(wrapt.FunctionWrapper):
    def __init__(
        self,
        wrapped: Any,
        wrapper: Any,
        enabled: Any,
    ) -> None:
        super().__init__(wrapped, wrapper, enabled)
        if not hasattr(wrapped, "layer"):
            wrapped.layer = LayerSettings()
        self.layer: LayerSettings = wrapped.layer

    # wrapt doesn't implement this method and based on this https://github.com/GrahamDumpleton/wrapt/issues/102#issuecomment-899937490
    # we give it a shot and it seems to be working
    def __reduce_ex__(self, protocol: Any) -> Any:
        return type(self), (self.__wrapped__, self._self_wrapper, self._self_enabled)

    def __copy__(self) -> None:
        pass

    def __deepcopy__(self, memo: Dict[int, object]) -> None:
        pass

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        pass


class LayerAssetFunctionWrapper(LayerFunctionWrapper):
    def __init__(
        self,
        wrapped: Any,
        wrapper: Any,
        enabled: Any,
        asset_type: AssetType,
        name: str,
        dependencies: Optional[List[Union[str, Dataset, Model]]],
    ) -> None:
        super().__init__(wrapped, wrapper, enabled)
        self.layer.set_asset_type(asset_type)
        self.layer.set_asset_name(name)

        paths: List[AssetPath] = []
        if dependencies is not None:
            for dependency in dependencies:
                if isinstance(dependency, str):
                    paths.append(AssetPath.parse(dependency))
                elif isinstance(dependency, BaseAsset):
                    paths.append(dependency._path)
                else:
                    raise ValueError(
                        "Dependencies can only be a string, Dataset or Model."
                    )
        self.layer.set_dependencies(paths)

    def get_definition(self) -> FunctionDefinition:
        self.layer.validate()
        current_project_full_name = get_current_project_full_name()

        # get asset dependencies
        asset_dependencies: List[AssetPath] = []
        for d in self.layer.get_dependencies():
            full_path_dep = d
            if d.is_relative():
                if d.project_name is not None:
                    full_path_dep = d.with_project_full_name(
                        ProjectFullName(
                            account_name=current_project_full_name.account_name,
                            project_name=d.project_name,
                        )
                    )
                else:
                    full_path_dep = d.with_project_full_name(current_project_full_name)
            asset_dependencies.append(full_path_dep)

        # get pip dependencies
        pip_dependencies: List[str] = []
        if self.layer.get_pip_requirements_file():
            with open(self.layer.get_pip_requirements_file(), "r") as file:
                pip_dependencies = file.read().strip().split("\n")
        else:
            pip_dependencies = self.layer.get_pip_packages()

        return FunctionDefinition(
            func=self.__wrapped__,
            project_name=current_project_full_name.project_name,
            account_name=current_project_full_name.account_name,
            asset_type=self.layer.get_asset_type(),
            asset_name=self.layer.get_asset_name(),
            fabric=self.layer.get_fabric(),
            asset_dependencies=asset_dependencies,
            pip_dependencies=pip_dependencies,
            resource_paths=self.layer.get_resource_paths(),
            assertions=self.layer.get_assertions(),
        )
