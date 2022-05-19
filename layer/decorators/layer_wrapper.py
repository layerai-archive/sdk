from typing import Any, Dict, List, Optional, Tuple, Union

import wrapt  # type: ignore

from layer.contracts.asset import AssetPath, AssetType, BaseAsset
from layer.contracts.datasets import Dataset
from layer.contracts.models import Model
from layer.settings import LayerSettings


def ensure_has_layer_settings(wrapped: Any) -> None:
    if not hasattr(wrapped, "layer"):
        wrapped.layer = LayerSettings()


# See https://wrapt.readthedocs.io/en/latest/wrappers.html#custom-function-wrappers for more.
class LayerFunctionWrapper(wrapt.FunctionWrapper):
    def __init__(
        self,
        wrapped: Any,
        wrapper: Any,
        enabled: Any,
    ) -> None:
        super().__init__(wrapped, wrapper, enabled)
        ensure_has_layer_settings(self.__wrapped__)

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
        ensure_has_layer_settings(self.__wrapped__)
        self.__wrapped__.layer.set_asset_type(asset_type)
        self.__wrapped__.layer.set_entity_name(name)

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
        self.__wrapped__.layer.set_dependencies(paths)
