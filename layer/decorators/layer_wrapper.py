from typing import Any, Dict, Tuple, Union

import wrapt  # type: ignore


# See https://wrapt.readthedocs.io/en/latest/wrappers.html#custom-function-wrappers for more.
class LayerFunctionWrapper(wrapt.FunctionWrapper):
    def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
        super().__init__(wrapped, wrapper, enabled)

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
