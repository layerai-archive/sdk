from abc import ABCMeta
from typing import Any, Callable, Dict, List

import wrapt  # type: ignore

from layer.decorators.layer_wrapper import LayerFunctionWrapper


def fabric(name: str) -> Callable[..., Any]:
    """
    Allows specifying the resources the Layer backend should use to run decorated function.

    Note: The fabric decorator should only ever be used alongside Layer `dataset` or `model` decorators. GPU fabrics are available for model training only and can only be used in combination with `model` decorator.

    This decorator overrides the fabric set via `layer.init`. It has no impact on local execution of your functions, only on remote execution in the Layer backend.

    :param name: `Name of the fabric </docs/reference/fabrics#predefined-fabrics>`_ to use in Layer backend.
    :return: Function object.

    .. code-block:: python

        import layer
        from layer.decorators import dataset, fabric

        # fabric below will determine resources used to run `create_my_dataset` function in Layer backend
        @fabric("f-medium")
        @dataset("product-data")
        def create_product_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return dataframe

        # fabric setting will be used when running your function below
        layer.run(create_product_dataset)
    """

    @wrapt.decorator(proxy=_fabric_wrapper(name))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        return wrapped(*args, **kwargs)

    return wrapper


def _fabric_wrapper(fabric_name: str) -> Any:
    class FabricFunctionWrapper(LayerFunctionWrapper, metaclass=ABCMeta):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any = None) -> None:
            super().__init__(wrapped, wrapper, enabled)
            self.__wrapped__.layer.set_fabric(fabric_name)

    return FabricFunctionWrapper
