from typing import Any, Callable, Dict, List, Optional

import wrapt  # type: ignore

from layer.decorators.layer_wrapper import LayerFunctionWrapper


def pip_requirements(
    file: Optional[str] = None, packages: Optional[List[str]] = None
) -> Callable[..., Any]:
    """
    Allows specifying requirements to install in Layer backend when running decorated function.

    Only one of file or packages can be defined. These settings will override requirements set via ``layer.init()``.

    The decorator should only be used alongside Layer ``dataset`` or ``model`` decorators. It will only have effect on the remote execution of your decorated functions. If you require specific packages and you want to run the decorated function locally, you will need to install those requirements locally yourself.

    :param file: Path to file listing requirements to install.
    :param packages: List of packages (and optionally their version) to install in Layer backend when running decorated function.
    :return: Function object.

    .. code-block:: python

        from layer.decorators import dataset, pip_requirements

        layer.init("your-project-name", pip_packages=["tensorflow"])

        # You can specify libraries to install via pip_requirements. Doing so will override
        # requirements specified via layer.init above when running `create_product_dataset`
        @pip_requirements(packages=["pandas==1.4.1"])
        @dataset("product-data")
        def create_product_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return dataframe
    """
    if file and packages:
        raise ValueError("either file or packages should be provided, not both")

    @wrapt.decorator(proxy=_pip_requirements_wrapper(file, packages))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _pip_requirements_wrapper(
    file: Optional[str], packages: Optional[List[str]]
) -> Any:
    class PipRequirementsFunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(wrapped, wrapper, enabled)
            self.layer.set_pip_requirements_file(file)
            self.layer.set_pip_packages(packages)

    return PipRequirementsFunctionWrapper
