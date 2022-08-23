import logging
from typing import Any, Callable, Dict, List, Optional, Union

import wrapt  # type: ignore

from layer import Dataset, Model
from layer.contracts.asset import AssetType
from layer.decorators.layer_wrapper import LayerAssetFunctionWrapper


logger = logging.getLogger(__name__)


def dataset(
    name: str,
    dependencies: Optional[List[Union[str, Dataset, Model]]] = None,
    description: Optional[str] = None,
) -> Callable[..., Any]:
    """
    Decorator function that wraps a dataset function.

    The decorator ensures that:
    - If decorated function is passed in to ``layer.run(your_function)``, then Layer runs the function remotely and stores its output as a dataset.
    - If you run the function locally, ``your_function()``, then Layer stores the output in the Layer backend as a dataset. This does not affect function execution.
    - The result of the function is a pandas.DataFrame or an array of column name -> value dict-s, which under the hood is converted to a pandas.DataFrame

    Supported Pandas dataframe types:
    * `bool`
    * `(u)int{8,16,32,64}`
    * `float32`
    * `float64`
    * `str`
    * `pd.Categorical` (categorical maps to only integer categories, label information is lost during the conversion)
    * `pd.Timestamp`
    * `datetime.date`
    * `datetime.time`
    * `PIL.Image.Image`
    * `numpy.ndarray`

    :param name: Name with which the dataset will be stored in Layer backend.
    :param dependencies: List of ``Datasets`` or ``Models`` that will be built by Layer backend prior to building the current function. This hints Layer what entities this function depends on and optimizes the build process.
    :param description: Optional description to be displayed in the UI.
    :return: Function object being decorated.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset

        # Define a new function for dataset generation:
        # - The dependencies list includes entities that needs to be built before running the `create_my_dataset` code
        # - `titanic` is a publicly accessible dataset to everyone using Layer
        @dataset("my_titanic_dataset", dependencies=Dataset("titanic"))
        def create_my_titanic_dataset():
            df = layer.get_dataset("titanic").to_pandas()
            return df

        # Run function locally
        df = create_my_titanic_dataset()
        # Dataset will be stored in Layer backend and will be retrievable later
        assert df == layer.get_dataset("my_titanic_dataset").to_pandas()

    Here's another way to create a dataset.
    As long as the function outputs a Pandas dataframe or an array of dict-s, Layer doesn't really care how you load the data.

    .. code-block:: python

        import pandas as pd
        from layer.decorators import dataset

        @dataset("my_products")
        def create_product_dataset():
            data_as_dict = [
                {"Id": 1, "Product": "product1", "Price": 15},
                {"Id": 2, "Product": "product2", "Price": 20},
                {"Id": 3, "Product": "product3", "Price": 10}
            ]
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            return pd.DataFrame(data, columns=["Id", "Product", "Price"])
            #  return data_as_dict  # You can also return this

        product_dataset = create_product_dataset()


    If you use a dataset that is outside of your project, then you must explicitly call it out as a dependency.
    These dependencies are displayed in the **External assets** section in your Layer project.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset

        @dataset("raw_spam_dataset", dependencies=[layer.Dataset('layer/spam-detection/datasets/spam_messages')], description="Imporant dataset!")
        def raw_spam_dataset():
            # Get the spam_messages dataset and convert to Pandas dataframe.
            df = layer.get_dataset("spam-detection/datasets/spam_messages").to_pandas()
            return df
    """

    @wrapt.decorator(proxy=_dataset_wrapper(name, dependencies, description))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _dataset_wrapper(
    name: str,
    dependencies: Optional[List[Union[str, Dataset, Model]]] = None,
    description: Optional[str] = None,
) -> Any:
    class DatasetFunctionWrapper(LayerAssetFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(
                wrapped,
                wrapper,
                enabled,
                AssetType.DATASET,
                name,
                dependencies,
                description,
            )

    return DatasetFunctionWrapper
