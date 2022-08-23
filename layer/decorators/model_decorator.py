import logging
from typing import Any, Callable, Dict, List, Optional, Union

import wrapt  # type: ignore

from layer import Dataset, Model
from layer.contracts.asset import AssetType
from layer.decorators.layer_wrapper import LayerAssetFunctionWrapper


logger = logging.getLogger(__name__)


def model(
    name: str,
    dependencies: Optional[List[Union[str, Dataset, Model]]] = None,
    description: Optional[str] = None,
) -> Callable[..., Any]:
    """
    Decorator function used to wrap another function that trains a model. The function that decorator has been applied to needs to return a ML model object from a supported framework.

    The decorator ensures that:
    - If decorated function is executed locally, then the model trained is stored in the Layer backend.
    - If decorated function is passed in to `layer.run(your_function)`, then Layer runs the function remotely and stores the model in Layer backend.

    :param name: Name with which the model are stored in Layer's backend.
    :param dependencies: List of datasets or models that will be built by Layer backend prior to building the current function. Layer understand what entities this function depends on so it optimizes the build process accordingly.
    :param description: Optional description to be displayed in the UI.
    :return: Function object being decorated.

    .. code-block:: python

        from layer
        from layer.decorators import model

        # Define a new function for model generation

        @model("model_1")
        def create_model_1():
            from sklearn import datasets
            from sklearn.svm import SVC

            iris = datasets.load_iris()
            clf = SVC()
            return clf.fit(iris.data, iris.target)

        # below will train model using Layer's backend
        layer.run([create_model_1])

    If you use a dataset that is outside of your project, then you must explicitly call it out as a dependency.
    These dependencies are displayed in the **External assets** section in your Layer project.

    The dataset used in the model example below is from the a public Layer project.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset, model

        @model(name='survival_model', dependencies=[layer.Dataset('layer/titanic/datasets/features')], description="Some description")
        @assert_true(test_survival_probability)
        def train():
            df = layer.get_dataset("layer/titanic/datasets/features").to_pandas()
            X = df.drop(["Survived"], axis=1)
            y = df["Survived"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            random_forest = RandomForestClassifier(n_estimators=100)
            random_forest.fit(X_train, y_train)
            y_pred = random_forest.predict(X_test)
            return random_forest
    """

    @wrapt.decorator(proxy=_model_wrapper(name, dependencies, description))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _model_wrapper(
    name: str,
    dependencies: Optional[List[Union[str, Dataset, Model]]] = None,
    description: Optional[str] = None,
) -> Any:
    class FunctionWrapper(LayerAssetFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(
                wrapped,
                wrapper,
                enabled,
                AssetType.MODEL,
                name,
                dependencies,
                description,
            )

    return FunctionWrapper
