import logging
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

import wrapt  # type: ignore

from layer import Dataset, Model
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.contracts.asset import AssetType
from layer.decorators.layer_wrapper import LayerFunctionWrapper
from layer.decorators.utils import ensure_has_layer_settings
from layer.definitions import ModelDefinition
from layer.projects.util import (
    get_current_project_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.tracker.local_execution_project_progress_tracker import (
    LocalExecutionProjectProgressTracker,
)
from layer.training.runtime.model_train_failure_reporter import (
    ModelTrainFailureReporter,
)
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)


def model(
    name: str, dependencies: Optional[List[Union[Dataset, Model]]] = None
) -> Callable[..., Any]:
    """
    Decorator function used to wrap another function that trains a model. The function that decorator has been applied to needs to return a ML model object from a supported framework.

    The decorator ensures that:
    - If decorated function is executed locally, then the model trained is stored in the Layer backend.
    - If decorated function is passed in to `layer.run(your_function)`, then Layer runs the function remotely and stores the model in Layer backend.

    :param name: Name with which the model are stored in Layer's backend.
    :param dependencies: List of datasets or models that will be built by Layer backend prior to building the current function. Layer understand what entities this function depends on so it optimizes the build process accordingly.
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

        @model(name='survival_model', dependencies=[layer.Dataset('layer/titanic/datasets/features')])
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

    @wrapt.decorator(proxy=_model_wrapper(name, dependencies))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _model_wrapper(
    name: str,
    dependencies: Optional[List[Union[Dataset, Model]]] = None,
) -> Any:
    class FunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            self.__wrapped__.layer.set_asset_type(AssetType.MODEL)
            self.__wrapped__.layer.set_entity_name(name)
            self.__wrapped__.layer.set_dependencies(dependencies)

        # This is not serialized with cloudpickle, so it will only be run locally.
        # See https://layerco.slack.com/archives/C02R5B3R3GU/p1646144705414089 for detail.
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            current_project_name_ = get_current_project_name()
            config = asyncio_run_in_thread(ConfigManager().refresh())
            with LayerClient(config.client, logger).init() as client:
                account_name = client.account.get_my_account().name
                project_progress_tracker = LocalExecutionProjectProgressTracker(
                    project_name=current_project_name_,
                    config=config,
                    account_name=account_name,
                )

                with project_progress_tracker.track() as tracker:
                    tracker.add_model(self.__wrapped__.layer.get_entity_name())
                    model_definition = ModelDefinition(
                        self.__wrapped__, current_project_name_
                    )
                    return self._train_model_locally_and_store_remotely(
                        model_definition, tracker, client
                    )

        def _train_model_locally_and_store_remotely(
            self,
            model_definition: ModelDefinition,
            tracker: LocalExecutionProjectProgressTracker,
            client: LayerClient,
        ) -> Any:
            from layer.training.runtime.model_trainer import (
                LocalTrainContext,
                ModelTrainer,
            )

            model = model_definition.get_local_entity()
            assert model.project_name is not None
            verify_project_exists_and_retrieve_project_id(client, model.project_name)

            model_version = client.model_catalog.create_model_version(
                model.project_name, model
            ).model_version
            train_id = client.model_catalog.create_model_train_from_version_id(
                model_version.id
            )
            train = client.model_catalog.get_model_train(train_id)

            context = LocalTrainContext(  # noqa: F841
                logger=logger,
                model_name=model.name,
                model_version=model_version.name,
                train_id=UUID(train_id.value),
                source_folder=model.local_path.parent,
                source_entrypoint=model.training.entrypoint,
                train_index=str(train.index),
            )
            failure_reporter = ModelTrainFailureReporter(
                client.model_catalog,
                logger,
                context.train_id,
                context.source_folder,
            )
            trainer = ModelTrainer(
                client=client,
                train_context=context,
                logger=logger,
                failure_reporter=failure_reporter,
                tracker=tracker,
            )
            return trainer.train()

    return FunctionWrapper
