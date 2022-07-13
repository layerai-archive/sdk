import logging
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import UUID

import wrapt  # type: ignore

from layer import Dataset, Model
from layer.clients.layer import LayerClient
from layer.config import ConfigManager, is_executables_feature_active
from layer.config.config import Config
from layer.contracts.assets import AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.decorators.layer_wrapper import LayerAssetFunctionWrapper
from layer.projects.utils import (
    get_current_project_full_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.tracker.progress_tracker import RunProgressTracker
from layer.tracker.utils import get_progress_tracker
from layer.training.runtime.model_train_failure_reporter import (
    ModelTrainFailureReporter,
)
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)


def model(
    name: str, dependencies: Optional[List[Union[str, Dataset, Model]]] = None
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
    dependencies: Optional[List[Union[str, Dataset, Model]]] = None,
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
            )

        # This is not serialized with cloudpickle, so it will only be run locally.
        # See https://layerco.slack.com/archives/C02R5B3R3GU/p1646144705414089 for detail.
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if is_executables_feature_active():
                # execute the function, metadata capture will be done by the runtime
                return self.__wrapped__(*args, **kwargs)
            model_definition = self.get_definition()
            model_definition.package()
            config: Config = asyncio_run_in_thread(ConfigManager().refresh())
            current_project_full_name_ = get_current_project_full_name()
            with LayerClient(config.client, logger).init() as client:
                progress_tracker = get_progress_tracker(
                    url=config.url,
                    project_name=current_project_full_name_.project_name,
                    account_name=current_project_full_name_.account_name,
                )

                with progress_tracker.track() as tracker:
                    tracker.add_asset(AssetType.MODEL, self.layer.get_asset_name())
                    return self._train_model_locally_and_store_remotely(
                        model_definition, tracker, client
                    )

        @staticmethod
        def _train_model_locally_and_store_remotely(
            model: FunctionDefinition,
            tracker: RunProgressTracker,
            client: LayerClient,
        ) -> Any:
            from layer.training.runtime.model_trainer import (
                LocalTrainContext,
                ModelTrainer,
            )

            assert model.project_name is not None
            verify_project_exists_and_retrieve_project_id(
                client, model.project_full_name
            )

            model_version = client.model_catalog.create_model_version(
                model.asset_path,
                model.description,
                model.source_code_digest.hexdigest(),
                model.get_fabric(True),
            ).model_version
            train_id = client.model_catalog.create_model_train_from_version_id(
                model_version.id
            )
            train = client.model_catalog.get_model_train(train_id)

            context = LocalTrainContext(  # noqa: F841
                logger=logger,
                model_name=model.asset_name,
                model_version=model_version.name,
                train_id=UUID(train_id.value),
                source_folder=model.function_home_dir,
                source_entrypoint=model.entrypoint,
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
            result = trainer.train()
            return result

    return FunctionWrapper
