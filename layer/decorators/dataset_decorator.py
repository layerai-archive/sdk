import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

import wrapt  # type: ignore

from layer import Dataset, Model
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.context import Context
from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetType
from layer.contracts.datasets import DatasetBuild, DatasetBuildStatus
from layer.contracts.runs import DatasetTransferState
from layer.decorators.assertions import get_assertion_functions_data
from layer.decorators.layer_wrapper import LayerFunctionWrapper
from layer.decorators.utils import ensure_has_layer_settings
from layer.definitions import DatasetDefinition
from layer.global_context import reset_active_context, set_active_context
from layer.projects.project_runner import register_derived_datasets
from layer.projects.util import (
    get_current_project_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.settings import LayerSettings
from layer.tracker.local_execution_project_progress_tracker import (
    LocalExecutionProjectProgressTracker,
)
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)


def dataset(
    name: str, dependencies: Optional[List[Union[Dataset, Model]]] = None
) -> Callable[..., Any]:
    """
    Decorator function that wraps a dataset function. The wrapped function must output a Pandas dataframe.

    The decorator ensures that:
    - If decorated function is passed in to ``layer.run(your_function)``, then Layer runs the function remotely and stores its output as a dataset.
    - If you run the function locally, ``your_function()``, then Layer stores the output in the Layer backend as a dataset. This does not affect function execution.

    :param name: Name with which the dataset will be stored in Layer backend.
    :param dependencies: List of ``Datasets`` or ``Models`` that will be built by Layer backend prior to building the current function. This hints Layer what entities this function depends on and optimizes the build process.
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
    As long as the function outputs a Pandas dataframe, Layer doesn't really care how you get the data into it.

    .. code-block:: python

        import pandas as pd
        from layer.decorators import dataset

        @dataset("my_products")
        def create_product_dataset():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return dataframe

        product_dataset = create_product_dataset()


    If you use a dataset that is outside of your project, then you must explicitly call it out as a dependency.
    These dependencies are displayed in the **External assets** section in your Layer project.

    .. code-block:: python

        import pandas as pd
        from layer
        from layer.decorators import dataset

        @dataset("raw_spam_dataset", dependencies=[layer.Dataset('layer/spam-detection/datasets/spam_messages')])
        def raw_spam_dataset():
            # Get the spam_messages dataset and convert to Pandas dataframe.
            df = layer.get_dataset("spam-detection/datasets/spam_messages").to_pandas()
            return df
    """

    @wrapt.decorator(proxy=_dataset_wrapper(name, dependencies))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _dataset_wrapper(
    name: str, dependencies: Optional[List[Union[Dataset, Model]]] = None
) -> Any:
    class DatasetFunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            self.__wrapped__.layer.set_asset_type(AssetType.DATASET)
            self.__wrapped__.layer.set_entity_name(name)
            self.__wrapped__.layer.set_dependencies(dependencies)

        # This is not serialized with cloudpickle, so it will only be run locally.
        # See https://layerco.slack.com/archives/C02R5B3R3GU/p1646144705414089 for detail.
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.__wrapped__.layer.validate()
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
                    dataset_definition = DatasetDefinition(
                        self.__wrapped__, current_project_name_
                    )
                    result = _build_dataset_locally_and_store_remotely(
                        lambda: super(  # pylint: disable=super-with-arguments
                            DatasetFunctionWrapper, self
                        ).__call__(*args, **kwargs),
                        self.layer,
                        dataset_definition,
                        tracker,
                        client,
                        get_assertion_functions_data(self),
                    )
                    return result

    return DatasetFunctionWrapper


def _build_dataset_locally_and_store_remotely(
    building_func: Callable[..., Any],
    layer: LayerSettings,
    dataset_definition: DatasetDefinition,
    tracker: LocalExecutionProjectProgressTracker,
    client: LayerClient,
    assertions: List[Assertion],
) -> Any:
    tracker.add_build(layer.get_entity_name())  # type: ignore

    current_project_uuid = verify_project_exists_and_retrieve_project_id(
        client, get_current_project_name()
    )

    derived_dataset = dataset_definition.get_local_entity()
    derived_dataset_updated = register_derived_datasets(
        client, current_project_uuid, derived_dataset, tracker
    )
    tracker.mark_derived_dataset_building(layer.get_entity_name())  # type: ignore

    (result, build_uuid) = _build_locally_update_remotely(
        client,
        building_func,
        derived_dataset_updated,
        current_project_uuid,
        tracker,
        assertions,
    )

    transfer_state = DatasetTransferState(len(result))
    tracker.mark_dataset_saving_result(derived_dataset.name, transfer_state)

    # this call would store the resulting dataset, extract the schema and complete the build from remote
    client.data_catalog.store_dataset(
        name="",
        data=result,
        build_id=build_uuid,
        progress_callback=transfer_state.increment_num_transferred_rows,
    )
    tracker.mark_derived_dataset_built(derived_dataset.name)
    return result


def _build_locally_update_remotely(
    client: LayerClient,
    function_that_builds_dataset: Callable[..., Any],
    derived_dataset_updated: Dataset,
    current_project_uuid: UUID,
    tracker: LocalExecutionProjectProgressTracker,
    assertions: List[Assertion],
) -> Tuple[Any, UUID]:
    try:
        with Context() as context:
            initiate_build_response = client.data_catalog.initiate_build(
                derived_dataset_updated, current_project_uuid
            )
            dataset_build_id = UUID(initiate_build_response.id.value)
            context.with_dataset_build(
                DatasetBuild(id=dataset_build_id, status=DatasetBuildStatus.STARTED)
            )
            context.with_tracker(tracker)
            context.with_entity_name(derived_dataset_updated.name)
            set_active_context(context)
            try:
                result = function_that_builds_dataset()
                _run_assertions(
                    derived_dataset_updated.name, result, assertions, tracker
                )
            except Exception as e:
                client.data_catalog.complete_build(
                    initiate_build_response.id, derived_dataset_updated, e
                )
                context.with_dataset_build(
                    DatasetBuild(id=dataset_build_id, status=DatasetBuildStatus.FAILED)
                )
                raise e
            reset_active_context()

            context.with_dataset_build(
                DatasetBuild(id=dataset_build_id, status=DatasetBuildStatus.COMPLETED)
            )
            return result, UUID(str(initiate_build_response.id.value))
    finally:
        reset_active_context()


def _run_assertions(
    entity_name: str,
    result: Any,
    assertions: List[Assertion],
    tracker: LocalExecutionProjectProgressTracker,
) -> None:
    failed_assertions = []
    tracker.mark_dataset_running_assertions(entity_name)
    for assertion in reversed(assertions):
        try:
            tracker.mark_dataset_running_assertion(entity_name, assertion)
            assertion.function(result)
        except Exception:
            failed_assertions.append(assertion)
    if len(failed_assertions) > 0:
        tracker.mark_dataset_failed_assertions(entity_name, failed_assertions)
        raise Exception(f"Failed assertions {failed_assertions}\n")
    else:
        tracker.mark_dataset_completed_assertions(entity_name)
