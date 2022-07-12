import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

import wrapt  # type: ignore
from layerapi.api.ids_pb2 import ProjectId

from layer import Dataset, Model
from layer.clients.layer import LayerClient
from layer.config import ConfigManager, is_executables_feature_active
from layer.config.config import Config
from layer.context import Context
from layer.contracts.assertions import Assertion
from layer.contracts.assets import AssetType
from layer.contracts.datasets import DatasetBuild, DatasetBuildStatus
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.tracker import DatasetTransferState
from layer.decorators.layer_wrapper import LayerAssetFunctionWrapper
from layer.global_context import reset_active_context, set_active_context
from layer.projects.project_runner import register_function
from layer.projects.utils import (
    get_current_project_full_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.settings import LayerSettings
from layer.tracker.progress_tracker import RunProgressTracker
from layer.tracker.utils import get_progress_tracker
from layer.utils.async_utils import asyncio_run_in_thread
from layer.utils.runtime_utils import check_and_convert_to_df


logger = logging.getLogger(__name__)


def dataset(
    name: str, dependencies: Optional[List[Union[str, Dataset, Model]]] = None
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
    name: str, dependencies: Optional[List[Union[str, Dataset, Model]]] = None
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
            )

        # This is not serialized with cloudpickle, so it will only be run locally.
        # See https://layerco.slack.com/archives/C02R5B3R3GU/p1646144705414089 for detail.
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if is_executables_feature_active():
                # execute the function, metadata capture will be done by the runtime
                return self.__wrapped__(*args, **kwargs)
            self.layer.validate()
            dataset_definition = self.get_definition()
            dataset_definition.package()
            config: Config = asyncio_run_in_thread(ConfigManager().refresh())
            current_project_full_name_ = get_current_project_full_name()
            with LayerClient(config.client, logger).init() as client:
                progress_tracker = get_progress_tracker(
                    url=config.url,
                    project_name=current_project_full_name_.project_name,
                    account_name=current_project_full_name_.account_name,
                )
                with progress_tracker.track() as tracker:
                    tracker.add_asset(AssetType.DATASET, self.layer.get_asset_name())
                    result = _build_dataset_locally_and_store_remotely(
                        lambda: dataset_definition.func(*args, **kwargs),
                        self.layer,
                        dataset_definition,
                        tracker,
                        client,
                    )
                    return result

    return DatasetFunctionWrapper


def _build_dataset_locally_and_store_remotely(
    building_func: Callable[..., Any],
    layer: LayerSettings,
    dataset: FunctionDefinition,
    tracker: RunProgressTracker,
    client: LayerClient,
) -> Any:
    tracker.add_asset(AssetType.DATASET, layer.get_asset_name())

    register_function(client, func=dataset, tracker=tracker)
    tracker.mark_dataset_building(layer.get_asset_name())

    (result, build_uuid) = _build_locally_update_remotely(
        client,
        building_func,
        dataset,
        tracker,
    )

    transfer_state = DatasetTransferState(len(result))
    tracker.mark_dataset_saving_result(dataset.asset_name, transfer_state)

    # this call would store the resulting dataset, extract the schema and complete the build from remote
    client.data_catalog.store_dataset(
        data=result,
        build_id=build_uuid,
        progress_callback=transfer_state.increment_num_transferred_rows,
    )
    tracker.mark_dataset_built(dataset.asset_name)
    return result


def _build_locally_update_remotely(
    client: LayerClient,
    function_that_builds_dataset: Callable[..., Any],
    dataset: FunctionDefinition,
    tracker: RunProgressTracker,
) -> Tuple[Any, UUID]:
    try:
        with Context() as context:
            # TODO pass path to API instead
            current_project_uuid = verify_project_exists_and_retrieve_project_id(
                client, dataset.project_full_name
            )
            initiate_build_response = client.data_catalog.initiate_build(
                ProjectId(value=str(current_project_uuid)),
                dataset.asset_name,
                dataset.get_fabric(True),
            )
            dataset_build_id = UUID(initiate_build_response.id.value)
            context.with_dataset_build(
                DatasetBuild(id=dataset_build_id, status=DatasetBuildStatus.STARTED)
            )
            context.with_tracker(tracker)
            context.with_asset_name(dataset.asset_name)
            set_active_context(context)
            try:
                result = function_that_builds_dataset()
                result = check_and_convert_to_df(result)
                _run_assertions(dataset.asset_name, result, dataset.assertions, tracker)
            except Exception as e:
                client.data_catalog.complete_build(
                    initiate_build_response.id, dataset.asset_name, dataset.uri, e
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
    asset_name: str,
    result: Any,
    assertions: List[Assertion],
    tracker: RunProgressTracker,
) -> None:
    failed_assertions = []
    tracker.mark_dataset_running_assertions(asset_name)
    for assertion in reversed(assertions):
        try:
            tracker.mark_dataset_running_assertion(asset_name, assertion)
            assertion.function(result)
        except Exception:
            failed_assertions.append(assertion)
    if len(failed_assertions) > 0:
        tracker.mark_dataset_failed_assertions(asset_name, failed_assertions)
        raise Exception(f"Failed assertions {failed_assertions}\n")
    else:
        tracker.mark_dataset_completed_assertions(asset_name)
