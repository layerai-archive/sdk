import logging
import os
import pickle  # nosec import_pickle
from typing import Any, List
from uuid import UUID

from layerapi.api.ids_pb2 import ProjectId

import layer
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.context import Context
from layer.contracts.assertions import Assertion
from layer.contracts.assets import AssetType
from layer.contracts.datasets import DatasetBuild, DatasetBuildStatus
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.tracker import DatasetTransferState
from layer.global_context import (
    reset_active_context,
    set_active_context,
    set_has_shown_update_message,
)
from layer.projects.project_runner import register_dataset_function
from layer.projects.utils import (
    get_current_project_full_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.settings import LayerSettings
from layer.tracker.progress_tracker import RunProgressTracker
from layer.utils.async_utils import asyncio_run_in_thread
from layer.utils.runtime_utils import check_and_convert_to_df


logger = logging.getLogger(__name__)
set_has_shown_update_message(True)


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


def _run(user_function: Any) -> None:
    settings: LayerSettings = user_function.layer
    current_project_full_name_ = get_current_project_full_name()
    config: Config = asyncio_run_in_thread(ConfigManager().refresh())
    with LayerClient(config.client, logger).init() as client:
        progress_tracker = RunProgressTracker(
            url=config.url,
            project_name=current_project_full_name_.project_name,
            account_name=current_project_full_name_.account_name,
        )

        with progress_tracker.track() as tracker:
            tracker.add_asset(AssetType.DATASET, settings.get_asset_name())
            dataset = FunctionDefinition(
                func=user_function,
                project_name=current_project_full_name_.project_name,
                account_name=current_project_full_name_.account_name,
                asset_name=settings.get_asset_name(),
                asset_type=AssetType.DATASET,
                fabric=settings.get_fabric(),
                asset_dependencies=[],
                pip_dependencies=[],
                resource_paths=settings.get_resource_paths(),
                assertions=settings.get_assertions(),
            )

            register_dataset_function(client, dataset, True, tracker)
            tracker.mark_dataset_building(settings.get_asset_name())

            try:
                with Context() as context:
                    # TODO pass path to API instead
                    current_project_uuid = (
                        verify_project_exists_and_retrieve_project_id(
                            client, dataset.project_full_name
                        )
                    )
                    initiate_build_response = client.data_catalog.initiate_build(
                        ProjectId(value=str(current_project_uuid)),
                        dataset.asset_name,
                        dataset.get_fabric(True),
                    )
                    dataset_build_id = UUID(initiate_build_response.id.value)
                    context.with_dataset_build(
                        DatasetBuild(
                            id=dataset_build_id, status=DatasetBuildStatus.STARTED
                        )
                    )
                    context.with_tracker(tracker)
                    context.with_asset_name(dataset.asset_name)
                    set_active_context(context)
                    try:
                        result = dataset.func()
                        result = check_and_convert_to_df(result)
                        _run_assertions(
                            dataset.asset_name, result, dataset.assertions, tracker
                        )
                    except Exception as e:
                        client.data_catalog.complete_build(
                            initiate_build_response.id,
                            dataset.asset_name,
                            dataset.uri,
                            e,
                        )
                        context.with_dataset_build(
                            DatasetBuild(
                                id=dataset_build_id, status=DatasetBuildStatus.FAILED
                            )
                        )
                        raise e
                    reset_active_context()

                    context.with_dataset_build(
                        DatasetBuild(
                            id=dataset_build_id, status=DatasetBuildStatus.COMPLETED
                        )
                    )
                    build_uuid = UUID(str(initiate_build_response.id.value))
            finally:
                reset_active_context()

            transfer_state = DatasetTransferState(len(result))
            tracker.mark_dataset_saving_result(dataset.asset_name, transfer_state)

            # this call would store the resulting dataset, extract the schema and complete the build from remote
            client.data_catalog.store_dataset(
                data=result,
                build_id=build_uuid,
                progress_callback=transfer_state.increment_num_transferred_rows,
            )
            tracker.mark_dataset_built(dataset.asset_name)


# load the entrypoint function
with open("function.pkl", "rb") as file:
    layer.init(os.environ["LAYER_PROJECT_NAME"])
    user_function = pickle.load(file)  # nosec pickle

_run(user_function)
