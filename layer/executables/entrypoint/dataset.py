import logging
import uuid
from typing import Any, List

from layerapi.api.entity.task_pb2 import Task
from layerapi.api.ids_pb2 import RunId

from layer.clients.layer import LayerClient
from layer.config.config import Config
from layer.context import Context, reset_active_context, set_active_context
from layer.contracts.assertions import Assertion
from layer.contracts.assets import AssetType
from layer.contracts.datasets import DatasetBuild, DatasetBuildStatus
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.contracts.tracker import DatasetTransferState
from layer.exceptions.exceptions import (
    LayerClientException,
    LayerClientServiceUnavailableException,
    LayerServiceUnavailableExceptionDuringInitialization,
    ProjectInitializationException,
)
from layer.global_context import set_has_shown_update_message
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
from layer.tracker.progress_tracker import RunProgressTracker
from layer.tracker.utils import get_progress_tracker
from layer.utils.runtime_utils import check_and_convert_to_df

from .common import make_runner


logger = logging.getLogger(__name__)
set_has_shown_update_message(True)


def _run(
    dataset_definition: FunctionDefinition, config: Config, fabric: Fabric, run_id: str
) -> None:

    with LayerClient(config.client, logger).init() as client:

        progress_tracker = get_progress_tracker(
            url=config.url,
            project_name=dataset_definition.project_name,
            account_name=dataset_definition.account_name,
        )

        with progress_tracker.track() as tracker:
            tracker.add_asset(AssetType.DATASET, dataset_definition.asset_name)

            _register_function(
                client, dataset=dataset_definition, tracker=tracker, fabric=fabric
            )
            tracker.mark_dataset_building(dataset_definition.asset_name)
            # TODO pass path to API instead
            current_project_uuid = verify_project_exists_and_retrieve_project_id(
                client, dataset_definition.project_full_name
            )
            dataset_build_id = client.data_catalog.initiate_build(
                current_project_uuid,
                dataset_definition.asset_name,
                fabric.value,
            )

            try:
                with Context(
                    asset_type=AssetType.DATASET,
                    asset_name=dataset_definition.asset_name,
                    dataset_build=DatasetBuild(
                        id=dataset_build_id, status=DatasetBuildStatus.STARTED
                    ),
                    tracker=tracker,
                ) as context:
                    set_active_context(context)
                    if run_id:
                        client.flow_manager.update_run_metadata(
                            run_id=RunId(value=run_id),
                            task_id=dataset_definition.asset_path.path(),
                            task_type=Task.Type.TYPE_DATASET_BUILD,
                            key="build-id",
                            value=str(dataset_build_id),
                        )
                    try:
                        result = dataset_definition.func(
                            *dataset_definition.args, **dataset_definition.kwargs
                        )
                        result = check_and_convert_to_df(result)
                        _run_assertions(
                            dataset_definition.asset_name,
                            result,
                            dataset_definition.assertions,
                            tracker,
                        )
                    except Exception as e:
                        client.data_catalog.complete_build(
                            dataset_build_id,
                            dataset_definition.asset_name,
                            dataset_definition.uri,
                            e,
                        )
                        raise e
                    reset_active_context()
            finally:
                reset_active_context()

            transfer_state = DatasetTransferState(len(result))
            tracker.mark_dataset_saving_result(
                dataset_definition.asset_name, transfer_state
            )

            # this call would store the resulting dataset, extract the schema and complete the build from remote
            client.data_catalog.store_dataset(
                data=result,
                build_id=dataset_build_id,
                progress_callback=transfer_state.increment_num_transferred_rows,
            )
            tracker.mark_dataset_built(dataset_definition.asset_name)

            return result


def _register_function(
    client: LayerClient,
    dataset: FunctionDefinition,
    tracker: RunProgressTracker,
    fabric: Fabric,
) -> None:
    try:
        project_id = verify_project_exists_and_retrieve_project_id(
            client, dataset.project_full_name
        )
        dataset_id = client.data_catalog.add_dataset(
            dataset.asset_path,
            project_id,
            dataset.description,
            fabric.value,
            dataset.func_source,
            dataset.entrypoint,
            dataset.environment,
        )
        dataset.set_repository_id(uuid.UUID(dataset_id))
        assert dataset.repository_id
        tracker.mark_dataset_saved(dataset.asset_name, id_=dataset.repository_id)
    except LayerClientServiceUnavailableException as e:
        tracker.mark_dataset_failed(dataset.asset_name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_dataset_failed(dataset.asset_name, "")
        raise ProjectInitializationException(
            f"Failed to save derived dataset {dataset.asset_name!r}: {e}",
            "Please retry",
        )


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


RUNNER = make_runner(_run)
