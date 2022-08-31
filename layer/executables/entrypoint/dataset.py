import logging
import uuid
from typing import Any, List

from layerapi.api.entity.task_pb2 import Task
from layerapi.api.ids_pb2 import RunId
from yarl import URL

from layer.clients.layer import LayerClient
from layer.context import Context
from layer.contracts.assertions import Assertion
from layer.contracts.datasets import DatasetBuild
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.exceptions.exceptions import (
    LayerClientException,
    LayerClientServiceUnavailableException,
    LayerServiceUnavailableExceptionDuringInitialization,
    ProjectInitializationException,
)
from layer.global_context import set_has_shown_update_message
from layer.logged_data.logged_data_destination import LoggedDataDestination
from layer.logged_data.system_metrics import SystemMetrics
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
from layer.tracker.progress_tracker import RunProgressTracker
from layer.utils.runtime_utils import check_and_convert_to_df

from .common import make_runner


logger = logging.getLogger(__name__)
set_has_shown_update_message(True)


def _run(
    url: URL,
    dataset_definition: FunctionDefinition,
    client: LayerClient,
    tracker: RunProgressTracker,
    fabric: Fabric,
    run_id: str,
    logged_data_destination: LoggedDataDestination,
    **kwargs: Any,
) -> None:
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

    if run_id:
        client.flow_manager.update_run_metadata(
            run_id=RunId(value=run_id),
            task_id=dataset_definition.asset_path.path(),
            task_type=Task.Type.TYPE_DATASET_BUILD,
            key="build-id",
            value=str(dataset_build_id),
        )

    with Context(
        url=url,
        client=client,
        asset_path=dataset_definition.asset_path,
        dataset_build=DatasetBuild(id=dataset_build_id),
        tracker=tracker,
        logged_data_destination=logged_data_destination,
    ) as ctx:
        try:
            with SystemMetrics(logger):
                result = dataset_definition.func(
                    *dataset_definition.args, **dataset_definition.kwargs
                )

            # in case client does not return the dataset
            if result is None:
                tracker.mark_dataset_built(dataset_definition.asset_name)
                client.data_catalog.complete_build(
                    dataset_build_id,
                    dataset_definition.asset_name,
                    dataset_definition.uri,
                )
                return

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

    ctx.save_dataset(result)

    tracker.mark_dataset_built(
        dataset_definition.asset_name,
        warnings=logged_data_destination.close_and_get_errors(),
    )

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
