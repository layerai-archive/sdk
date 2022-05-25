import typing
import uuid
from typing import Dict, List, Tuple

from layerapi.api.entity.history_event_pb2 import HistoryEvent
from layerapi.api.entity.run_metadata_pb2 import RunMetadata
from layerapi.api.entity.run_pb2 import Run as PBRun
from layerapi.api.entity.task_pb2 import Task as PBTask
from layerapi.api.ids_pb2 import ModelTrainId, ModelVersionId

from layer.clients.layer import LayerClient
from layer.contracts.projects import ApplyResult
from layer.contracts.runs import Run
from layer.exceptions.exceptions import (
    LayerClientTimeoutException,
    ProjectDatasetBuildExecutionException,
    ProjectModelExecutionException,
    ProjectRunnerError,
    ProjectRunTerminatedError,
)
from layer.exceptions.status_report import ExecutionStatusReportFactory
from layer.projects.utils import get_current_project_name
from layer.tracker.project_progress_tracker import RunProgressTracker
from layer.utils.session import is_layer_debug_on


class PollingStepFunction:
    def __init__(self, max_backoff_sec: float, backoff_multiplier: float):
        self._max_backoff = max_backoff_sec
        self._backoff_multiplier = backoff_multiplier

    def step(self, step: float) -> float:
        return min(self._max_backoff, step * self._backoff_multiplier)


_FormattedRunMetadata = Dict[Tuple["PBTask.Type.V", str, str], str]


class ProgressTrackerUpdater:
    tracker: RunProgressTracker
    client: LayerClient
    apply_metadata: ApplyResult
    run: Run
    run_metadata: _FormattedRunMetadata

    def __init__(
        self,
        tracker: RunProgressTracker,
        apply_metadata: ApplyResult,
        run: Run,
        client: LayerClient,
    ):
        self.tracker = tracker
        self.apply_metadata = apply_metadata
        self.run = run
        self.client = client

    @staticmethod
    def _format_run_metadata(
        run_metadata: RunMetadata,
    ) -> _FormattedRunMetadata:
        return {
            (entry.task_type, entry.task_id, entry.key): entry.value
            for entry in list(run_metadata.entries)
        }

    def check_completion_and_update_tracker(
        self,
        response: typing.Union[
            Tuple[List[HistoryEvent], RunMetadata],
            LayerClientTimeoutException,
        ],
    ) -> bool:
        if isinstance(response, LayerClientTimeoutException):
            return False

        (events, run_metadata) = response
        self.run_metadata = self._format_run_metadata(run_metadata)

        for event in events:
            event_type = event.WhichOneof("event")

            if event_type == "run":
                run_status = event.run.run_status
                if run_status == PBRun.STATUS_TERMINATED:
                    raise ProjectRunTerminatedError(run_id=self.run.run_id)

                elif run_status == PBRun.STATUS_FAILED:
                    raise ProjectRunnerError("Run failed", self.run.run_id)

                elif run_status in [PBRun.STATUS_SUCCEEDED]:
                    return True

                elif run_status == PBRun.STATUS_INVALID:
                    # TODO: alert for this
                    _print_debug("Run status INVALID")

            if event_type == "task":
                self._update_tracker(event.task)

        return False

    def _update_tracker(self, task: PBTask) -> None:
        task_status = task.status
        if task_status == PBTask.STATUS_SCHEDULED:
            self._handle_task_scheduled(task)
        elif task_status == PBTask.STATUS_SUCCEEDED:
            self._handle_task_succeeded(task)
        elif task_status == PBTask.STATUS_FAILED:
            self._handle_task_failed(task)
        elif task_status == PBTask.STATUS_INVALID:
            # TODO: alert for this
            _print_debug("Task status INVALID")

    def _handle_task_succeeded(self, task: PBTask) -> None:
        task_id = task.id
        task_type = task.type
        if task_type == PBTask.TYPE_DATASET_BUILD:
            dataset_name = task_id
            dataset_path = f"{get_current_project_name()}/datasets/{dataset_name}"
            dataset_build_id = uuid.UUID(
                self.run_metadata[(task_type, dataset_path, "build-id")]
            )
            dataset = self.client.data_catalog.get_dataset_by_build_id(dataset_build_id)
            self.tracker.mark_derived_dataset_built(
                name=dataset_name,
                version=dataset.version,
                build_index=dataset.build.index,
            )
        elif task_type == PBTask.TYPE_MODEL_TRAIN:
            train_id = uuid.UUID(self.run_metadata[(task_type, task_id, "train-id")])
            model_name = self._find_model_name_by_version_id(task_id)
            train_index = self.client.model_catalog.get_model_train(
                ModelTrainId(value=str(train_id))
            ).index
            version_name = self.client.model_catalog.get_model_version(
                ModelVersionId(value=str(task_id))
            ).name
            self.tracker.mark_model_trained(
                name=model_name,
                train_index=str(train_index),
                version=version_name,
            )
        else:
            # TODO: alert for this
            _print_debug(f"Task type not handled {task_type}")

    def _handle_task_failed(self, task: PBTask) -> None:
        task_id = task.id
        task_type = task.type
        task_info = task.info
        if task_type == PBTask.TYPE_DATASET_BUILD:
            exc_ds = ProjectDatasetBuildExecutionException(
                self.run.run_id,
                task_id,
                ExecutionStatusReportFactory.from_json(task_info),
            )
            self.tracker.mark_derived_dataset_failed(
                name=task_id, reason=exc_ds.message
            )
            raise exc_ds
        elif task_type == PBTask.TYPE_MODEL_TRAIN:
            exc_model = ProjectModelExecutionException(
                self.run.run_id,
                task_id,
                ExecutionStatusReportFactory.from_json(task_info),
            )
            self.tracker.mark_model_train_failed(
                name=self._find_model_name_by_version_id(task_id),
                reason=f"{exc_model.message}",
            )
            raise exc_model
        else:
            # TODO: alert for this
            _print_debug(f"Task type not handled {task_type}")

    def _handle_task_scheduled(self, task: PBTask) -> None:
        task_id = task.id
        task_type = task.type
        if task_type == PBTask.TYPE_DATASET_BUILD:
            self.tracker.mark_derived_dataset_building(name=task_id)
        elif task_type == PBTask.TYPE_MODEL_TRAIN:
            self.tracker.mark_model_training(
                name=self._find_model_name_by_version_id(task_id)
            )
        else:
            # TODO: alert for this
            _print_debug(f"Task type not handled {task_type}")

    def _find_model_name_by_version_id(self, version_id: str) -> str:
        for definition in self.run.definitions:
            if str(definition.version_id) == version_id:
                return definition.name
        raise KeyError(version_id)


def _print_debug(msg: str) -> None:
    if is_layer_debug_on():
        print(msg)
