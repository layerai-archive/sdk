import typing
import uuid
from typing import Dict, List, Tuple

from layerapi.api.entity.history_event_pb2 import HistoryEvent
from layerapi.api.entity.run_metadata_pb2 import RunMetadata
from layerapi.api.entity.run_pb2 import Run as PBRun
from layerapi.api.entity.task_pb2 import Task as PBTask

from layer.clients.layer import LayerClient
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.remote_runs import RemoteRun
from layer.exceptions.exceptions import (
    LayerClientTimeoutException,
    ProjectDatasetBuildExecutionException,
    ProjectModelExecutionException,
    ProjectRunnerError,
    ProjectRunTerminatedError,
)
from layer.exceptions.status_report import ExecutionStatusReportFactory
from layer.tracker.progress_tracker import RunProgressTracker
from layer.utils.session import is_layer_debug_on

from ..contracts.asset import AssetPath, AssetType


class PollingStepFunction:
    def __init__(self, max_backoff_sec: float, backoff_multiplier: float):
        self._max_backoff = max_backoff_sec
        self._backoff_multiplier = backoff_multiplier

    def step(self, step: float) -> float:
        return min(self._max_backoff, step * self._backoff_multiplier)


_FormattedRunMetadata = Dict[Tuple["PBTask.Type.ValueType", str, str], str]


class ProgressTrackerUpdater:
    tracker: RunProgressTracker
    client: LayerClient
    run: RemoteRun
    definitions: List[FunctionDefinition]
    run_metadata: _FormattedRunMetadata

    def __init__(
        self,
        tracker: RunProgressTracker,
        run: RemoteRun,
        definitions: List[FunctionDefinition],
        client: LayerClient,
    ):
        self.tracker = tracker
        self.run = run
        self.definitions = definitions
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
                    raise ProjectRunTerminatedError(run_id=self.run.id)

                elif run_status == PBRun.STATUS_FAILED:
                    raise ProjectRunnerError(
                        f"Run failed: {event.run.info}", self.run.id
                    )

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
        task_name = self._get_name_from_task_id(task_id)
        task_type = task.type
        if task_type == PBTask.TYPE_DATASET_BUILD:
            build_id = self.run_metadata.get((task_type, task_id, "build-id"))
            tag = None
            if build_id:
                build = self.client.data_catalog.get_build_by_id(uuid.UUID(build_id))
                tag = build.tag
            self.tracker.mark_done(
                asset_type=AssetType.DATASET,
                name=task_name,
                tag=tag,
            )
        elif task_type == PBTask.TYPE_MODEL_TRAIN:
            train_id = self.run_metadata.get((task_type, task_id, "train-id"))
            tag = None
            if train_id:
                model_train = self.client.model_catalog.get_model_train(
                    uuid.UUID(train_id)
                )
                tag = model_train.tag
            self.tracker.mark_done(
                asset_type=AssetType.MODEL,
                name=task_name,
                tag=tag,
            )
        else:
            # TODO: alert for this
            _print_debug(f"Task type not handled {task_type}")

    def _handle_task_failed(self, task: PBTask) -> None:
        assert self.run.id
        task_name = self._get_name_from_task_id(task.id)
        task_type = task.type
        task_id = task.id
        task_info = task.info
        if task_type == PBTask.TYPE_DATASET_BUILD:
            build_info = None
            build_id = self.run_metadata.get((task_type, task_id, "build-id"))
            if build_id:
                build_info = self.client.data_catalog.get_build_info_by_build_id(
                    uuid.UUID(build_id)
                )
            status_report = (
                ExecutionStatusReportFactory.from_json(build_info)
                if build_info
                else ExecutionStatusReportFactory.from_plain_text(task_info)
            )
            exc_ds = ProjectDatasetBuildExecutionException(
                self.run.id,
                task_name,
                status_report,
            )
            self.tracker.mark_failed(
                asset_type=AssetType.DATASET, name=task_name, reason=exc_ds.message
            )
            raise exc_ds
        elif task_type == PBTask.TYPE_MODEL_TRAIN:
            train_status_info = None
            train_id = self.run_metadata.get((task_type, task_id, "train-id"))
            if train_id:
                train_status_info = (
                    self.client.model_catalog.get_model_train_status_info(
                        uuid.UUID(train_id)
                    )
                )
            status_report = (
                ExecutionStatusReportFactory.from_json(train_status_info)
                if train_status_info
                else ExecutionStatusReportFactory.from_plain_text(task_info)
            )
            exc_model = ProjectModelExecutionException(
                self.run.id,
                task_name,
                status_report,
            )
            self.tracker.mark_failed(
                asset_type=AssetType.MODEL,
                name=task_name,
                reason=exc_model.message,
            )
            raise exc_model
        else:
            # TODO: alert for this
            _print_debug(f"Task type not handled {task_type}")

    def _handle_task_scheduled(self, task: PBTask) -> None:
        task_name = self._get_name_from_task_id(task.id)
        task_type = task.type
        if task_type == PBTask.TYPE_DATASET_BUILD:
            self.tracker.mark_running(asset_type=AssetType.DATASET, name=task_name)
        elif task_type == PBTask.TYPE_MODEL_TRAIN:
            self.tracker.mark_running(asset_type=AssetType.MODEL, name=task_name)
        else:
            # TODO: alert for this
            _print_debug(f"Task type not handled {task_type}")

    @staticmethod
    def _get_name_from_task_id(task_id: str) -> str:
        return AssetPath.parse(task_id).asset_name


def _print_debug(msg: str) -> None:
    if is_layer_debug_on():
        print(msg)
