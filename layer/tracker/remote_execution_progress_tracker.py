import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple, Union

from rich.progress import TaskID
from rich.text import Text
from yarl import URL

from layer.config import Config
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.runs import Run
from layer.contracts.tracker import (
    EntityTracker,
    EntityTrackerStatus,
    ResourceTransferState,
)
from layer.exceptions.exceptions import ProjectBaseException, ProjectRunnerError

from .output import get_progress_ui
from .progress_tracker import RunProgressTracker


class RemoteExecutionRunProgressTracker(RunProgressTracker):
    def __init__(self, config: Config, run: Run) -> None:
        self._config = config
        self._run = run
        self._task_ids: Dict[Tuple[AssetType, str], TaskID] = {}
        self._progress = get_progress_ui()

    @contextmanager
    def track(self) -> Iterator["RemoteExecutionRunProgressTracker"]:
        with self._progress:
            self._init_tasks()
            yield self

    def _init_tasks(self) -> None:
        for definition in self._run.definitions:
            task_id = self._progress.add_task(
                EntityTrackerStatus.PENDING,
                start=False,
                entity=EntityTracker(
                    type=definition.asset_type,
                    name=definition.name,
                    status=EntityTrackerStatus.PENDING,
                ),
            )
            self._task_ids[(definition.asset_type, definition.name)] = task_id

    def _update_entity(
        self,
        type_: AssetType,
        name: str,
        *,
        status: Optional[EntityTrackerStatus] = None,
        url: Optional[URL] = None,
        version: Optional[str] = None,
        build_idx: Optional[str] = None,
        error_reason: str = "",
        description: Optional[str] = None,
        state: Optional[ResourceTransferState] = None,
    ) -> None:
        task_id = self._task_ids[(type_, name)]
        # noinspection PyProtectedMember
        task = self._progress._tasks[task_id]  # pylint: disable=protected-access
        entity = task.fields["entity"]

        task_description = description if description is not None else status
        if status:
            entity.status = status
        if url:
            entity.base_url = url
        if version:
            entity.version = version
        if build_idx:
            entity.build_idx = build_idx
        if error_reason:
            entity.error_reason = error_reason
        if state:
            entity.resource_transfer_state = state

        self._progress.update(task_id, description=task_description)

        if status == EntityTrackerStatus.PENDING:
            self._progress.stop_task(task_id)
            task.start_time = task.stop_time = None
        elif status in (EntityTrackerStatus.BUILDING, EntityTrackerStatus.TRAINING):
            if not task.started:
                task.start_time = task.stop_time = None
                self._progress.start_task(task_id)
                assert task.total is not None
                self._progress.update(task_id, completed=task.total * 0.1)
        elif status == EntityTrackerStatus.DONE:
            self._progress.update(task_id, completed=task.total)
        elif status == EntityTrackerStatus.ERROR:
            self._progress.stop_task(task_id)
            self._progress.update(task_id, completed=0)

    def _get_entity(self, type_: AssetType, name: str) -> EntityTracker:
        task_id = self._task_ids[(type_, name)]
        # noinspection PyProtectedMember
        task = self._progress._tasks[task_id]  # pylint: disable=protected-access
        entity = task.fields["entity"]
        return entity

    def _get_url(self, asset_type: AssetType, name: str) -> URL:
        assert self._run.account
        return AssetPath(
            entity_name=name,
            asset_type=asset_type,
            org_name=self._run.account.name,
            project_name=self._run.project_name,
        ).url(self._config.url)

    def mark_error_messages(
        self, exc: Union[ProjectBaseException, ProjectRunnerError]
    ) -> None:
        self._progress.stop()  # Stops the progress of non-failed tasks
        self._progress.print(
            Text.from_markup(":x:").append(
                Text(
                    " Project execution error.",
                    style="bold red",
                )
            )
        )
        self._progress.print(
            Text.from_markup(":mag_right:").append(
                Text.from_markup(
                    f" {exc.error_msg_rich if isinstance(exc, ProjectBaseException) else str(exc)}"
                )
            )
        )
        if isinstance(exc, ProjectBaseException):
            self._progress.print(
                Text.from_markup(":bulb:").append(
                    Text.from_markup(
                        f" {exc.suggestion_rich}",
                    )
                )
            )

    def mark_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        status = EntityTrackerStatus.PENDING
        self._update_entity(
            AssetType.DATASET,
            name,
            url=self._get_url(AssetType.DATASET, name),
            status=status,
        )

    def mark_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the dataset is successfully built and
        # it causes a flicker in the progress bar. This is a workaround.
        entity = self._get_entity(AssetType.DATASET, name)
        if entity.status == EntityTrackerStatus.DONE:
            return

        self._update_entity(
            AssetType.DATASET,
            name,
            status=EntityTrackerStatus.BUILDING,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_idx,
        )

    def mark_dataset_failed(self, name: str, reason: str) -> None:
        self._update_entity(
            AssetType.DATASET,
            name,
            status=EntityTrackerStatus.ERROR,
            error_reason=f"{reason}",
        )

    def mark_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        self._update_entity(
            AssetType.DATASET,
            name,
            status=EntityTrackerStatus.DONE,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_index,
        )

    def mark_model_saving(self, name: str) -> None:
        self._update_entity(AssetType.MODEL, name, status=EntityTrackerStatus.SAVING)

    def mark_model_saved(self, name: str) -> None:
        status = EntityTrackerStatus.PENDING
        self._update_entity(
            AssetType.MODEL,
            name,
            status=status,
        )

    def mark_model_training(
        self, name: str, version: Optional[str] = None, train_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the model is successfully trained and
        # it causes a flicker in the progress bar. This is a workaround.
        entity = self._get_entity(AssetType.MODEL, name)
        if entity.status == EntityTrackerStatus.DONE:
            return

        self._update_entity(
            AssetType.MODEL,
            name,
            status=EntityTrackerStatus.TRAINING,
            url=self._get_url(AssetType.MODEL, name),
            version=version,
            build_idx=train_idx,
        )

    def mark_model_trained(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        train_index: Optional[str] = None,
    ) -> None:
        self._update_entity(
            AssetType.MODEL,
            name,
            status=EntityTrackerStatus.DONE,
            url=self._get_url(AssetType.MODEL, name),
            version=version,
            build_idx=train_index,
        )

    def mark_model_train_failed(self, name: str, reason: str) -> None:
        self._update_entity(
            AssetType.MODEL,
            name,
            status=EntityTrackerStatus.ERROR,
            error_reason=f"{reason}",
        )

    def mark_model_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        self._update_entity(
            AssetType.MODEL,
            name,
            description="uploading",
            state=state,
            status=EntityTrackerStatus.RESOURCE_UPLOADING,
        )

    def mark_model_resources_uploaded(self, name: str) -> None:
        self._update_entity(
            AssetType.MODEL,
            name,
            description="uploaded",
            status=EntityTrackerStatus.PENDING,
        )

    def mark_dataset_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        self._update_entity(
            AssetType.DATASET,
            name,
            description="uploading",
            state=state,
            status=EntityTrackerStatus.RESOURCE_UPLOADING,
        )

    def mark_dataset_resources_uploaded(self, name: str) -> None:
        self._update_entity(
            AssetType.DATASET,
            name,
            description="uploaded",
            status=EntityTrackerStatus.PENDING,  # Pending here meaning that training has not started yet
        )
