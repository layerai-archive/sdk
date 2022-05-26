import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple, Union

from rich.progress import TaskID
from rich.text import Text
from yarl import URL

from layer.config import Config
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.entities import Entity, EntityStatus, EntityType
from layer.contracts.runs import ResourceTransferState, Run
from layer.exceptions.exceptions import ProjectBaseException, ProjectRunnerError
from layer.tracker.output import get_progress_ui
from layer.tracker.project_progress_tracker import RunProgressTracker


class RemoteExecutionRunProgressTracker(RunProgressTracker):
    def __init__(self, config: Config, run: Run) -> None:
        self._config = config
        self._run = run
        self._task_ids: Dict[Tuple[EntityType, str], TaskID] = {}
        self._progress = get_progress_ui()

    @contextmanager
    def track(self) -> Iterator["RemoteExecutionRunProgressTracker"]:
        with self._progress:
            self._init_tasks()
            yield self

    def _init_tasks(self) -> None:
        for definition in self._run.definitions:
            entity_type: Optional[EntityType] = None
            if definition.asset_type == AssetType.DATASET:
                entity_type = EntityType.DERIVED_DATASET
            elif definition.asset_type == AssetType.MODEL:
                entity_type = EntityType.MODEL
            else:
                continue

            task_id = self._progress.add_task(
                EntityStatus.PENDING,
                start=False,
                entity=Entity(
                    type=entity_type,
                    name=definition.name,
                    status=EntityStatus.PENDING,
                ),
            )
            self._task_ids[(entity_type, definition.name)] = task_id

    def _update_entity(
        self,
        type_: EntityType,
        name: str,
        *,
        status: Optional[EntityStatus] = None,
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

        if status == EntityStatus.PENDING:
            self._progress.stop_task(task_id)
            task.start_time = task.stop_time = None
        elif status in (EntityStatus.BUILDING, EntityStatus.TRAINING):
            if not task.started:
                task.start_time = task.stop_time = None
                self._progress.start_task(task_id)
                assert task.total is not None
                self._progress.update(task_id, completed=task.total * 0.1)
        elif status == EntityStatus.DONE:
            self._progress.update(task_id, completed=task.total)
        elif status == EntityStatus.ERROR:
            self._progress.stop_task(task_id)
            self._progress.update(task_id, completed=0)

    def _get_entity(self, type_: EntityType, name: str) -> Entity:
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

    def mark_derived_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        status = EntityStatus.PENDING
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            url=self._get_url(AssetType.DATASET, name),
            status=status,
        )

    def mark_derived_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the dataset is successfully built and
        # it causes a flicker in the progress bar. This is a workaround.
        entity = self._get_entity(EntityType.DERIVED_DATASET, name)
        if entity.status == EntityStatus.DONE:
            return

        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.BUILDING,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_idx,
        )

    def mark_derived_dataset_failed(self, name: str, reason: str) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.ERROR,
            error_reason=f"{reason}",
        )

    def mark_derived_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.DONE,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_index,
        )

    def mark_model_saving(self, name: str) -> None:
        self._update_entity(EntityType.MODEL, name, status=EntityStatus.SAVING)

    def mark_model_saved(self, name: str) -> None:
        status = EntityStatus.PENDING
        self._update_entity(
            EntityType.MODEL,
            name,
            status=status,
        )

    def mark_model_training(
        self, name: str, version: Optional[str] = None, train_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the model is successfully trained and
        # it causes a flicker in the progress bar. This is a workaround.
        entity = self._get_entity(EntityType.MODEL, name)
        if entity.status == EntityStatus.DONE:
            return

        self._update_entity(
            EntityType.MODEL,
            name,
            status=EntityStatus.TRAINING,
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
            EntityType.MODEL,
            name,
            status=EntityStatus.DONE,
            url=self._get_url(AssetType.MODEL, name),
            version=version,
            build_idx=train_index,
        )

    def mark_model_train_failed(self, name: str, reason: str) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            status=EntityStatus.ERROR,
            error_reason=f"{reason}",
        )

    def mark_model_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            description="uploading",
            state=state,
            status=EntityStatus.RESOURCE_UPLOADING,
        )

    def mark_model_resources_uploaded(self, name: str) -> None:
        self._update_entity(
            EntityType.MODEL, name, description="uploaded", status=EntityStatus.PENDING
        )

    def mark_dataset_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            description="uploading",
            state=state,
            status=EntityStatus.RESOURCE_UPLOADING,
        )

    def mark_dataset_resources_uploaded(self, name: str) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            description="uploaded",
            status=EntityStatus.PENDING,  # Pending here meaning that training has not started yet
        )
