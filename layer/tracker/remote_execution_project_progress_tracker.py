import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple, Union

from rich.progress import TaskID
from rich.text import Text
from yarl import URL

from layer.config import Config
from layer.contracts.entities import Entity, EntityStatus, EntityType
from layer.contracts.projects import Project
from layer.contracts.runs import ResourceTransferState
from layer.exceptions.exceptions import ProjectBaseException, ProjectRunnerError
from layer.tracker.output import get_progress_ui
from layer.tracker.project_progress_tracker import ProjectProgressTracker


class RemoteExecutionProjectProgressTracker(ProjectProgressTracker):
    def __init__(self, config: Config, project: Project) -> None:
        self._config = config
        self._project = project
        self._task_ids: Dict[Tuple[EntityType, str], TaskID] = {}
        self._progress = get_progress_ui()

    @contextmanager
    def track(self) -> Iterator["RemoteExecutionProjectProgressTracker"]:
        with self._progress:
            self._init_tasks()
            yield self

    def _init_tasks(self) -> None:
        for raw_dataset in self._project.raw_datasets:
            task_id = self._progress.add_task(
                EntityStatus.PENDING,
                start=False,
                entity=Entity(
                    type=EntityType.RAW_DATASET,
                    name=raw_dataset.name,
                    status=EntityStatus.PENDING,
                ),
            )
            self._task_ids[(EntityType.RAW_DATASET, raw_dataset.name)] = task_id

        for derived_dataset in self._project.derived_datasets:
            task_id = self._progress.add_task(
                EntityStatus.PENDING,
                start=False,
                entity=Entity(
                    type=EntityType.DERIVED_DATASET,
                    name=derived_dataset.name,
                    status=EntityStatus.PENDING,
                ),
            )
            self._task_ids[(EntityType.DERIVED_DATASET, derived_dataset.name)] = task_id

        for model in self._project.models:
            task_id = self._progress.add_task(
                EntityStatus.PENDING,
                start=False,
                entity=Entity(
                    type=EntityType.MODEL, name=model.name, status=EntityStatus.PENDING
                ),
            )
            self._task_ids[(EntityType.MODEL, model.name)] = task_id

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

    def mark_raw_dataset_saved(self, name: str) -> None:
        status = EntityStatus.DONE
        self._update_entity(EntityType.RAW_DATASET, name, status=status)

    def mark_raw_dataset_save_failed(self, name: str, reason: str) -> None:
        self._update_entity(
            EntityType.RAW_DATASET, name, status=EntityStatus.ERROR, error_reason=reason
        )

    def mark_derived_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        status = EntityStatus.PENDING
        assert self._project.account
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project.name,
            self._project.account.name,
            name=name,
            id=id_,
        )
        self._update_entity(EntityType.DERIVED_DATASET, name, url=url, status=status)

    def mark_derived_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the dataset is successfully built and
        # it causes a flicker in the progress bar. This is a workaround.
        entity = self._get_entity(EntityType.DERIVED_DATASET, name)
        if entity.status == EntityStatus.DONE:
            return

        assert self._project.account is not None
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project.name,
            self._project.account.name,
            name=name,
        )

        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.BUILDING,
            url=url,
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
        assert self._project.account is not None
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project.name,
            self._project.account.name,
            name=name,
        )
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.DONE,
            url=url,
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

        assert self._project.account is not None
        url = EntityType.MODEL.get_url(
            self._config.url,
            self._project.name,
            self._project.account.name,
            name=name,
        )
        self._update_entity(
            EntityType.MODEL,
            name,
            status=EntityStatus.TRAINING,
            url=url,
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
        assert self._project.account is not None
        url = EntityType.MODEL.get_url(
            self._config.url,
            self._project.name,
            self._project.account.name,
            name=name,
        )
        self._update_entity(
            EntityType.MODEL,
            name,
            url=url,
            status=EntityStatus.DONE,
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
