import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

from rich.console import Console
from rich.progress import TaskID
from yarl import URL

from layer.assertion_utils import Assertion
from layer.config import Config
from layer.projects.entity import Entity, EntityStatus, EntityType
from layer.projects.tracker.dataset_transfer_state import DatasetTransferState
from layer.projects.tracker.output import get_progress_ui
from layer.projects.tracker.project_progress_tracker import ProjectProgressTracker
from layer.projects.tracker.resource_transfer_state import ResourceTransferState


class LocalExecutionProjectProgressTracker(ProjectProgressTracker):
    def __init__(
        self,
        project_name: str,
        config: Config,
        account_name: str,
    ):
        self._project_name = project_name
        self._account_name = account_name
        self._config = config
        self._console = Console()
        self._progress = get_progress_ui()
        self._task_ids: Dict[Tuple[EntityType, str], TaskID] = {}

    @contextmanager
    def track(self) -> Iterator["LocalExecutionProjectProgressTracker"]:
        """
        Initializes tracking. Meant to be used with a `with` construct.
        """
        with self._progress:
            yield self

    def add_build(self, name: str) -> None:
        self._get_or_create_task(EntityType.DERIVED_DATASET, name)

    def add_model(self, name: str) -> None:
        self._get_or_create_task(EntityType.MODEL, name)

    def _get_or_create_task(self, entity_type: EntityType, name: str) -> TaskID:
        if (entity_type, name) not in self._task_ids:
            task_id = self._progress.add_task(
                start=False,
                entity=Entity(type=entity_type, name=name, status=EntityStatus.PENDING),
                description="pending",
            )
            self._task_ids[(entity_type, name)] = task_id
        return self._task_ids[(entity_type, name)]

    def _update_entity(
        self,
        type_: EntityType,
        name: str,
        *,
        status: Optional[EntityStatus] = None,
        cur_step: int = 0,
        total_steps: int = 0,
        url: Optional[URL] = None,
        error_reason: str = "",
        description: Optional[str] = None,
        model_transfer_state: Optional[ResourceTransferState] = None,
        dataset_transfer_state: Optional[DatasetTransferState] = None,
    ) -> None:
        task_id = self._get_or_create_task(type_, name)
        # noinspection PyProtectedMember
        task = self._progress._tasks[task_id]  # pylint: disable=protected-access
        entity = task.fields["entity"]
        if status:
            entity.status = status
        if url:
            entity.url = url
        if error_reason:
            entity.error_reason = error_reason
        if dataset_transfer_state:
            entity.dataset_transfer_state = dataset_transfer_state
        if model_transfer_state:
            entity.model_transfer_state = model_transfer_state

        if description is not None:
            self._progress.update(
                task_id,
                description=description,
            )

        if status == EntityStatus.PENDING:
            self._progress.update(task_id, description="pending")
        elif status == EntityStatus.SAVING:
            if type_ == EntityType.DERIVED_DATASET:
                # Even if we go through all steps, we still want to keep track of time elapsed until the status is DONE, so we add +1 to total_steps.
                self._progress.update(
                    task_id,
                    completed=cur_step,
                    total=(total_steps + 1),
                    description=f"saved {cur_step}/{total_steps} rows",
                )
            elif type_ == EntityType.MODEL:
                self._progress.update(
                    task_id,
                    description="saving",
                )
        elif status == EntityStatus.BUILDING:
            if not task.started:
                self._progress.start_task(task_id)
            self._progress.update(
                task_id, completed=task.total * 0.1, description="building"
            )
        elif status == EntityStatus.TRAINING:
            if not task.started:
                self._progress.start_task(task_id)
            self._progress.update(
                task_id, completed=task.total * 0.1, description="training"
            )
        elif status == EntityStatus.DONE:
            self._progress.update(task_id, completed=task.total, description="done")
            self._progress.stop_task(task_id)
        elif status == EntityStatus.ERROR:
            self._progress.stop_task(task_id)
            self._progress.update(task_id, completed=0, description="error")

    def mark_derived_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
            name=name,
            id=id_,
        )
        self._update_entity(EntityType.DERIVED_DATASET, name, url=url)

    def mark_derived_dataset_building(self, name: str) -> None:
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
            name=name,
        )
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.BUILDING,
            url=url,
        )

    def mark_derived_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
            name=name,
            version=version,
            build_index=build_index,
        )
        self._update_entity(
            EntityType.DERIVED_DATASET, name, status=EntityStatus.DONE, url=url
        )

    def mark_model_saving(self, name: str) -> None:
        self._update_entity(EntityType.MODEL, name, status=EntityStatus.SAVING)

    def mark_model_saved(self, name: str) -> None:
        self._update_entity(EntityType.MODEL, name, status=EntityStatus.DONE)

    def mark_model_training(self, name: str) -> None:
        url = EntityType.MODEL.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
            name=name,
        )
        self._update_entity(
            EntityType.MODEL, name, status=EntityStatus.TRAINING, url=url
        )

    def mark_model_trained(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        train_index: Optional[str] = None,
    ) -> None:
        url = EntityType.MODEL.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
            name=name,
            version=version,
            train_index=train_index,
        )
        self._update_entity(EntityType.MODEL, name, url=url, status=EntityStatus.SAVING)

    def mark_model_train_failed(self, name: str, reason: str) -> None:
        self._update_entity(EntityType.MODEL, name, status=EntityStatus.ERROR)

    def update_derived_dataset_saving_progress(
        self, name: str, cur_step: int, total_steps: int
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.SAVING,
            cur_step=cur_step,
            total_steps=total_steps,
        )

    def mark_model_running_assertions(self, name: str) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            status=EntityStatus.ASSERTING,
            description="asserting...",
        )

    def mark_model_running_assertion(self, name: str, assertion: Assertion) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            description=str(assertion),
        )

    def mark_model_completed_assertions(self, name: str) -> None:
        self._update_entity(EntityType.MODEL, name, description="asserted")

    def mark_model_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        stringified = [str(assertion) for assertion in assertions]
        error_msg = f"failed: {', '.join(stringified)}"
        self._update_entity(
            EntityType.MODEL, name, status=EntityStatus.ERROR, error_reason=error_msg
        )

    def mark_dataset_running_assertions(self, name: str) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.ASSERTING,
            description="asserting...",
        )

    def mark_dataset_running_assertion(self, name: str, assertion: Assertion) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            description=str(assertion),
        )

    def mark_dataset_completed_assertions(self, name: str) -> None:
        self._update_entity(EntityType.DERIVED_DATASET, name, description="asserted")

    def mark_dataset_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        stringified = [str(assertion) for assertion in assertions]
        error_msg = f"failed: {', '.join(stringified)}"
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            status=EntityStatus.ERROR,
            error_reason=error_msg,
        )

    def mark_dataset_saving_result(
        self, name: str, state: DatasetTransferState
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            dataset_transfer_state=state,
            status=EntityStatus.RESULT_UPLOADING,
        )

    def mark_model_saving_result(self, name: str, state: ResourceTransferState) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            model_transfer_state=state,
            status=EntityStatus.RESULT_UPLOADING,
        )
