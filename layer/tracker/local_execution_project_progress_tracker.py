import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple, Union

from rich.console import Console
from rich.progress import TaskID
from yarl import URL

from layer.config import Config
from layer.contracts.assertions import Assertion
from layer.contracts.entities import Entity, EntityStatus, EntityType
from layer.contracts.runs import DatasetTransferState, ResourceTransferState
from layer.tracker.output import get_progress_ui
from layer.tracker.project_progress_tracker import ProjectProgressTracker


class LocalExecutionProjectProgressTracker(ProjectProgressTracker):
    def __init__(
        self,
        config: Config,
        account_name: Optional[str] = None,
        project_name: Optional[str] = None,
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

    def _update_entity(  # noqa: C901
        self,
        type_: EntityType,
        name: str,
        *,
        status: Optional[EntityStatus] = None,
        cur_step: int = 0,
        total_steps: int = 0,
        url: Optional[URL] = None,
        version: Optional[str] = None,
        build_idx: Optional[str] = None,
        error_reason: str = "",
        description: Optional[str] = None,
        model_transfer_state: Optional[ResourceTransferState] = None,
        dataset_transfer_state: Optional[DatasetTransferState] = None,
        resource_transfer_state: Optional[ResourceTransferState] = None,
        loading_cache_entity: Optional[str] = None,
        entity_download_transfer_state: Optional[
            Union[ResourceTransferState, DatasetTransferState]
        ] = None,
    ) -> None:
        task_id = self._get_or_create_task(type_, name)
        # noinspection PyProtectedMember
        task = self._progress._tasks[task_id]  # pylint: disable=protected-access
        entity = task.fields["entity"]
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
        if dataset_transfer_state:
            entity.dataset_transfer_state = dataset_transfer_state
        if model_transfer_state:
            entity.model_transfer_state = model_transfer_state
        if resource_transfer_state:
            entity.resource_transfer_state = resource_transfer_state
        if loading_cache_entity:
            entity.loading_cache_entity = loading_cache_entity
        if entity_download_transfer_state:
            entity.entity_download_transfer_state = entity_download_transfer_state

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
            self._progress.update(task_id, description="building")
        elif (
            status == EntityStatus.ENTITY_DOWNLOADING
            or status == EntityStatus.ENTITY_FROM_CACHE
        ):
            if not task.started:
                self._progress.start_task(task_id)
        elif status == EntityStatus.TRAINING:
            if not task.started:
                self._progress.start_task(task_id)
            self._progress.update(task_id, description="training")
        elif status == EntityStatus.DONE:
            self._progress.update(task_id, completed=task.total, description="done")
            self._progress.stop_task(task_id)
        elif status == EntityStatus.ERROR:
            self._progress.stop_task(task_id)
            self._progress.update(task_id, completed=0, description="error")

    def mark_derived_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        assert self._project_name
        assert self._account_name
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
            name=name,
            id=id_,
        )
        self._update_entity(EntityType.DERIVED_DATASET, name, url=url)

    def mark_derived_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        assert self._project_name
        assert self._account_name
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
            version=version,
            build_idx=build_idx,
        )

    def mark_derived_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        assert self._project_name
        assert self._account_name
        url = EntityType.DERIVED_DATASET.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
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
        self._update_entity(EntityType.MODEL, name, status=EntityStatus.DONE)

    def mark_model_training(
        self, name: str, version: Optional[str] = None, train_idx: Optional[str] = None
    ) -> None:
        assert self._project_name
        assert self._account_name
        url = EntityType.MODEL.get_url(
            self._config.url,
            self._project_name,
            self._account_name,
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
        self._update_entity(
            EntityType.MODEL, name, url=None, status=EntityStatus.SAVING
        )

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

    def mark_model_getting_model(
        self,
        name: str,
        getting_entity_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            entity_download_transfer_state=state,
            status=EntityStatus.ENTITY_DOWNLOADING
            if not from_cache
            else EntityStatus.ENTITY_FROM_CACHE,
            loading_cache_entity=None if not from_cache else getting_entity_name,
        )

    def mark_model_getting_dataset(
        self, name: str, getting_entity_name: str, from_cache: bool
    ) -> None:
        self._update_entity(
            EntityType.MODEL,
            name,
            entity_download_transfer_state=DatasetTransferState(0, getting_entity_name),
            status=EntityStatus.ENTITY_DOWNLOADING
            if not from_cache
            else EntityStatus.ENTITY_FROM_CACHE,
            loading_cache_entity=None if not from_cache else getting_entity_name,
        )

    def mark_dataset_getting_model(
        self,
        name: str,
        getting_entity_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            entity_download_transfer_state=state,
            status=EntityStatus.ENTITY_DOWNLOADING
            if not from_cache
            else EntityStatus.ENTITY_FROM_CACHE,
            loading_cache_entity=None if not from_cache else getting_entity_name,
        )

    def mark_dataset_getting_dataset(
        self, name: str, getting_entity_name: str, from_cache: bool
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET,
            name,
            entity_download_transfer_state=DatasetTransferState(0, getting_entity_name),
            status=EntityStatus.ENTITY_DOWNLOADING
            if not from_cache
            else EntityStatus.ENTITY_FROM_CACHE,
            loading_cache_entity=None if not from_cache else getting_entity_name,
        )

    def mark_model_loaded(
        self,
        name: str,
    ) -> None:
        self._update_entity(EntityType.MODEL, name, status=EntityStatus.ENTITY_LOADED)

    def mark_dataset_loaded(
        self,
        name: str,
    ) -> None:
        self._update_entity(
            EntityType.DERIVED_DATASET, name, status=EntityStatus.ENTITY_LOADED
        )
