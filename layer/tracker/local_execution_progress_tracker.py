import uuid
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple, Union

from rich.console import Console
from rich.progress import TaskID
from yarl import URL

from layer.config import Config
from layer.contracts.assertions import Assertion
from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.tracker import (
    AssetTracker,
    AssetTrackerStatus,
    DatasetTransferState,
    ResourceTransferState,
)

from .output import get_progress_ui
from .progress_tracker import RunProgressTracker


class LocalExecutionRunProgressTracker(RunProgressTracker):
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
        self._task_ids: Dict[Tuple[AssetType, str], TaskID] = {}

    @contextmanager
    def track(self) -> Iterator["LocalExecutionRunProgressTracker"]:
        """
        Initializes tracking. Meant to be used with a `with` construct.
        """
        with self._progress:
            yield self

    def add_build(self, name: str) -> None:
        self._get_or_create_task(AssetType.DATASET, name)

    def add_model(self, name: str) -> None:
        self._get_or_create_task(AssetType.MODEL, name)

    def _get_or_create_task(self, asset_type: AssetType, name: str) -> TaskID:
        if (asset_type, name) not in self._task_ids:
            task_id = self._progress.add_task(
                start=False,
                asset=AssetTracker(
                    type=asset_type, name=name, status=AssetTrackerStatus.PENDING
                ),
                description="pending",
            )
            self._task_ids[(asset_type, name)] = task_id
        return self._task_ids[(asset_type, name)]

    def _update_asset(  # noqa: C901
        self,
        type_: AssetType,
        name: str,
        *,
        status: Optional[AssetTrackerStatus] = None,
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
        loading_cache_asset: Optional[str] = None,
        asset_download_transfer_state: Optional[
            Union[ResourceTransferState, DatasetTransferState]
        ] = None,
    ) -> None:
        task_id = self._get_or_create_task(type_, name)
        # noinspection PyProtectedMember
        task = self._progress._tasks[task_id]  # pylint: disable=protected-access
        asset: AssetTracker = task.fields["asset"]
        if status:
            asset.status = status
        if url:
            asset.base_url = url
        if version:
            asset.version = version
        if build_idx:
            asset.build_idx = build_idx
        if error_reason:
            asset.error_reason = error_reason
        if dataset_transfer_state:
            asset.dataset_transfer_state = dataset_transfer_state
        if model_transfer_state:
            asset.model_transfer_state = model_transfer_state
        if resource_transfer_state:
            asset.resource_transfer_state = resource_transfer_state
        if loading_cache_asset:
            asset.loading_cache_asset = loading_cache_asset
        if asset_download_transfer_state:
            asset.asset_download_transfer_state = asset_download_transfer_state

        if description is not None:
            self._progress.update(
                task_id,
                description=description,
            )

        if status == AssetTrackerStatus.PENDING:
            self._progress.update(task_id, description="pending")
        elif status == AssetTrackerStatus.SAVING:
            if type_ == AssetType.DATASET:
                # Even if we go through all steps, we still want to keep track of time elapsed until the status is DONE, so we add +1 to total_steps.
                self._progress.update(
                    task_id,
                    completed=cur_step,
                    total=(total_steps + 1),
                    description=f"saved {cur_step}/{total_steps} rows",
                )
            elif type_ == AssetType.MODEL:
                self._progress.update(
                    task_id,
                    description="saving",
                )
        elif status == AssetTrackerStatus.BUILDING:
            if not task.started:
                self._progress.start_task(task_id)
            self._progress.update(task_id, description="building")
        elif (
            status == AssetTrackerStatus.ASSET_DOWNLOADING
            or status == AssetTrackerStatus.ASSET_FROM_CACHE
        ):
            if not task.started:
                self._progress.start_task(task_id)
        elif status == AssetTrackerStatus.TRAINING:
            if not task.started:
                self._progress.start_task(task_id)
            self._progress.update(task_id, description="training")
        elif status == AssetTrackerStatus.DONE:
            self._progress.update(task_id, completed=task.total, description="done")
            self._progress.stop_task(task_id)
        elif status == AssetTrackerStatus.ERROR:
            self._progress.stop_task(task_id)
            self._progress.update(task_id, completed=0, description="error")

    def _get_url(self, asset_type: AssetType, name: str) -> URL:
        assert self._project_name
        assert self._account_name
        return AssetPath(
            asset_name=name,
            asset_type=asset_type,
            org_name=self._account_name,
            project_name=self._project_name,
        ).url(self._config.url)

    def mark_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            url=self._get_url(AssetType.DATASET, name),
        )

    def mark_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.BUILDING,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_idx,
        )

    def mark_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.DONE,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_index,
        )

    def update_dataset_saving_progress(
        self, name: str, cur_step: int, total_steps: int
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.SAVING,
            cur_step=cur_step,
            total_steps=total_steps,
        )

    def mark_model_saving(self, name: str) -> None:
        self._update_asset(AssetType.MODEL, name, status=AssetTrackerStatus.SAVING)

    def mark_model_saved(self, name: str) -> None:
        self._update_asset(AssetType.MODEL, name, status=AssetTrackerStatus.DONE)

    def mark_model_training(
        self, name: str, version: Optional[str] = None, train_idx: Optional[str] = None
    ) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.TRAINING,
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
        self._update_asset(
            AssetType.MODEL,
            name,
            url=self._get_url(AssetType.MODEL, name),
            status=AssetTrackerStatus.SAVING,
        )

    def mark_model_train_failed(self, name: str, reason: str) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ERROR,
        )

    def mark_model_running_assertions(self, name: str) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ASSERTING,
            description="asserting...",
        )

    def mark_model_running_assertion(self, name: str, assertion: Assertion) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            description=str(assertion),
        )

    def mark_model_completed_assertions(self, name: str) -> None:
        self._update_asset(AssetType.MODEL, name, description="asserted")

    def mark_model_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        stringified = [str(assertion) for assertion in assertions]
        error_msg = f"failed: {', '.join(stringified)}"
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ERROR,
            error_reason=error_msg,
        )

    def mark_dataset_running_assertions(self, name: str) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.ASSERTING,
            description="asserting...",
        )

    def mark_dataset_running_assertion(self, name: str, assertion: Assertion) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            description=str(assertion),
        )

    def mark_dataset_completed_assertions(self, name: str) -> None:
        self._update_asset(AssetType.DATASET, name, description="asserted")

    def mark_dataset_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        stringified = [str(assertion) for assertion in assertions]
        error_msg = f"failed: {', '.join(stringified)}"
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.ERROR,
            error_reason=error_msg,
        )

    def mark_dataset_saving_result(
        self, name: str, state: DatasetTransferState
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            dataset_transfer_state=state,
            status=AssetTrackerStatus.RESULT_UPLOADING,
        )

    def mark_model_saving_result(self, name: str, state: ResourceTransferState) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            model_transfer_state=state,
            status=AssetTrackerStatus.RESULT_UPLOADING,
        )

    def mark_model_getting_model(
        self,
        name: str,
        getting_asset_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            asset_download_transfer_state=state,
            status=AssetTrackerStatus.ASSET_DOWNLOADING
            if not from_cache
            else AssetTrackerStatus.ASSET_FROM_CACHE,
            loading_cache_asset=None if not from_cache else getting_asset_name,
        )

    def mark_model_getting_dataset(
        self, name: str, getting_asset_name: str, from_cache: bool
    ) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            asset_download_transfer_state=DatasetTransferState(0, getting_asset_name),
            status=AssetTrackerStatus.ASSET_DOWNLOADING
            if not from_cache
            else AssetTrackerStatus.ASSET_FROM_CACHE,
            loading_cache_asset=None if not from_cache else getting_asset_name,
        )

    def mark_dataset_getting_model(
        self,
        name: str,
        getting_asset_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            asset_download_transfer_state=state,
            status=AssetTrackerStatus.ASSET_DOWNLOADING
            if not from_cache
            else AssetTrackerStatus.ASSET_FROM_CACHE,
            loading_cache_asset=None if not from_cache else getting_asset_name,
        )

    def mark_dataset_getting_dataset(
        self, name: str, getting_asset_name: str, from_cache: bool
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            asset_download_transfer_state=DatasetTransferState(0, getting_asset_name),
            status=AssetTrackerStatus.ASSET_DOWNLOADING
            if not from_cache
            else AssetTrackerStatus.ASSET_FROM_CACHE,
            loading_cache_asset=None if not from_cache else getting_asset_name,
        )

    def mark_model_loaded(
        self,
        name: str,
    ) -> None:
        self._update_asset(
            AssetType.MODEL, name, status=AssetTrackerStatus.ASSET_LOADED
        )

    def mark_dataset_loaded(
        self,
        name: str,
    ) -> None:
        self._update_asset(
            AssetType.DATASET, name, status=AssetTrackerStatus.ASSET_LOADED
        )
