import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from layerapi.api.ids_pb2 import RunId
from rich.progress import Task
from rich.text import Text
from yarl import URL

from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.tracker import (
    AssetTracker,
    AssetTrackerStatus,
    DatasetTransferState,
    ResourceTransferState,
)
from layer.exceptions.exceptions import ProjectBaseException, ProjectRunnerError

from .output import get_progress_ui
from .progress_tracker import RunProgressTracker


class UIRunProgressTracker(RunProgressTracker):
    def __init__(
        self,
        url: URL,
        account_name: str,
        project_name: str,
        assets: Optional[List[Tuple[AssetType, str]]] = None,
    ):
        self._url = url
        self._account_name = account_name
        self._project_name = project_name
        self._assets = assets
        self._progress = get_progress_ui()
        self._tasks: Dict[Tuple[AssetType, str], Task] = {}

    @staticmethod
    def __google_colab_ipykernel_fix() -> None:
        """
        Fixes https://linear.app/layer/issue/LAY-3286/replace-rich-as-a-dependency-for-the-ui-of-the-sdk
        It works by clearing the logger handlers as some of them appear to be interacting with the same resources
        rich interacts with, leading to hitting a bug in old ipykernel-s(<5.2) (https://github.com/ipython/ipykernel/pull/463)
        """
        import logging.config

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
            }
        )

    @contextmanager
    def track(self) -> Iterator["RunProgressTracker"]:
        """
        Initializes tracking. Meant to be used with a `with` construct.
        """
        self.__google_colab_ipykernel_fix()
        with self._progress:
            self._init_tasks()
            yield self
            self._progress.refresh()

    def _get_url(self, asset_type: AssetType, name: str) -> URL:
        return AssetPath(
            asset_name=name,
            asset_type=asset_type,
            org_name=self._account_name,
            project_name=self._project_name,
        ).url(self._url)

    def _init_tasks(self) -> None:
        if self._assets:
            for asset in self._assets:
                self._get_or_create_task(
                    asset_type=asset[0],
                    asset_name=asset[1],
                )

    def _get_or_create_task(self, asset_type: AssetType, asset_name: str) -> Task:
        task_key = (asset_type, asset_name)
        if task_key not in self._tasks:
            task_id = self._progress.add_task(
                start=False,
                asset=AssetTracker(
                    type=asset_type, name=asset_name, status=AssetTrackerStatus.PENDING
                ),
                description="pending",
            )
            task = self._progress._tasks[task_id]  # pylint: disable=protected-access
            self._tasks[task_key] = task
        return self._tasks[task_key]

    def add_asset(self, asset_type: AssetType, asset_name: str) -> None:
        self._get_or_create_task(asset_type, asset_name)

    def _get_asset(self, asset_type: AssetType, asset_name: str) -> AssetTracker:
        task = self._get_or_create_task(asset_type, asset_name)
        asset = task.fields["asset"]
        return asset

    def _update_asset(
        self,
        asset_type: AssetType,
        asset_name: str,
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
        task = self._get_or_create_task(asset_type, asset_name)
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

        progress_args: Dict[str, Any] = {
            "description": description if description is not None else status,
        }

        if status == AssetTrackerStatus.SAVING:
            if asset_type == AssetType.DATASET:
                # Even if we go through all steps, we still want to keep track of time elapsed until the status is DONE, so we add +1 to total_steps.
                progress_args["completed"] = cur_step
                progress_args["total"] = total_steps + 1
                progress_args["description"] = f"saved {cur_step}/{total_steps} rows"
        elif status in (
            AssetTrackerStatus.ASSET_DOWNLOADING,
            AssetTrackerStatus.ASSET_FROM_CACHE,
        ):
            if not task.started:
                task.start_time = task.stop_time = None
                self._progress.start_task(task.id)
                assert task.total is not None
                progress_args["completed"] = task.total * 0.1
        elif status in (
            AssetTrackerStatus.BUILDING,
            AssetTrackerStatus.TRAINING,
        ):
            if not task.started:
                self._progress.start_task(task.id)
        elif status == AssetTrackerStatus.DONE:
            progress_args["completed"] = task.total
            self._progress.stop_task(task.id)
        elif status == AssetTrackerStatus.ERROR:
            self._progress.stop_task(task.id)
            progress_args["completed"] = 0
        self._progress.update(task.id, **progress_args)

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

    def mark_start_running(self, run_id: RunId) -> None:
        pass

    def mark_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            url=self._get_url(AssetType.DATASET, name),
            status=AssetTrackerStatus.PENDING,
        )

    def mark_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the dataset is successfully built and
        # it causes a flicker in the progress bar. This is a workaround.
        asset = self._get_asset(AssetType.DATASET, name)
        if asset.status == AssetTrackerStatus.DONE:
            return

        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.BUILDING,
            url=self._get_url(AssetType.DATASET, name),
            version=version,
            build_idx=build_idx,
        )

    def mark_dataset_failed(self, name: str, reason: str) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.ERROR,
            error_reason=f"{reason}",
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

    def mark_model_saving(self, name: str) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.SAVING,
        )

    def mark_model_saved(
        self,
        name: str,
        version: Optional[str] = None,
        train_index: Optional[str] = None,
    ) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.DONE,
            url=self._get_url(AssetType.MODEL, name),
            version=version,
            build_idx=train_index,
        )

    def mark_model_training(
        self, name: str, version: Optional[str] = None, train_idx: Optional[str] = None
    ) -> None:
        # For some reason, this function is called even after the model is successfully trained and
        # it causes a flicker in the progress bar. This is a workaround.
        asset = self._get_asset(AssetType.MODEL, name)
        if asset.status == AssetTrackerStatus.DONE:
            return

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
            status=AssetTrackerStatus.DONE,
            url=self._get_url(AssetType.MODEL, name),
            version=version,
            build_idx=train_index,
        )

    def mark_model_train_failed(
        self, name: str, reason: str
    ) -> None:  # TODO(volkan) check that reason is always given
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ERROR,
            error_reason=f"{reason}",
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

    def mark_dataset_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            description="uploading",
            resource_transfer_state=state,
            status=AssetTrackerStatus.RESOURCE_UPLOADING,
        )

    def mark_dataset_resources_uploaded(self, name: str) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            description="uploaded",
            status=AssetTrackerStatus.PENDING,  # Pending here meaning that training has not started yet
        )

    def mark_model_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            description="uploading",
            resource_transfer_state=state,
            status=AssetTrackerStatus.RESOURCE_UPLOADING,
        )

    def mark_model_resources_uploaded(self, name: str) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            description="uploaded",
            status=AssetTrackerStatus.PENDING,
        )

    def mark_model_running_assertions(self, name: str) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ASSERTING,
        )

    def mark_model_running_assertion(self, name: str, assertion: Assertion) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ASSERTING,
            description=str(assertion),
        )

    def mark_model_completed_assertions(self, name: str) -> None:
        self._update_asset(
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ASSERTED,
        )

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
        )

    def mark_dataset_running_assertion(self, name: str, assertion: Assertion) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            description=str(assertion),
        )

    def mark_dataset_completed_assertions(self, name: str) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.ASSERTED,
        )

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
            AssetType.MODEL,
            name,
            status=AssetTrackerStatus.ASSET_LOADED,
        )

    def mark_dataset_loaded(
        self,
        name: str,
    ) -> None:
        self._update_asset(
            AssetType.DATASET,
            name,
            status=AssetTrackerStatus.ASSET_LOADED,
        )
