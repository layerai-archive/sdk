from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
            account_name=self._account_name,
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
        url: Optional[URL] = None,
        tag: Optional[str] = None,
        error_reason: str = "",
        warnings: Optional[str] = None,
        description: Optional[str] = None,
        model_transfer_state: Optional[ResourceTransferState] = None,
        dataset_transfer_state: Optional[DatasetTransferState] = None,
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
        if tag:
            asset.tag = tag
        if error_reason:
            asset.error_reason = error_reason
        if warnings:
            asset.warnings = warnings
        if dataset_transfer_state:
            asset.dataset_transfer_state = dataset_transfer_state
        if model_transfer_state:
            asset.model_transfer_state = model_transfer_state
        if loading_cache_asset:
            asset.loading_cache_asset = loading_cache_asset
        if asset_download_transfer_state:
            asset.asset_download_transfer_state = asset_download_transfer_state

        progress_args: Dict[str, Any] = {
            "description": description if description is not None else status,
        }

        if status in (
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

    def mark_running(
        self,
        asset_type: AssetType,
        name: str,
        *,
        tag: Optional[str] = None,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            url=self._get_url(asset_type, name),
            tag=tag,
            status=AssetTrackerStatus.BUILDING
            if asset_type == AssetType.DATASET
            else AssetTrackerStatus.TRAINING,
        )

    def mark_asserting(
        self,
        asset_type: AssetType,
        name: str,
        *,
        assertion: Optional[Assertion] = None,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            status=AssetTrackerStatus.ASSERTING,
            description=str(assertion) if assertion else None,
        )

    def mark_failed_assertions(
        self, asset_type: AssetType, name: str, assertions: List[Assertion]
    ) -> None:
        stringified = [str(assertion) for assertion in assertions]
        error_msg = f"failed: {', '.join(stringified)}"
        self._update_asset(
            asset_type,
            name,
            status=AssetTrackerStatus.ERROR,
            error_reason=error_msg,
        )

    def mark_asserted(self, asset_type: AssetType, name: str) -> None:
        self._update_asset(asset_type, name, status=AssetTrackerStatus.ASSERTED)

    def mark_asset_uploading(
        self,
        asset_type: AssetType,
        name: str,
        *,
        dataset_transfer_state: Optional[DatasetTransferState] = None,
        model_transfer_state: Optional[ResourceTransferState] = None,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            status=AssetTrackerStatus.UPLOADING,
            dataset_transfer_state=dataset_transfer_state,
            model_transfer_state=model_transfer_state,
        )

    def mark_done(
        self,
        asset_type: AssetType,
        name: str,
        *,
        warnings: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            status=AssetTrackerStatus.DONE,
            url=self._get_url(asset_type, name),
            tag=tag,
            warnings=warnings,
        )

    def mark_failed(
        self,
        asset_type: AssetType,
        name: str,
        *,
        reason: str,
        tag: Optional[str] = None,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            status=AssetTrackerStatus.ERROR,
            tag=tag,
            error_reason=f"{reason}",
        )

    def mark_asset_downloading(
        self,
        asset_type: AssetType,
        name: str,
        getting_asset_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            asset_download_transfer_state=state,
            status=AssetTrackerStatus.ASSET_DOWNLOADING
            if not from_cache
            else AssetTrackerStatus.ASSET_FROM_CACHE,
            loading_cache_asset=None if not from_cache else getting_asset_name,
        )

    def mark_asset_downloaded(
        self,
        asset_type: AssetType,
        name: str,
    ) -> None:
        self._update_asset(
            asset_type,
            name,
            status=AssetTrackerStatus.ASSET_LOADED,
        )
