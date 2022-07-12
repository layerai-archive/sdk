from datetime import timedelta
from enum import Enum
from typing import Any, List

import humanize
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.progress import ProgressColumn, Task
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from layer.contracts.tracker import (
    AssetTracker,
    AssetTrackerStatus,
    DatasetTransferState,
    ResourceTransferState,
)


class ProgressStyle(str, Enum):
    NONE = ""

    GREY = "rgb(161,161,169)"
    GREEN = "rgb(21,127,61)"
    ORANGE = "rgb(251,147,60)"
    BLUE = "rgb(37,99,234)"
    GRAY = "rgb(155,155,159)"
    BLACK = "rgb(0,0,0)"
    DEFAULT = "default"

    LINK = f"underline {GREY}"
    PENDING = GREY
    DONE = GREEN
    ERROR = ORANGE


class AssetColumn(ProgressColumn):
    _status_style_map = {
        AssetTrackerStatus.PENDING: ProgressStyle.NONE,
        AssetTrackerStatus.SAVING: ProgressStyle.NONE,
        AssetTrackerStatus.BUILDING: ProgressStyle.NONE,
        AssetTrackerStatus.TRAINING: ProgressStyle.NONE,
        AssetTrackerStatus.DONE: ProgressStyle.DONE,
        AssetTrackerStatus.ERROR: ProgressStyle.ERROR,
        AssetTrackerStatus.ASSERTING: ProgressStyle.NONE,
        AssetTrackerStatus.RESOURCE_UPLOADING: ProgressStyle.NONE,
        AssetTrackerStatus.RESULT_UPLOADING: ProgressStyle.NONE,
        AssetTrackerStatus.ASSET_DOWNLOADING: ProgressStyle.BLUE,
        AssetTrackerStatus.ASSET_FROM_CACHE: ProgressStyle.BLUE,
        AssetTrackerStatus.ASSET_LOADED: ProgressStyle.DONE,
    }

    def render(self, task: Task) -> RenderableType:
        asset = self._get_asset(task)
        description_text = self._render_description(task)
        stats_text = self._render_stats(task)

        table = Table.grid(padding=(0, 1, 0, 1), pad_edge=True, expand=False)
        table.add_column(width=20, min_width=20, max_width=20)  # name
        table.add_column(width=10, min_width=10, max_width=10)  # bar
        table.add_column(width=len(description_text))  # task description
        table.add_column()  # stats
        table.add_row(
            asset.name,
            self._render_progress_bar(task),
            description_text,
            stats_text,
        )
        renderables: List[RenderableType] = [table]
        if asset.base_url:
            table = Table.grid(padding=(0, 1, 0, 1), pad_edge=True)
            table.add_column()
            table.add_row(self._render_url(asset))
            renderables.append(table)
        if asset.error_reason:
            table = Table.grid(padding=(0, 1, 0, 1), pad_edge=True)
            table.add_column(overflow="fold")
            table.add_row(Text.from_markup(f"[red]{escape(asset.error_reason)}[/red]"))
            table.add_row(Text("Aborting...", style="bold"))
            renderables.append(table)

        return Group(*renderables)

    @staticmethod
    def _render_state(state: ResourceTransferState, show_num_files: bool = True) -> str:
        render_parts = []
        if show_num_files:
            render_parts.append(
                f"{state.transferred_num_files}/{state.total_num_files} files"
            )
        at_least_a_gig = state.total_resource_size_bytes > 1_000_000_000
        size_format = "%.1f" if at_least_a_gig else "%.f"
        transferred_bytes = humanize.naturalsize(
            state.transferred_resource_size_bytes, format=size_format
        )
        total_resources_bytes = humanize.naturalsize(
            state.total_resource_size_bytes, format=size_format
        )
        render_parts.append(f"{transferred_bytes}/{total_resources_bytes}")
        if state.transferred_num_files < state.total_num_files:
            render_parts.append(
                f"{humanize.naturalsize(state.get_bandwidth_in_previous_seconds(), format='%.1f')}/s"
            )
        return " | ".join(render_parts)

    @staticmethod
    def _render_dataset_state(state: DatasetTransferState) -> str:
        return f"{state.transferred_num_rows}/{state.total_num_rows} rows"

    def _render_progress_bar(self, task: Task) -> RenderableType:
        # Set task.completed = min(..., int(task.total -  1)) to prevent task timer from stopping
        asset = self._get_asset(task)
        style_map = {
            AssetTrackerStatus.PENDING: (
                True,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
            AssetTrackerStatus.SAVING: (
                True,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
            AssetTrackerStatus.BUILDING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLACK,
            ),
            AssetTrackerStatus.TRAINING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLACK,
            ),
            AssetTrackerStatus.DONE: (False, ProgressStyle.DONE, ProgressStyle.DONE),
            AssetTrackerStatus.ERROR: (
                False,
                ProgressStyle.ERROR,
                ProgressStyle.ERROR,
            ),
            AssetTrackerStatus.ASSERTING: (
                True,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
            AssetTrackerStatus.RESOURCE_UPLOADING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLUE,
            ),
            AssetTrackerStatus.RESULT_UPLOADING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLUE,
            ),
            AssetTrackerStatus.ASSET_DOWNLOADING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLUE,
            ),
            AssetTrackerStatus.ASSET_FROM_CACHE: (
                True,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
            AssetTrackerStatus.ASSET_LOADED: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
        }
        pulse, style, completed_style = style_map.get(
            asset.status, (False, ProgressStyle.DEFAULT, ProgressStyle.DONE)
        )
        if asset.status == AssetTrackerStatus.ASSET_DOWNLOADING and isinstance(
            asset.asset_download_transfer_state, DatasetTransferState
        ):
            pulse = True  # Pulsing bar as we currently cannot closely track the downloading of a dataset

        assert task.total is not None
        if (
            asset.status == AssetTrackerStatus.TRAINING
            or asset.status == AssetTrackerStatus.BUILDING
        ):
            fraction = round(task.total * 0.05)
            task.completed = min(
                (task.completed + fraction) % task.total, task.total - 1
            )
        elif asset.status == AssetTrackerStatus.RESOURCE_UPLOADING:
            assert asset.resource_transfer_state
            state = asset.resource_transfer_state
            completed = (
                (
                    state.transferred_resource_size_bytes
                    / state.total_resource_size_bytes
                )
                if state.total_resource_size_bytes > 0
                else 0
            )
            task.completed = min(int(task.total * completed), int(task.total - 1))
        elif asset.status == AssetTrackerStatus.ASSET_LOADED:
            task.completed = task.total
        elif (
            asset.status == AssetTrackerStatus.RESULT_UPLOADING
            and asset.dataset_transfer_state
        ):
            dataset_state = asset.dataset_transfer_state
            completed = (
                (dataset_state.transferred_num_rows / dataset_state.total_num_rows)
                if dataset_state.total_num_rows > 0
                else 0
            )
            task.completed = min(int(task.total * completed), int(task.total - 1))
        elif (
            asset.status == AssetTrackerStatus.RESULT_UPLOADING
        ) and asset.model_transfer_state:
            model_state = asset.model_transfer_state
            completed = (
                (
                    model_state.transferred_resource_size_bytes
                    / model_state.total_resource_size_bytes
                )
                if model_state.total_resource_size_bytes > 0
                else 0
            )
            task.completed = min(int(task.total * completed), int(task.total - 1))
        elif (
            (asset.status == AssetTrackerStatus.ASSET_DOWNLOADING)
            and asset.asset_download_transfer_state
            and isinstance(asset.asset_download_transfer_state, ResourceTransferState)
        ):
            model_state = asset.asset_download_transfer_state
            completed = (
                (
                    model_state.transferred_resource_size_bytes
                    / model_state.total_resource_size_bytes
                )
                if model_state.total_resource_size_bytes > 0
                else 0
            )
            task.completed = min(int(task.total * completed), int(task.total - 1))

        return ProgressBar(
            total=max(0.0, task.total),
            completed=max(0.0, task.completed),
            animation_time=task.get_time(),
            style=style,
            pulse=pulse,
            complete_style=completed_style,
            finished_style=completed_style,
            pulse_style=ProgressStyle.PENDING,
        )

    @staticmethod
    def _get_asset(task: Task) -> AssetTracker:
        return task.fields["asset"]

    @staticmethod
    def _get_elapsed_time_s(task: Task) -> int:
        elapsed_time = task.finished_time if task.finished else task.elapsed
        return int(elapsed_time) if elapsed_time else 0

    def _compute_time_string(self, task: Task) -> str:
        asset = self._get_asset(task)
        delta: Any = timedelta(seconds=self._get_elapsed_time_s(task))
        if asset.status == AssetTrackerStatus.RESOURCE_UPLOADING:
            assert asset.resource_transfer_state
            delta = timedelta(seconds=asset.resource_transfer_state.get_eta_seconds())
        elif (
            asset.status == AssetTrackerStatus.RESULT_UPLOADING
            and asset.dataset_transfer_state
        ):
            delta = timedelta(seconds=asset.dataset_transfer_state.get_eta_seconds())
        elif (
            asset.status == AssetTrackerStatus.RESULT_UPLOADING
            and asset.model_transfer_state
        ):
            delta = timedelta(seconds=asset.model_transfer_state.get_eta_seconds())
        elif asset.status == AssetTrackerStatus.ASSET_DOWNLOADING:
            assert asset.asset_download_transfer_state
            if isinstance(asset.asset_download_transfer_state, ResourceTransferState):
                delta = timedelta(
                    seconds=asset.asset_download_transfer_state.get_eta_seconds()
                )
            else:
                delta = "-:--:--"
        elif (
            asset.status == AssetTrackerStatus.PENDING
            or asset.status == AssetTrackerStatus.ASSET_FROM_CACHE
        ):
            delta = "-:--:--"
        elif (
            asset.status == AssetTrackerStatus.TRAINING
            or asset.status == AssetTrackerStatus.BUILDING
        ):
            delta = timedelta(seconds=self._get_elapsed_time_s(task))
        return str(delta)

    def _render_stats(self, task: Task) -> RenderableType:
        asset = self._get_asset(task)
        rendered_time: str = self._compute_time_string(task)
        rendered_state = None

        if (
            asset.resource_transfer_state
            and asset.status == AssetTrackerStatus.RESOURCE_UPLOADING
        ):
            rendered_state = self._render_state(asset.resource_transfer_state)
        elif (
            asset.dataset_transfer_state
            and asset.status == AssetTrackerStatus.RESULT_UPLOADING
        ):
            rendered_state = self._render_dataset_state(asset.dataset_transfer_state)
        elif (
            asset.model_transfer_state
            and asset.status == AssetTrackerStatus.RESULT_UPLOADING
        ):
            rendered_state = self._render_state(asset.model_transfer_state, False)
        elif asset.status == AssetTrackerStatus.ASSET_DOWNLOADING:
            assert asset.asset_download_transfer_state
            if isinstance(asset.asset_download_transfer_state, ResourceTransferState):
                rendered_state = self._render_state(
                    asset.asset_download_transfer_state, False
                )

        color = ProgressStyle.GRAY
        stats = (
            [rendered_time]
            if rendered_state is None
            else [rendered_state, rendered_time]
        )
        stats = (" | ".join(stats)).split(" | ")
        stats = [f"[{color}]{stat}[/{color}]" for stat in stats]
        text = ", ".join(stats)
        return Text.from_markup(f"[{text}]", style="default", justify="center")

    def _render_description(self, task: Task) -> Text:
        asset = self._get_asset(task)
        if (
            asset.status == AssetTrackerStatus.RESOURCE_UPLOADING
            or asset.status == AssetTrackerStatus.RESULT_UPLOADING
        ):
            text = "uploading"
        elif asset.status == AssetTrackerStatus.ASSET_DOWNLOADING:
            text = "downloading"
        elif asset.status == AssetTrackerStatus.ASSET_FROM_CACHE:
            text = "from cache"
        elif asset.status == AssetTrackerStatus.ASSET_LOADED:
            text = "loaded"
        elif asset.status == AssetTrackerStatus.DONE:
            text = "done"
        else:
            text = task.description
        if (
            asset.status == AssetTrackerStatus.ASSET_LOADED
            or asset.status == AssetTrackerStatus.DONE
            or asset.status == AssetTrackerStatus.ERROR
        ):
            style = self._status_style_map[asset.status]
        else:
            style = ProgressStyle.GRAY
        return Text(
            text.upper(),
            overflow="fold",
            style=style,
            justify="center",
        )

    @staticmethod
    def _render_url(asset: AssetTracker) -> RenderableType:
        link = (
            f"{asset.base_url}?v={asset.version}.{asset.build_idx}"
            if (asset.version and asset.build_idx)
            else asset.base_url
        )
        return Group(
            *[
                Text.from_markup(
                    f"↳ [link={str(link)}]{str(link)}[/link]",
                    style=ProgressStyle.LINK,
                    overflow="fold",
                    justify="default",
                )
            ]
        )
