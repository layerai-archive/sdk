from datetime import timedelta
from enum import Enum
from typing import Any, List

import humanize  # type: ignore
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.progress import ProgressColumn, Task
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from layer.contracts.entities import Entity, EntityStatus
from layer.contracts.runs import DatasetTransferState, ResourceTransferState


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


class EntityColumn(ProgressColumn):
    _status_style_map = {
        EntityStatus.PENDING: ProgressStyle.NONE,
        EntityStatus.SAVING: ProgressStyle.NONE,
        EntityStatus.BUILDING: ProgressStyle.NONE,
        EntityStatus.TRAINING: ProgressStyle.NONE,
        EntityStatus.DONE: ProgressStyle.DONE,
        EntityStatus.ERROR: ProgressStyle.ERROR,
        EntityStatus.ASSERTING: ProgressStyle.NONE,
        EntityStatus.RESOURCE_UPLOADING: ProgressStyle.NONE,
        EntityStatus.RESULT_UPLOADING: ProgressStyle.NONE,
        EntityStatus.ENTITY_DOWNLOADING: ProgressStyle.BLUE,
        EntityStatus.ENTITY_FROM_CACHE: ProgressStyle.BLUE,
        EntityStatus.ENTITY_LOADED: ProgressStyle.DONE,
    }

    def render(self, task: Task) -> RenderableType:
        entity = self._get_entity(task)
        description_text = self._render_description(task)
        stats_text = self._render_stats(task)

        table = Table.grid(padding=(0, 1, 0, 1), pad_edge=True, expand=False)
        table.add_column(width=20, min_width=20, max_width=20)  # name
        table.add_column(width=10, min_width=10, max_width=10)  # bar
        table.add_column(width=len(description_text))  # task description
        table.add_column()  # stats
        table.add_row(
            entity.name,
            self._render_progress_bar(task),
            description_text,
            stats_text,
        )
        renderables: List[RenderableType] = [table]
        if entity.base_url:
            table = Table.grid(padding=(0, 1, 0, 1), pad_edge=True)
            table.add_column()
            table.add_row(self._render_url(entity))
            renderables.append(table)
        if entity.error_reason:
            table = Table.grid(padding=(0, 1, 0, 1), pad_edge=True)
            table.add_column(overflow="fold")
            table.add_row(Text.from_markup(f"[red]{escape(entity.error_reason)}[/red]"))
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
        entity = self._get_entity(task)
        style_map = {
            EntityStatus.PENDING: (True, ProgressStyle.DEFAULT, ProgressStyle.DONE),
            EntityStatus.SAVING: (True, ProgressStyle.DEFAULT, ProgressStyle.DONE),
            EntityStatus.BUILDING: (False, ProgressStyle.DEFAULT, ProgressStyle.BLACK),
            EntityStatus.TRAINING: (False, ProgressStyle.DEFAULT, ProgressStyle.BLACK),
            EntityStatus.DONE: (False, ProgressStyle.DONE, ProgressStyle.DONE),
            EntityStatus.ERROR: (False, ProgressStyle.ERROR, ProgressStyle.ERROR),
            EntityStatus.ASSERTING: (True, ProgressStyle.DEFAULT, ProgressStyle.DONE),
            EntityStatus.RESOURCE_UPLOADING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLUE,
            ),
            EntityStatus.RESULT_UPLOADING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLUE,
            ),
            EntityStatus.ENTITY_DOWNLOADING: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.BLUE,
            ),
            EntityStatus.ENTITY_FROM_CACHE: (
                True,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
            EntityStatus.ENTITY_LOADED: (
                False,
                ProgressStyle.DEFAULT,
                ProgressStyle.DONE,
            ),
        }
        pulse, style, completed_style = style_map.get(
            entity.status, (False, ProgressStyle.DEFAULT, ProgressStyle.DONE)
        )
        if entity.status == EntityStatus.ENTITY_DOWNLOADING and isinstance(
            entity.entity_download_transfer_state, DatasetTransferState
        ):
            pulse = True  # Pulsing bar as we currently cannot closely track the downloading of a dataset

        assert task.total is not None
        if (
            entity.status == EntityStatus.TRAINING
            or entity.status == EntityStatus.BUILDING
        ):
            fraction = round(task.total * 0.05)
            task.completed = min(
                (task.completed + fraction) % task.total, task.total - 1
            )
        elif entity.status == EntityStatus.RESOURCE_UPLOADING:
            assert entity.resource_transfer_state
            state = entity.resource_transfer_state
            completed = (
                (
                    state.transferred_resource_size_bytes
                    / state.total_resource_size_bytes
                )
                if state.total_resource_size_bytes > 0
                else 0
            )
            task.completed = min(int(task.total * completed), int(task.total - 1))
        elif entity.status == EntityStatus.ENTITY_LOADED:
            task.completed = task.total
        elif (
            entity.status == EntityStatus.RESULT_UPLOADING
            and entity.dataset_transfer_state
        ):
            dataset_state = entity.dataset_transfer_state
            completed = (
                (dataset_state.transferred_num_rows / dataset_state.total_num_rows)
                if dataset_state.total_num_rows > 0
                else 0
            )
            task.completed = min(int(task.total * completed), int(task.total - 1))
        elif (
            entity.status == EntityStatus.RESULT_UPLOADING
        ) and entity.model_transfer_state:
            model_state = entity.model_transfer_state
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
            (entity.status == EntityStatus.ENTITY_DOWNLOADING)
            and entity.entity_download_transfer_state
            and isinstance(entity.entity_download_transfer_state, ResourceTransferState)
        ):
            model_state = entity.entity_download_transfer_state
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
    def _get_entity(task: Task) -> Entity:
        return task.fields["entity"]

    @staticmethod
    def _get_elapsed_time_s(task: Task) -> int:
        elapsed_time = task.finished_time if task.finished else task.elapsed
        return int(elapsed_time) if elapsed_time else 0

    def _compute_time_string(self, task: Task) -> str:
        entity = self._get_entity(task)
        delta: Any = timedelta(seconds=self._get_elapsed_time_s(task))
        if entity.status == EntityStatus.RESOURCE_UPLOADING:
            assert entity.resource_transfer_state
            delta = timedelta(seconds=entity.resource_transfer_state.get_eta_seconds())
        elif (
            entity.status == EntityStatus.RESULT_UPLOADING
            and entity.dataset_transfer_state
        ):
            delta = timedelta(seconds=entity.dataset_transfer_state.get_eta_seconds())
        elif (
            entity.status == EntityStatus.RESULT_UPLOADING
            and entity.model_transfer_state
        ):
            delta = timedelta(seconds=entity.model_transfer_state.get_eta_seconds())
        elif entity.status == EntityStatus.ENTITY_DOWNLOADING:
            assert entity.entity_download_transfer_state
            if isinstance(entity.entity_download_transfer_state, ResourceTransferState):
                delta = timedelta(
                    seconds=entity.entity_download_transfer_state.get_eta_seconds()
                )
            else:
                delta = "-:--:--"
        elif (
            entity.status == EntityStatus.PENDING
            or entity.status == EntityStatus.ENTITY_FROM_CACHE
        ):
            delta = "-:--:--"
        elif (
            entity.status == EntityStatus.TRAINING
            or entity.status == EntityStatus.BUILDING
        ):
            delta = timedelta(seconds=self._get_elapsed_time_s(task))
        return str(delta)

    def _render_stats(self, task: Task) -> RenderableType:
        entity = self._get_entity(task)
        rendered_time: str = self._compute_time_string(task)
        rendered_state = None

        if (
            entity.resource_transfer_state
            and entity.status == EntityStatus.RESOURCE_UPLOADING
        ):
            rendered_state = self._render_state(entity.resource_transfer_state)
        elif (
            entity.dataset_transfer_state
            and entity.status == EntityStatus.RESULT_UPLOADING
        ):
            rendered_state = self._render_dataset_state(entity.dataset_transfer_state)
        elif (
            entity.model_transfer_state
            and entity.status == EntityStatus.RESULT_UPLOADING
        ):
            rendered_state = self._render_state(entity.model_transfer_state, False)
        elif entity.status == EntityStatus.ENTITY_DOWNLOADING:
            assert entity.entity_download_transfer_state
            if isinstance(entity.entity_download_transfer_state, ResourceTransferState):
                rendered_state = self._render_state(
                    entity.entity_download_transfer_state, False
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
        entity = self._get_entity(task)
        if (
            entity.status == EntityStatus.RESOURCE_UPLOADING
            or entity.status == EntityStatus.RESULT_UPLOADING
        ):
            text = "uploading"
        elif entity.status == EntityStatus.ENTITY_DOWNLOADING:
            text = "downloading"
        elif entity.status == EntityStatus.ENTITY_FROM_CACHE:
            text = "from cache"
        elif entity.status == EntityStatus.ENTITY_LOADED:
            text = "loaded"
        else:
            text = task.description
        if (
            entity.status == EntityStatus.RESOURCE_UPLOADING
            or entity.status == EntityStatus.RESULT_UPLOADING
            or entity.status == EntityStatus.ENTITY_DOWNLOADING
            or entity.status == EntityStatus.ENTITY_FROM_CACHE
        ):
            style = ProgressStyle.BLACK
        else:
            style = self._status_style_map[entity.status]
        return Text(
            text.upper(),
            overflow="fold",
            style=style,
            justify="center",
        )

    @staticmethod
    def _render_url(entity: Entity) -> RenderableType:
        link = (
            f"{entity.base_url}?v={entity.version}.{entity.build_idx}"
            if (entity.version and entity.build_idx)
            else entity.base_url
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
