import time

from ipywidgets import GridspecLayout
from ipywidgets import Button, Layout, Box, HBox, GridBox, VBox, Image, Label, HTML, IntProgress, Output, Widget
from layer.contracts.tracker import AssetTracker, AssetType
from datetime import datetime
from typing import Optional, Tuple, Dict
from types import TracebackType
from IPython.display import display, update_display, display_html, clear_output
import threading
import random
import string


def create_expanded_button(description, button_style):
    return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))


class RunTask:
    def __init__(self, asset_tracker: AssetTracker):
        self._asset_tracker: AssetTracker = asset_tracker
        self._description: str = ""
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._total_steps = 100
        self._completed_steps = 2

    @property
    def asset_tracker(self) -> AssetTracker:
        return self._asset_tracker

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def completed_steps(self) -> int:
        return self._completed_steps


class NotebookUiRenderer:
    def __init__(self, tasks: Dict[Tuple[str, AssetType], RunTask]):
        self._tasks: Dict[Tuple[str, AssetType], RunTask] = tasks

    @staticmethod
    def _render_spinner(task: RunTask) -> Widget:
        return HTML(
            value="Spinner",
            # placeholder='Some HTML',
            # description='Some HTML',
        )

    @staticmethod
    def _render_asset_name(task: RunTask) -> Widget:
        return Label(value=task.asset_tracker.name)

    @staticmethod
    def _render_progress_bar(task: RunTask) -> Widget:
        return IntProgress(
            value=(task.completed_steps + random.randint(0, 100)) % 100,
            min=0,
            max=task.total_steps,
            description='',
            bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
            style={'bar_color': 'maroon'},
            orientation='horizontal'
        )

    @staticmethod
    def _render_status(task: RunTask) -> Widget:
        return Label(value=task.asset_tracker.status.value)

    @staticmethod
    def _render_stats(task: RunTask) -> Widget:
        return Label(value="[ SOME STATS HERE ]")

    @staticmethod
    def _render_url_link(task: RunTask) -> Widget:
        return Label(value="Link here")

    @staticmethod
    def _render_task_error(task: RunTask) -> Widget:
        return Label(value="Error message")

    @staticmethod
    def _render_task(task: RunTask) -> Box:
        spinner = NotebookUiRenderer._render_spinner(task)
        asset_name = NotebookUiRenderer._render_asset_name(task)
        progress_bar = NotebookUiRenderer._render_progress_bar(task)
        status = NotebookUiRenderer._render_status(task)
        stats = NotebookUiRenderer._render_stats(task)
        task_rows = []
        main_row = HBox([spinner, asset_name, progress_bar, status, stats])
        task_rows.append(main_row)
        if task.asset_tracker.base_url:
            url_link = NotebookUiRenderer._render_url_link(task)
            task_rows.append(url_link)
        if task.asset_tracker.error_reason:
            error_message = NotebookUiRenderer._render_task_error(task)
            task_rows.append(error_message)
        return VBox(task_rows)

    def _render_run_description_area(self) -> Widget:
        return Label(value="Large description box here")

    def _render_user_log_lines(self) -> Widget:
        return Label(value="User log lines here")

    def render_ui(self) -> Box:
        task_widget_rows = [self._render_task(task_value) for task_key, task_value in self._tasks.items()]
        run_description_widget = self._render_run_description_area()
        user_log_lines_widget = self._render_user_log_lines()
        grid = VBox(task_widget_rows + [run_description_widget] + [user_log_lines_widget])
        return grid


class NotebookUi:
    def __init__(self):
        self._tasks: Dict[Tuple[str, AssetType], RunTask] = {
            ("asd", AssetType.MODEL): RunTask(AssetTracker(AssetType.MODEL, "asd")),
            ("asd2", AssetType.MODEL): RunTask(AssetTracker(AssetType.MODEL, "asd2")),
            ("asd3", AssetType.MODEL): RunTask(AssetTracker(AssetType.MODEL, "asd3")),
        }
        self._ui_thread = threading.Thread(target=self.display_ui, args=[])
        self._render_ui: bool = False
        self._ui_renders_per_sec = 10
        self._ui_renderer = NotebookUiRenderer(self._tasks)
        # self._display_id: str = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    def display_ui(self) -> None:
        render_object = Box()
        display(render_object)
        while self._render_ui:
            widgets_ui = self._ui_renderer.render_ui()
            render_object.children = [widgets_ui]
            time.sleep(1 / self._ui_renders_per_sec)

    def __enter__(self) -> None:
        self._render_ui = True
        self._ui_thread.start()

    def __exit__(
            self,
            exc_type: Optional[BaseException],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
    ) -> None:
        self._render_ui = False
        self._ui_thread.join(timeout=None)
