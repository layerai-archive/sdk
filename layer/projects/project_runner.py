import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Callable, List, Optional

import aiohttp
import polling  # type: ignore

from layer.clients.layer import LayerClient
from layer.config import Config
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import Run
from layer.exceptions.exceptions import (
    LayerClientServiceUnavailableException,
    LayerClientTimeoutException,
    LayerResourceExhaustedException,
    LayerServiceUnavailableExceptionDuringExecution,
    LayerServiceUnavailableExceptionDuringInitialization,
    ProjectBaseException,
    ProjectRunnerError,
)
from layer.executables.entrypoint.common import ENV_LAYER_API_TOKEN, ENV_LAYER_API_URL
from layer.projects.execution_planner import (
    build_execution_plan,
    check_asset_dependencies,
)
from layer.projects.progress_tracker_updater import (
    PollingStepFunction,
    ProgressTrackerUpdater,
)
from layer.projects.utils import (
    calculate_hash_by_definitions,
    get_or_create_remote_project,
)
from layer.tracker.progress_tracker import RunProgressTracker
from layer.tracker.utils import get_progress_tracker
from layer.user_logs import LOGS_BUFFER_INTERVAL, show_pipeline_run_logs
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger()


class RunContext:
    def __init__(
        self, is_running: bool, run_completion_time: Optional[datetime] = None
    ) -> None:
        self._is_running = is_running
        self._run_completion_time: Optional[datetime] = run_completion_time

    @property
    def is_running(self) -> bool:
        return self._is_running

    @is_running.setter
    def is_running(self, is_running: bool) -> None:
        self._is_running = is_running

    @property
    def run_completion_time(self) -> Optional[datetime]:
        return self._run_completion_time

    @run_completion_time.setter
    def run_completion_time(self, run_completion_time: Optional[datetime]) -> None:
        self._run_completion_time = run_completion_time

    def user_logs_check_predicate(self) -> bool:
        if self._is_running:
            return True
        assert self._run_completion_time
        return (
            datetime.now() - self._run_completion_time
        ).total_seconds() < LOGS_BUFFER_INTERVAL


class ProjectRunner:
    def __init__(
        self,
        config: Config,
        project_full_name: ProjectFullName,
        functions: List[Any],
    ) -> None:
        self._config = config
        self.project_full_name = project_full_name
        self.definitions: List[FunctionDefinition] = [
            f.get_definition_with_bound_arguments() for f in functions
        ]
        self.files_hash = calculate_hash_by_definitions(self.definitions)

    def run(self, debug: bool = False, printer: Callable[[str], Any] = print) -> Run:
        check_asset_dependencies(self.definitions)
        with LayerClient(self._config.client, logger).init() as client:
            get_or_create_remote_project(client, self.project_full_name)

            with get_progress_tracker(
                url=self._config.url,
                account_name=self.project_full_name.account_name,
                project_name=self.project_full_name.project_name,
                assets=[(d.asset_type, d.asset_name) for d in self.definitions],
            ).track() as tracker:
                try:
                    run = self._start_run(client)
                except LayerClientServiceUnavailableException as e:
                    raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
                try:
                    if debug:
                        run_context = RunContext(is_running=True)
                        logs_thread = threading.Thread(
                            target=show_pipeline_run_logs,
                            args=(client, run.id.value, True),
                            kwargs={
                                "printer": printer,
                                "evaluate_until_callback": run_context.user_logs_check_predicate,
                            },
                        )
                        logs_thread.start()
                    tracker.mark_start_running(run.id)
                    try:
                        self._poll_until_completed(client, tracker, run)
                    except (ProjectBaseException, ProjectRunnerError) as e:
                        tracker.mark_error_messages(e)
                        raise e
                    finally:
                        if debug:
                            self._finish_checking_user_logs(run_context, logs_thread)
                except LayerClientServiceUnavailableException as e:
                    raise LayerServiceUnavailableExceptionDuringExecution(
                        run.id, str(e)
                    )
        return run

    def _start_run(self, client: LayerClient) -> Run:
        asyncio_run_in_thread(self._upload_executable_packages(client))
        try:
            run_id = client.flow_manager.start_run(
                self.project_full_name,
                build_execution_plan(self.definitions),
                self.files_hash,
                self._get_user_command(),
                env_variables={
                    ENV_LAYER_API_URL: str(self._config.url),
                    ENV_LAYER_API_TOKEN: self._config.credentials.access_token,
                },
            )
        except LayerResourceExhaustedException as e:
            raise ProjectRunnerError(f"{e}")
        else:
            return Run(id=run_id, project_full_name=self.project_full_name)

    async def _upload_executable_packages(self, client: LayerClient) -> None:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            upload_tasks = [
                self._upload_executable_package(client, definition, session)
                for definition in self.definitions
            ]
            await asyncio.gather(*upload_tasks)

    async def _upload_executable_package(
        self,
        client: LayerClient,
        function: FunctionDefinition,
        session: aiohttp.ClientSession,
    ) -> None:
        function.package()
        function_name = (
            f"{function.asset_path.asset_type.value}/{function.asset_path.asset_name}"
        )
        with open(function.executable_path, "rb") as package_file:
            presigned_url = client.executor_service_client.get_function_upload_path(
                project_full_name=self.project_full_name,
                function_name=function_name,
            )
            await session.put(
                presigned_url,
                data=package_file,
                timeout=aiohttp.ClientTimeout(total=None),
            )
            download_url = client.executor_service_client.get_function_download_path(
                project_full_name=self.project_full_name,
                function_name=function_name,
            )
            function.set_package_download_url(download_url)

    def _get_user_command(self) -> str:
        functions_string = ", ".join(
            definition.func_name for definition in self.definitions
        )
        return f"{ProjectRunner.run.__name__}([{functions_string}])"

    @staticmethod
    def _finish_checking_user_logs(
        run_context: RunContext, logs_thread: threading.Thread
    ) -> None:
        run_context.run_completion_time = datetime.now()
        run_context.is_running = False
        logs_thread.join()

    def _poll_until_completed(
        self, client: LayerClient, tracker: RunProgressTracker, run: Run
    ) -> None:
        updater = ProgressTrackerUpdater(
            tracker=tracker,
            run=run,
            definitions=self.definitions,
            client=client,
        )

        initial_step_sec = 2
        step_function = PollingStepFunction(max_backoff_sec=3.0, backoff_multiplier=1.2)

        polling.poll(
            lambda: client.flow_manager.get_run_status_history_and_metadata(
                run_id=run.id,
            ),
            check_success=updater.check_completion_and_update_tracker,
            step=initial_step_sec,
            step_function=step_function.step,
            ignore_exceptions=(LayerClientTimeoutException,),
            poll_forever=True,
        )
