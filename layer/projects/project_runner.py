import asyncio
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence

import aiohttp
import polling  # type: ignore
from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.ids_pb2 import RunId

from layer.clients.layer import LayerClient
from layer.config import Config, is_executables_feature_active
from layer.contracts.assets import AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import ApplyResult
from layer.contracts.runs import Run
from layer.exceptions.exceptions import (
    LayerClientException,
    LayerClientServiceUnavailableException,
    LayerClientTimeoutException,
    LayerResourceExhaustedException,
    LayerServiceUnavailableExceptionDuringExecution,
    LayerServiceUnavailableExceptionDuringInitialization,
    ProjectBaseException,
    ProjectInitializationException,
    ProjectRunnerError,
)
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
    verify_project_exists_and_retrieve_project_id,
)
from layer.resource_manager import ResourceManager
from layer.tracker.progress_tracker import RunProgressTracker
from layer.tracker.utils import get_progress_tracker
from layer.user_logs import LOGS_BUFFER_INTERVAL, show_pipeline_run_logs


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
    _tracker: RunProgressTracker

    def __init__(
        self,
        config: Config,
        project_full_name: ProjectFullName,
        functions: List[Any],
    ) -> None:
        self._config = config
        self.project_full_name = project_full_name
        self.definitions: List[FunctionDefinition] = [
            f.get_definition() for f in functions
        ]
        self.files_hash = calculate_hash_by_definitions(self.definitions)

    def get_tracker(self) -> RunProgressTracker:
        return self._tracker

    def _apply(self, client: LayerClient) -> ApplyResult:
        for definition in self.definitions:
            register_function(client, definition, self._tracker)

        execution_plan = build_execution_plan(self.definitions)
        return ApplyResult(execution_plan=execution_plan)

    def run(self, debug: bool = False, printer: Callable[[str], Any] = print) -> Run:
        check_asset_dependencies(self.definitions)
        for definition in self.definitions:
            definition.package()
        with LayerClient(self._config.client, logger).init() as client:
            get_or_create_remote_project(client, self.project_full_name)
            with get_progress_tracker(
                url=self._config.url,
                account_name=self.project_full_name.account_name,
                project_name=self.project_full_name.project_name,
                assets=[(d.asset_type, d.asset_name) for d in self.definitions],
            ).track() as tracker:
                self._tracker = tracker
                try:
                    metadata = self._apply(client)
                    user_command = self._get_user_command(
                        execute_function=ProjectRunner.run, functions=self.definitions
                    )
                    if is_executables_feature_active():
                        asyncio.run(self._upload_executable_packages(client))
                    else:
                        ResourceManager(client).wait_resource_upload(
                            self.project_full_name, self.definitions, tracker
                        )
                    run_id = self._run(client, metadata.execution_plan, user_command)
                    run = Run(id=run_id, project_full_name=self.project_full_name)
                except LayerClientServiceUnavailableException as e:
                    raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
                try:
                    if debug:
                        run_context = RunContext(is_running=True)
                        logs_thread = threading.Thread(
                            target=show_pipeline_run_logs,
                            args=(client, run_id.value, True),
                            kwargs={
                                "printer": printer,
                                "evaluate_until_callback": run_context.user_logs_check_predicate,
                            },
                        )
                        logs_thread.start()
                    self._tracker.mark_start_running(run_id)
                    try:
                        self._poll_until_completed(client, metadata, run)
                    except (ProjectBaseException, ProjectRunnerError) as e:
                        tracker.mark_error_messages(e)
                        raise e
                    finally:
                        if debug:
                            self._finish_checking_user_logs(run_context, logs_thread)
                except LayerClientServiceUnavailableException as e:
                    raise LayerServiceUnavailableExceptionDuringExecution(
                        run_id, str(e)
                    )
        return run

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
        with open(function.executable_path, "rb") as package_file:
            presigned_url = client.executor_service_client.get_upload_path(
                project_full_name=self.project_full_name,
                function_name=function.func.__name__,
            )
            await session.put(
                presigned_url,
                data=package_file,
                timeout=aiohttp.ClientTimeout(total=None),
            )

    @staticmethod
    def _get_user_command(
        execute_function: Callable[..., Any], functions: Sequence[FunctionDefinition]
    ) -> str:
        functions_string = ", ".join(function.func_name for function in functions)
        return f"{execute_function.__name__}([{functions_string}])"

    @staticmethod
    def _finish_checking_user_logs(
        run_context: RunContext, logs_thread: threading.Thread
    ) -> None:
        run_context.run_completion_time = datetime.now()
        run_context.is_running = False
        logs_thread.join()

    def _poll_until_completed(
        self, client: LayerClient, apply_metadata: ApplyResult, run: Run
    ) -> None:
        updater = ProgressTrackerUpdater(
            tracker=self._tracker,
            apply_metadata=apply_metadata,
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

    def _run(
        self,
        client: LayerClient,
        execution_plan: ExecutionPlan,
        user_command: str,
    ) -> RunId:
        try:
            run_id = client.flow_manager.start_run(
                self.project_full_name,
                execution_plan,
                self.files_hash,
                user_command,
            )
        except LayerResourceExhaustedException as e:
            raise ProjectRunnerError(f"{e}")
        else:
            return run_id


def _register_dataset_function(
    client: LayerClient,
    dataset: FunctionDefinition,
    is_local: bool,
    tracker: RunProgressTracker,
) -> None:
    try:
        project_id = verify_project_exists_and_retrieve_project_id(
            client, dataset.project_full_name
        )
        dataset_id = client.data_catalog.add_dataset(
            dataset.asset_path,
            project_id,
            dataset.description,
            dataset.function_home_dir,
            dataset.get_fabric(is_local),
            dataset.func_source,
            dataset.entrypoint,
            dataset.environment,
        )
        dataset.set_repository_id(uuid.UUID(dataset_id))
        assert dataset.repository_id
        tracker.mark_dataset_saved(dataset.asset_name, id_=dataset.repository_id)
    except LayerClientServiceUnavailableException as e:
        tracker.mark_dataset_failed(dataset.asset_name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_dataset_failed(dataset.asset_name, "")
        raise ProjectInitializationException(
            f"Failed to save derived dataset {dataset.asset_name!r}: {e}",
            "Please retry",
        )


def _register_model_function(
    client: LayerClient,
    model: FunctionDefinition,
    is_local: bool,
    tracker: RunProgressTracker,
) -> None:
    try:
        response = client.model_catalog.create_model_version(
            model.asset_path,
            model.description,
            model.source_code_digest.hexdigest(),
            model.get_fabric(is_local),
        )
        version = response.model_version
        model.set_version_id(version.id.value)
        if response.should_upload_training_files:
            # in here we upload to path / train.gz
            version_id = uuid.UUID(version.id.value)
            s3_path = client.model_training.upload_training_files(
                model.asset_name, model.function_home_dir, version_id
            )
            # in here we reconstruct the path / train.gz to save in metadata
            client.model_catalog.store_training_metadata(
                model.asset_name,
                model.description,
                model.entrypoint,
                model.environment,
                s3_path,
                version,
                model.get_fabric(is_local),
            )

    except LayerClientServiceUnavailableException as e:
        tracker.mark_model_train_failed(model.asset_name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_model_train_failed(model.asset_name, "")
        raise ProjectInitializationException(
            f"Failed to save model {model.asset_name!r}: {e}",
            "Please retry",
        )


def register_function(
    client: LayerClient,
    func: FunctionDefinition,
    tracker: RunProgressTracker,
) -> None:
    local_fabric = func.fabric == Fabric.F_LOCAL
    function_registrator = _get_function_registrator(func.asset_type)
    function_registrator(client, func, local_fabric, tracker)


def _get_function_registrator(asset_type: AssetType) -> Callable[..., None]:
    if asset_type == AssetType.DATASET:
        return _register_dataset_function
    elif asset_type == AssetType.MODEL:
        return _register_model_function
    else:
        raise ValueError(f"unsupported asset type: {asset_type}")
