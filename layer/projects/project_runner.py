import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence

import polling  # type: ignore
from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.ids_pb2 import RunId

from layer.clients.layer import LayerClient
from layer.config import Config
from layer.contracts.assets import AssetType
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import ApplyResult
from layer.contracts.runs import FunctionDefinition, Run
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
    ) -> None:
        self._config = config

    def get_tracker(self) -> RunProgressTracker:
        return self._tracker

    def _apply(self, client: LayerClient, run: Run) -> ApplyResult:
        for definition in run.definitions:
            if definition.asset_type == AssetType.DATASET:
                register_dataset_function(client, definition, False, self._tracker)
            if definition.asset_type == AssetType.MODEL:
                register_model_function(client, definition, False, self._tracker)

        execution_plan = build_execution_plan(run)
        client.project_service_client.update_project_readme(
            run.project_full_name, run.readme
        )
        return ApplyResult(execution_plan=execution_plan)

    @staticmethod
    def with_functions(project_full_name: ProjectFullName, functions: List[Any]) -> Run:
        definitions: List[FunctionDefinition] = [f.get_definition() for f in functions]

        return Run(
            project_full_name=project_full_name,
            definitions=definitions,
            files_hash=calculate_hash_by_definitions(definitions),
        )

    def run(
        self,
        run: Run,
        debug: bool = False,
        printer: Callable[[str], Any] = print,
    ) -> Run:
        check_asset_dependencies(run.definitions)
        for definition in run.definitions:
            definition.package()
        with LayerClient(self._config.client, logger).init() as client:
            get_or_create_remote_project(client, run.project_full_name)
            with RunProgressTracker(
                url=self._config.url,
                account_name=run.project_full_name.account_name,
                project_name=run.project_full_name.project_name,
                assets=[(d.asset_type, d.asset_name) for d in run.definitions],
            ).track() as tracker:
                self._tracker = tracker
                try:
                    metadata = self._apply(client, run)
                    ResourceManager(client).wait_resource_upload(run, tracker)
                    user_command = self._get_user_command(
                        execute_function=ProjectRunner.run, functions=run.definitions
                    )
                    run_id = self._run(
                        client, run, metadata.execution_plan, user_command
                    )
                    run = run.with_run_id(run_id)
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
            client=client,
        )

        initial_step_sec = 2
        step_function = PollingStepFunction(max_backoff_sec=3.0, backoff_multiplier=1.2)

        assert run.run_id
        polling.poll(
            lambda: client.flow_manager.get_run_status_history_and_metadata(
                run_id=run.run_id,
            ),
            check_success=updater.check_completion_and_update_tracker,
            step=initial_step_sec,
            step_function=step_function.step,
            ignore_exceptions=(LayerClientTimeoutException,),
            poll_forever=True,
        )

    @staticmethod
    def _run(
        client: LayerClient,
        run: Run,
        execution_plan: ExecutionPlan,
        user_command: str,
    ) -> RunId:
        try:
            run_id = client.flow_manager.start_run(
                run.project_full_name, execution_plan, run.files_hash, user_command
            )
        except LayerResourceExhaustedException as e:
            raise ProjectRunnerError(f"{e}")
        else:
            return run_id

    def terminate_run(self, run_id: str) -> None:
        _run_id = RunId(value=run_id)
        with LayerClient(self._config.client, logger).init() as client:
            client.flow_manager.terminate_run(_run_id)


def register_dataset_function(
    client: LayerClient,
    dataset: FunctionDefinition,
    is_local: bool,
    tracker: RunProgressTracker,
) -> None:
    try:
        project_id = verify_project_exists_and_retrieve_project_id(
            client, dataset.project_full_name
        )
        dataset_id = client.data_catalog.add_dataset(project_id, dataset, is_local)
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


def register_model_function(
    client: LayerClient,
    model: FunctionDefinition,
    is_local: bool,
    tracker: RunProgressTracker,
) -> None:
    try:
        response = client.model_catalog.create_model_version(
            model.project_full_name, model, is_local
        )
        version = response.model_version
        model.set_version_id(version.id.value)
        if response.should_upload_training_files:
            # in here we upload to path / train.gz
            version_id = uuid.UUID(version.id.value)
            s3_path = client.model_training.upload_training_files(model, version_id)
            # in here we reconstruct the path / train.gz to save in metadata
            client.model_catalog.store_training_metadata(model, s3_path, version)
        tracker.mark_model_saved(model.asset_name)
    except LayerClientServiceUnavailableException as e:
        tracker.mark_model_train_failed(model.asset_name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_model_train_failed(model.asset_name, "")
        raise ProjectInitializationException(
            f"Failed to save model {model.asset_name!r}: {e}",
            "Please retry",
        )
