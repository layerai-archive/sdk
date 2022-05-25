import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Type

import polling  # type: ignore
from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.ids_pb2 import RunId

from layer.clients.layer import LayerClient
from layer.config import Config
from layer.contracts.asset import AssetType
from layer.contracts.projects import ApplyResult
from layer.contracts.runs import (
    DatasetFunctionDefinition,
    FunctionDefinition,
    ModelFunctionDefinition,
    Run,
)
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
    check_entity_dependencies,
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
from layer.settings import LayerSettings
from layer.tracker.project_progress_tracker import RunProgressTracker
from layer.tracker.remote_execution_project_progress_tracker import (
    RemoteExecutionRunProgressTracker,
)
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
    _project_progress_tracker_factory: Type[RemoteExecutionRunProgressTracker]

    def __init__(
        self,
        config: Config,
        project_progress_tracker_factory: Type[
            RemoteExecutionRunProgressTracker
        ] = RemoteExecutionRunProgressTracker,
    ) -> None:
        self._config = config
        self._project_progress_tracker_factory = project_progress_tracker_factory

    def get_tracker(self) -> RunProgressTracker:
        return self._tracker

    def _apply(self, client: LayerClient, run: Run) -> ApplyResult:
        updated_definitions: List[FunctionDefinition] = []
        for definition in run.definitions:
            if isinstance(definition, DatasetFunctionDefinition):
                definition = register_dataset_function(
                    client, run.project_id, definition, False, self._tracker
                )
            elif isinstance(definition, ModelFunctionDefinition):
                definition = register_model_function(
                    client, run.project_name, definition, False, self._tracker
                )
            updated_definitions.append(definition)
        run = run.with_definitions(updated_definitions)

        execution_plan = build_execution_plan(run)
        client.project_service_client.update_project_readme(
            run.project_name, run.readme
        )
        return ApplyResult(execution_plan=execution_plan)

    def with_functions(self, project_name: str, functions: List[Any]) -> Run:
        definitions: List[FunctionDefinition] = []
        for f in functions:
            layer_settings: LayerSettings = f.layer
            if layer_settings.get_asset_type() == AssetType.DATASET:
                dataset = DatasetFunctionDefinition(
                    func=f,
                    project_name=project_name,
                )
                definitions.append(dataset)
            elif layer_settings.get_asset_type() == AssetType.MODEL:
                model = ModelFunctionDefinition(
                    func=f,
                    project_name=project_name,
                )
                definitions.append(model)
        try:
            layer_client = LayerClient(self._config.client, logger)
            with layer_client.init() as client:
                project_id = verify_project_exists_and_retrieve_project_id(
                    client, project_name
                )

        except LayerClientServiceUnavailableException as e:
            raise LayerServiceUnavailableExceptionDuringInitialization(str(e))

        return Run(
            project_id=project_id,
            project_name=project_name,
            definitions=definitions,
            files_hash=calculate_hash_by_definitions(definitions),
        )

    def run(
        self,
        run: Run,
        debug: bool = False,
        printer: Callable[[str], Any] = print,
    ) -> Run:
        check_entity_dependencies(run.definitions)
        with LayerClient(self._config.client, logger).init() as client:
            project = get_or_create_remote_project(client, run.project_name)
            assert project.account
            run = run.with_account(project.account)
            with self._project_progress_tracker_factory(
                self._config, run
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
                run.project_name, execution_plan, run.files_hash, user_command
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
    project_id: uuid.UUID,
    dataset: DatasetFunctionDefinition,
    is_local: bool,
    tracker: Optional[RunProgressTracker] = None,
) -> DatasetFunctionDefinition:
    if not tracker:
        tracker = RunProgressTracker()
    try:
        dataset = client.data_catalog.add_dataset(project_id, dataset, is_local)
        assert dataset.repository_id
        tracker.mark_derived_dataset_saved(dataset.name, id_=dataset.repository_id)
        return dataset
    except LayerClientServiceUnavailableException as e:
        tracker.mark_derived_dataset_failed(dataset.name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_derived_dataset_failed(dataset.name, "")
        raise ProjectInitializationException(
            f"Failed to save derived dataset {dataset.name!r}: {e}",
            "Please retry",
        )


def register_model_function(
    client: LayerClient,
    project_name: str,
    model: ModelFunctionDefinition,
    is_local: bool,
    tracker: Optional[RunProgressTracker] = None,
) -> ModelFunctionDefinition:
    if not tracker:
        tracker = RunProgressTracker()

    try:
        response = client.model_catalog.create_model_version(
            project_name, model, is_local
        )
        version = response.model_version
        if response.should_upload_training_files:
            # in here we upload to path / train.gz
            client.model_training.upload_training_files(model, version.id.value)
            source_code_response = (
                client.model_training.get_source_code_upload_credentials(
                    version.id.value
                )
            )
            # in here we reconstruct the path / train.gz to save in metadata
            client.model_catalog.store_training_metadata(
                model, source_code_response.s3_path, version, False
            )

        tracker.mark_model_saved(model.name)
        return model.with_version_id(version.id.value)
    except LayerClientServiceUnavailableException as e:
        tracker.mark_model_train_failed(model.name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_model_train_failed(model.name, "")
        raise ProjectInitializationException(
            f"Failed to save model {model.name!r}: {e}",
            "Please retry",
        )
