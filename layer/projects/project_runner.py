import logging
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import polling  # type: ignore
from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.ids_pb2 import HyperparameterTuningId, ModelVersionId, RunId

from layer.clients.layer import LayerClient
from layer.config import Config
from layer.contracts.asset import AssetType
from layer.contracts.datasets import Dataset, DerivedDataset, PythonDataset, RawDataset
from layer.contracts.projects import ApplyResult, Asset, Function, Project, ResourcePath
from layer.definitions import DatasetDefinition, ModelDefinition
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
from layer.projects.project_hash_calculator import calculate_project_hash_by_definitions
from layer.projects.util import (
    get_or_create_remote_project,
    verify_project_exists_and_retrieve_project_id,
)
from layer.resource_manager import ResourceManager
from layer.tracker.project_progress_tracker import ProjectProgressTracker
from layer.tracker.remote_execution_project_progress_tracker import (
    RemoteExecutionProjectProgressTracker,
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
    _tracker: ProjectProgressTracker
    _project_progress_tracker_factory: Type[RemoteExecutionProjectProgressTracker]

    def __init__(
        self,
        config: Config,
        project_progress_tracker_factory: Type[
            RemoteExecutionProjectProgressTracker
        ] = RemoteExecutionProjectProgressTracker,
    ) -> None:
        self._config = config
        self._project_progress_tracker_factory = project_progress_tracker_factory

    def get_tracker(self) -> ProjectProgressTracker:
        return self._tracker

    def _create_entities_and_upload_user_code(
        self, client: LayerClient, project: Project
    ) -> Tuple[Dict[str, ModelVersionId], Dict[str, HyperparameterTuningId]]:
        for raw_dataset in project.raw_datasets:
            self._save_raw_datasets(client, project.id, raw_dataset)

        for derived_dataset in project.derived_datasets:
            register_derived_datasets(
                client, project.id, derived_dataset, self._tracker
            )

        models_metadata: Dict[str, ModelVersionId] = {}
        hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId] = {}
        for model in project.models:
            model = model.with_language_version(_language_version())
            response = client.model_catalog.create_model_version(project.name, model)
            version = response.model_version
            hyperparameter_tuning_id = (
                client.model_training.create_hpt_id(version)
                if model.training.hyperparameter_tuning is not None
                else None
            )

            if response.should_upload_training_files:
                # in here we upload to path / train.gz
                client.model_training.upload_training_files(model, version.id.value)
                source_code_response = (
                    client.model_training.get_source_code_upload_credentials(
                        version.id.value
                    )
                )
                # in here we reconstruct the path / train.gz to save in metadata
                client.model_catalog.store_training_files_metadata(
                    model, source_code_response.s3_path, version, model.language_version
                )

            if hyperparameter_tuning_id is not None:
                hyperparameter_tuning_metadata[model.name] = hyperparameter_tuning_id
                source_code_response = (
                    client.model_training.get_source_code_upload_credentials(
                        version.id.value
                    )
                )
                client.model_training.store_hyperparameter_tuning_metadata(
                    model,
                    source_code_response.s3_path,
                    version.name,
                    hyperparameter_tuning_id,
                    model.language_version,
                )

            models_metadata[model.name] = version.id
            self._tracker.mark_model_saved(model.name)

        return models_metadata, hyperparameter_tuning_metadata

    def _apply(self, client: LayerClient, project: Project) -> ApplyResult:
        (
            models_metadata,
            hyperparameter_tuning_metadata,
        ) = self._create_entities_and_upload_user_code(client, project)

        execution_plan = build_execution_plan(
            project, models_metadata, hyperparameter_tuning_metadata
        )
        client.project_service_client.update_project_readme(
            project.name, project.readme
        )

        return ApplyResult(execution_plan=execution_plan).with_models_metadata(
            models_metadata, hyperparameter_tuning_metadata, execution_plan
        )

    def _save_raw_datasets(
        self,
        client: LayerClient,
        project_id: uuid.UUID,
        dataset: RawDataset,
    ) -> None:
        try:
            client.data_catalog.add_raw_dataset(project_id, dataset)
        except LayerClientServiceUnavailableException as e:
            self._tracker.mark_raw_dataset_save_failed(dataset.name, "")
            raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
        except LayerClientException as e:
            self._tracker.mark_raw_dataset_save_failed(dataset.name, "")
            raise ProjectInitializationException(
                f"Failed to save raw dataset {dataset.name!r}: {e}", "Please retry"
            )
        self._tracker.mark_raw_dataset_saved(dataset.name)

    def with_functions(self, project_name: str, functions: List[Any]) -> Project:
        derived_datasets = []
        models = []
        definitions: List[Union[DatasetDefinition, ModelDefinition]] = []
        _functions: List[Function] = []
        for f in functions:
            resource_paths = f.layer.get_paths() or []
            if f.layer.get_asset_type() == AssetType.DATASET:
                dataset = DatasetDefinition(func=f, project_name=project_name)
                definitions.append(dataset)
                derived_datasets.append(dataset.get_remote_entity())
                _functions.append(
                    Function(
                        name=f.__name__,
                        asset=Asset(type=AssetType.DATASET, name=dataset.name),
                        resource_paths={
                            ResourcePath(path=path) for path in resource_paths
                        },
                    )
                )
            elif f.layer.get_asset_type() == AssetType.MODEL:
                model = ModelDefinition(func=f, project_name=project_name)
                definitions.append(model)
                models.append(model.get_remote_entity())
                _functions.append(
                    Function(
                        name=f.__name__,
                        asset=Asset(type=AssetType.MODEL, name=model.name),
                        resource_paths={
                            ResourcePath(
                                path=path,
                            )
                            for path in resource_paths
                        },
                    )
                )
        try:
            layer_client = LayerClient(self._config.client, logger)
            with layer_client.init() as initialized_client:
                project_id = verify_project_exists_and_retrieve_project_id(
                    initialized_client, project_name
                )

        except LayerClientServiceUnavailableException as e:
            raise LayerServiceUnavailableExceptionDuringInitialization(str(e))

        project_hash = calculate_project_hash_by_definitions(definitions)  # type: ignore
        project_with_entities = (
            Project(name=project_name, _id=project_id, functions=_functions)
            .with_derived_datasets(derived_datasets=derived_datasets)
            .with_models(models=models)
            .with_files_hash(project_hash)
        )

        return project_with_entities

    def run(
        self,
        project: Project,
        debug: bool = False,
        printer: Callable[[str], Any] = print,
    ) -> RunId:
        check_entity_dependencies(project)
        with LayerClient(self._config.client, logger).init() as client:
            project = get_or_create_remote_project(client, project)
            with self._project_progress_tracker_factory(
                self._config, project
            ).track() as tracker:
                self._tracker = tracker
                try:
                    metadata = self._apply(client, project)
                    ResourceManager(client).wait_resource_upload(project, tracker)
                    user_command = self._get_user_command(
                        execute_function=ProjectRunner.run, functions=project.functions
                    )
                    run_id = self._run(
                        client, project, metadata.execution_plan, user_command
                    )
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
                        self._poll_until_completed(client, metadata, run_id)
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

        return run_id

    @staticmethod
    def _get_user_command(
        execute_function: Callable[..., Any], functions: Sequence[Function]
    ) -> str:
        functions_string = ", ".join(function.name for function in functions)
        return f"{execute_function.__name__}([{functions_string}])"

    @staticmethod
    def _finish_checking_user_logs(
        run_context: RunContext, logs_thread: threading.Thread
    ) -> None:
        run_context.run_completion_time = datetime.now()
        run_context.is_running = False
        logs_thread.join()

    def _poll_until_completed(
        self, client: LayerClient, apply_metadata: ApplyResult, run_id: RunId
    ) -> None:
        updater = ProgressTrackerUpdater(
            tracker=self._tracker,
            apply_metadata=apply_metadata,
            run_id=run_id,
            client=client,
        )

        initial_step_sec = 2
        step_function = PollingStepFunction(max_backoff_sec=3.0, backoff_multiplier=1.2)

        polling.poll(
            lambda: client.flow_manager.get_run_status_history_and_metadata(
                run_id=run_id,
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
        project: Project,
        execution_plan: ExecutionPlan,
        user_command: str,
    ) -> RunId:
        try:
            run_id = client.flow_manager.start_run(
                project.name, execution_plan, project.project_files_hash, user_command
            )
        except LayerResourceExhaustedException as e:
            raise ProjectRunnerError(f"{e}")
        else:
            return run_id

    def terminate_run(self, run_id: str) -> None:
        _run_id = RunId(value=run_id)
        with LayerClient(self._config.client, logger).init() as client:
            client.flow_manager.terminate_run(_run_id)


def register_derived_datasets(
    client: LayerClient,
    project_id: uuid.UUID,
    dataset: DerivedDataset,
    tracker: Optional[ProjectProgressTracker] = None,
) -> Dataset:
    if not tracker:
        tracker = ProjectProgressTracker()
    try:
        if isinstance(dataset, PythonDataset):
            dataset = dataset.with_language_version(_language_version())
        resp = client.data_catalog.add_dataset(project_id, dataset)
        tracker.mark_derived_dataset_saved(dataset.name, id_=resp.id)
        return resp
    except LayerClientServiceUnavailableException as e:
        tracker.mark_derived_dataset_failed(dataset.name, "")
        raise LayerServiceUnavailableExceptionDuringInitialization(str(e))
    except LayerClientException as e:
        tracker.mark_derived_dataset_failed(dataset.name, "")
        raise ProjectInitializationException(
            f"Failed to save derived dataset {dataset.name!r}: {e}",
            "Please retry",
        )


def _language_version() -> Tuple[int, int, int]:
    return sys.version_info.major, sys.version_info.minor, sys.version_info.micro
