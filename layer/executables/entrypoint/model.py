import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, List, Mapping, Optional, Sequence
from uuid import UUID

from layerapi.api.entity.model_train_status_pb2 import ModelTrainStatus
from layerapi.api.entity.task_pb2 import Task
from layerapi.api.ids_pb2 import ModelTrainId, RunId
from yarl import URL

from layer import Context
from layer.clients.layer import LayerClient
from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.exceptions.exception_handler import exception_handler
from layer.exceptions.exceptions import LayerFailedAssertionsException
from layer.exceptions.status_report import (
    ExecutionStatusReport,
    ExecutionStatusReportFactory,
    PythonExecutionStatusReport,
)
from layer.global_context import current_project_full_name
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
from layer.resource_manager import ResourceManager
from layer.tracker.progress_tracker import RunProgressTracker
from layer.training.train import Train

from .common import make_runner


logger = logging.getLogger(__name__)


def _run(
    url: URL,
    model_definition: FunctionDefinition,
    client: LayerClient,
    tracker: RunProgressTracker,
    fabric: Fabric,
    run_id: str,
    **kwargs: Any,
) -> None:

    verify_project_exists_and_retrieve_project_id(
        client, model_definition.project_full_name
    )

    model_version = client.model_catalog.create_model_version(
        model_definition.asset_path,
        model_definition.description,
        model_definition.source_code_digest,
        fabric.value,
    ).model_version
    train_id = client.model_catalog.create_model_train_from_version_id(model_version.id)
    if run_id:
        client.flow_manager.update_run_metadata(
            run_id=RunId(value=run_id),
            task_id=model_definition.asset_path.path(),
            task_type=Task.Type.TYPE_MODEL_TRAIN,
            key="train-id",
            value=str(train_id.value),
        )
    train = client.model_catalog.get_model_train(train_id)

    context = TrainContext(
        model_name=model_definition.asset_name,
        model_version=model_version.name,
        train_id=UUID(train_id.value),
        function=model_definition.func,
        args=model_definition.args,
        kwargs=model_definition.kwargs,
        assertions=model_definition.assertions,
        train_index=str(train.index),
    )
    trainer = ModelTrainer(
        url=url,
        client=client,
        train_context=context,
        tracker=tracker,
    )
    result = trainer.train()

    tracker.mark_model_trained(
        name=model_definition.asset_name,
        train_index=str(train.index),
        version=model_version.name,
    )

    return result


RUNNER = make_runner(_run)


@dataclass
class TrainContext:
    model_name: str
    model_version: str
    train_id: UUID
    function: Callable[..., Any]
    args: Sequence[Any]
    kwargs: Mapping[str, Any]
    assertions: List[Assertion]
    train_index: Optional[str] = None
    initial_cwd: Optional[str] = None

    def __enter__(self) -> None:
        self.initial_cwd = os.getcwd()

    def get_working_directory(self) -> Path:
        assert self.initial_cwd
        return Path(self.initial_cwd)

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        assert self.initial_cwd
        os.chdir(
            self.initial_cwd
        )  # Important for local execution to have no such side effect
        return None


@dataclass(frozen=True)
class ModelTrainer:
    url: URL
    client: LayerClient
    train_context: TrainContext
    tracker: RunProgressTracker

    def train(self) -> Any:
        self.tracker.mark_model_training(
            self.train_context.model_name,
            version=self.train_context.model_version,
            train_idx=self.train_context.train_index,
        )
        try:
            with self.train_context:
                return self._train(callback=self._report_failure)
        except Exception:
            logger.error(
                f"Error performing model training with id: {self.train_context.train_id}",
                exc_info=True,
            )
            import sys
            import traceback

            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

    def _run_assertions(self, model: Any) -> None:
        failed_assertions = []
        self.tracker.mark_model_running_assertions(self.train_context.model_name)
        for assertion in reversed(self.train_context.assertions):
            try:
                self.tracker.mark_model_running_assertion(
                    self.train_context.model_name, assertion
                )
                assertion.function(model)
            except Exception:
                failed_assertions.append(assertion)
        if len(failed_assertions) > 0:
            self.tracker.mark_model_failed_assertions(
                self.train_context.model_name, failed_assertions
            )
            raise LayerFailedAssertionsException(failed_assertions)
        else:
            self.tracker.mark_model_completed_assertions(self.train_context.model_name)

    @exception_handler(stage="Training run")
    def _train(
        self, callback: Optional[Callable[[str, Exception], None]] = None
    ) -> Any:
        train_model_func = self.train_context.function
        project_full_name = current_project_full_name()
        if not project_full_name:
            raise Exception("Internal Error: missing current project full name")
        with Train(
            layer_client=self.client,
            name=self.train_context.model_name,
            project_full_name=project_full_name,
            version=self.train_context.model_version,
            train_id=self.train_context.train_id,
            train_index=self.train_context.train_index,
        ) as train:
            asset_path = AssetPath(
                account_name=project_full_name.account_name,
                project_name=project_full_name.project_name,
                asset_type=AssetType.MODEL,
                asset_name=self.train_context.model_name,
            )
            with Context(
                url=self.url,
                asset_path=asset_path,
                train=train,
                tracker=self.tracker,
            ):
                self._update_train_status(
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_IN_PROGRESS,
                )
                logger.info("Executing the train_model_func")
                work_dir = self.train_context.get_working_directory()
                os.chdir(work_dir)
                logger.info("Downloading resources")
                ResourceManager(self.client).wait_resource_download(
                    project_full_name,
                    train_model_func.__name__,
                    target_dir=str(work_dir),
                )
                model = train_model_func(
                    *self.train_context.args, **self.train_context.kwargs
                )
                self.tracker.mark_model_trained(
                    self.train_context.model_name,
                    version=train.get_version(),
                    train_index=train.get_train_index(),
                )
                logger.info("Executed train_model_func successfully")
                if model:
                    self._run_assertions(model)
                    self.tracker.mark_model_saving(self.train_context.model_name)
                    logger.info(f"Saving model artifact {model} to model registry")
                    train.save_model(model, tracker=self.tracker)
                    logger.info(
                        f"Saved model artifact {model} to model registry successfully"
                    )
                self._update_train_status(
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_SUCCESSFUL,
                )
                self.tracker.mark_model_saved(self.train_context.model_name)
                return model

    def _report_failure(self, stage: str, failure_exc: Exception) -> None:
        # Check to only keep cause for inner most exception, as __exit__
        # catches SystemExit exceptions, thus without this if a chain of status updates
        # could be triggered with the outer most exception message overriding inner ones
        existing_status = self._get_train_status()
        if existing_status != ModelTrainStatus.TRAIN_STATUS_FAILED:
            report: ExecutionStatusReport
            if isinstance(failure_exc, LayerFailedAssertionsException):
                report = failure_exc.to_status_report()
            else:
                report = PythonExecutionStatusReport.from_exception(failure_exc, None)
            self._update_train_status(
                self.train_context.train_id,
                ModelTrainStatus.TRAIN_STATUS_FAILED,
                info=ExecutionStatusReportFactory.to_json(report),
            )

    def _get_train_status(self) -> "ModelTrainStatus.TrainStatus.V":
        return self.client.model_catalog.get_model_train(
            ModelTrainId(value=str(self.train_context.train_id))
        ).train_status.train_status

    def _update_train_status(
        self,
        train_id: UUID,
        train_status: "ModelTrainStatus.TrainStatus.V",
        info: str = "",
    ) -> None:
        try:
            self.client.model_catalog.update_model_train_status(
                ModelTrainId(value=str(train_id)),
                ModelTrainStatus(train_status=train_status, info=info),
            )
        except Exception as e:
            reason = getattr(e, "message", repr(e))
            logger.error(
                f"Failure while trying to update the status of train ID {str(train_id)}: {reason}"
            )
