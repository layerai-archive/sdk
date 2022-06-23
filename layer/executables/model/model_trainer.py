import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, List, Optional
from uuid import UUID

from layerapi.api.entity.model_train_status_pb2 import ModelTrainStatus
from layerapi.api.ids_pb2 import ModelTrainId

from layer import Context
from layer.clients.layer import LayerClient
from layer.contracts.assertions import Assertion
from layer.exceptions.exception_handler import exception_handler
from layer.exceptions.exceptions import LayerFailedAssertionsException
from layer.exceptions.status_report import (
    ExecutionStatusReport,
    ExecutionStatusReportFactory,
    PythonExecutionStatusReport,
)
from layer.global_context import (
    current_project_full_name,
    reset_active_context,
    set_active_context,
)
from layer.resource_manager import ResourceManager
from layer.tracker.progress_tracker import RunProgressTracker
from layer.training.train import Train


@dataclass
class TrainContextDataclassMixin:
    model_name: str
    model_version: str
    train_id: UUID
    function: Callable[..., Any]
    logger: Logger
    train_index: Optional[str] = None


class TrainContext(ABC, TrainContextDataclassMixin):
    def init_or_save_context(self, context: Context) -> None:
        set_active_context(context)

    @abstractmethod
    def __enter__(self) -> None:
        pass

    @abstractmethod
    def get_working_directory(self) -> Path:
        pass

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        reset_active_context()


@dataclass
class LocalTrainContext(TrainContext):
    initial_cwd: Optional[str] = None

    def __enter__(self) -> None:
        super().__enter__()
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
        super().__exit__(exc_type, exc_val, exc_tb)
        assert self.initial_cwd
        os.chdir(
            self.initial_cwd
        )  # Important for local execution to have no such side effect
        return None


@dataclass(frozen=True)
class ModelTrainer:
    client: LayerClient
    train_context: TrainContext
    logger: Logger
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
            self.logger.error(
                f"Error performing model training with id: {self.train_context.train_id}",
                exc_info=True,
            )
            import sys
            import traceback

            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

    def _run_assertions(self, model: Any, assertions: List[Assertion]) -> None:
        failed_assertions = []
        self.tracker.mark_model_running_assertions(self.train_context.model_name)
        for assertion in reversed(assertions):
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
        with Context() as context:
            with Train(
                layer_client=self.client,
                name=self.train_context.model_name,
                project_full_name=project_full_name,
                version=self.train_context.model_version,
                train_id=self.train_context.train_id,
            ) as train:
                context.with_train(train)
                context.with_tracker(self.tracker)
                context.with_asset_name(self.train_context.model_name)
                self.train_context.init_or_save_context(context)
                self._update_train_status(
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_IN_PROGRESS,
                    self.logger,
                )
                self.logger.info("Executing the train_model_func")
                work_dir = self.train_context.get_working_directory()
                os.chdir(work_dir)
                self.logger.info("Downloading resources")
                ResourceManager(self.client).wait_resource_download(
                    project_full_name,
                    train_model_func.__name__,
                    target_dir=str(work_dir),
                )
                model = train_model_func()
                self.tracker.mark_model_trained(
                    self.train_context.model_name,
                    version=train.get_version(),
                    train_index=train.get_train_index(),
                )
                self.logger.info("Executed train_model_func successfully")
                self._run_assertions(
                    model,
                    train_model_func.layer.get_assertions(),  # type: ignore
                )
                self.tracker.mark_model_saving(self.train_context.model_name)
                self.logger.info(f"Saving model artifact {model} to model registry")
                train.save_model(model, tracker=self.tracker)
                self._update_train_status(
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_SUCCESSFUL,
                    self.logger,
                )
                self.logger.info(
                    f"Saved model artifact {model} to model registry successfully"
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
                self.logger,
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
        logger: Logger,
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
