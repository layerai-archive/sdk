import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, List, Optional
from uuid import UUID

from layerapi.api.entity.model_train_status_pb2 import ModelTrainStatus
from yarl import URL

from layer import Context
from layer.clients.layer import LayerClient
from layer.contracts.assertions import Assertion
from layer.exceptions.exception_handler import exception_handler
from layer.exceptions.exceptions import LayerFailedAssertionsException
from layer.global_context import current_project_full_name
from layer.resource_manager import ResourceManager
from layer.tracker.progress_tracker import RunProgressTracker
from layer.training.train import Train

from ...contracts.asset import AssetPath, AssetType
from ...contracts.tracker import ResourceTransferState
from ...logged_data.immediate_logged_data_destination import (
    ImmediateLoggedDataDestination,
)
from .common import import_function, update_train_status
from .model_train_failure_reporter import ModelTrainFailureReporter


@dataclass
class TrainContextDataclassMixin:
    model_name: str
    model_version: str
    train_id: UUID
    source_entrypoint: str
    source_folder: Path
    logger: Logger
    train_index: Optional[str] = None


class TrainContext(ABC, TrainContextDataclassMixin):
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
        pass


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
    url: URL
    client: LayerClient
    train_context: TrainContext
    logger: Logger
    failure_reporter: ModelTrainFailureReporter
    tracker: RunProgressTracker
    args: Any = None
    kwargs: Any = None

    def train(self) -> Any:
        self.tracker.mark_model_training(
            self.train_context.model_name,
            version=self.train_context.model_version,
            train_idx=self.train_context.train_index,
        )
        try:
            with self.train_context:
                return self._train(callback=self.failure_reporter.report_failure)
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
        self.logger.info(
            f"Importing user code({self.train_context.source_entrypoint}) from {self.train_context.source_folder}"
        )
        train_model_func = import_function(
            self.train_context.source_folder,
            self.train_context.source_entrypoint,
            "train_model",
        )
        self.logger.info("train_model_func function imported successfully")
        project_full_name = current_project_full_name()
        if not project_full_name:
            raise Exception("Internal Error: missing current project full name")

        with Train(
            layer_client=self.client,
            name=self.train_context.model_name,
            project_full_name=project_full_name,
            version=self.train_context.model_version,
            train_id=self.train_context.train_id,
            train_index=str(self.train_context.train_index),
        ) as train, ImmediateLoggedDataDestination(
            self.client.logged_data_service_client
        ) as logged_data_dest:
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
                logged_data_destination=logged_data_dest,
            ):
                update_train_status(
                    self.client.model_catalog,
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_FETCHING_FEATURES,
                    self.logger,
                )
                update_train_status(
                    self.client.model_catalog,
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
                args = self.args or []
                kwargs = self.kwargs or {}
                model = train_model_func(*args, **kwargs)
                self.tracker.mark_model_trained(self.train_context.model_name)
                self.logger.info("Executed train_model_func successfully")
                if model:
                    self._run_assertions(
                        model,
                        train_model_func.layer.get_assertions(),  # type: ignore
                    )
                    self.tracker.mark_model_saving(self.train_context.model_name)
                    self.logger.info(f"Saving model artifact {model} to model registry")
                    transfer_state = ResourceTransferState()
                    train.save_model(model, transfer_state=transfer_state)
                    self.tracker.mark_model_saving_result(model.name, transfer_state)
                    self.logger.info(
                        f"Saved model artifact {model} to model registry successfully"
                    )
                update_train_status(
                    self.client.model_catalog,
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_SUCCESSFUL,
                    self.logger,
                )
                self.tracker.mark_model_saved(
                    self.train_context.model_name,
                    warnings=logged_data_dest.close_and_get_errors(),
                )
                return model
