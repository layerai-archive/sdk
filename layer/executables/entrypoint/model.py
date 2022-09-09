import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence

from yarl import URL

from layer import Context
from layer.clients.layer import LayerClient
from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.contracts.models import ModelTrain, ModelTrainStatus
from layer.contracts.runs import TaskType
from layer.exceptions.exception_handler import exception_handler
from layer.exceptions.exceptions import LayerFailedAssertionsException
from layer.exceptions.status_report import (
    ExecutionStatusReport,
    ExecutionStatusReportFactory,
    PythonExecutionStatusReport,
)
from layer.logged_data.logged_data_destination import LoggedDataDestination
from layer.logged_data.system_metrics import SystemMetrics
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
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
    run_id: uuid.UUID,
    logged_data_destination: LoggedDataDestination,
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
    )
    train_id = client.model_catalog.create_model_train_from_version_id(model_version.id)
    if run_id:
        client.flow_manager.update_run_metadata(
            run_id=run_id,
            task_id=model_definition.asset_path.path(),
            task_type=TaskType.MODEL_TRAIN,
            key="train-id",
            value=str(train_id),
        )
    model_train = client.model_catalog.get_model_train(train_id)
    train = Train(
        layer_client=client,
        project_full_name=model_definition.project_full_name,
        name=model_definition.asset_name,
        version=model_version.name,
        train_id=train_id,
        train_index=str(model_train.index),
    )
    trainer = ModelTrainer(
        url=url,
        client=client,
        train=train,
        model_train=model_train,
        function=model_definition.func,
        args=model_definition.args,
        kwargs=model_definition.kwargs,
        assertions=model_definition.assertions,
        tracker=tracker,
        logged_data_destination=logged_data_destination,
    )
    result = trainer.run_train()

    tracker.mark_done(
        AssetType.MODEL,
        name=model_definition.asset_name,
        tag=model_train.tag,
    )

    return result


@dataclass(frozen=True)
class ModelTrainer:
    url: URL
    client: LayerClient
    tracker: RunProgressTracker
    train: Train
    model_train: ModelTrain
    function: Callable[..., Any]
    args: Sequence[Any]
    kwargs: Mapping[str, Any]
    assertions: List[Assertion]
    logged_data_destination: LoggedDataDestination

    def run_train(self) -> Any:
        self.tracker.mark_running(
            AssetType.MODEL,
            self.train.get_name(),
            tag=f"{self.train.get_version()}.{self.train.get_train_index()}",
        )
        try:
            return self._train(callback=self._report_failure)
        except Exception:
            logger.error(
                f"Error performing model training with id: {self.train.get_id()}",
                exc_info=True,
            )
            import sys
            import traceback

            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

    def _run_assertions(self, model: Any) -> None:
        failed_assertions = []
        self.tracker.mark_asserting(AssetType.MODEL, self.train.get_name())
        for assertion in reversed(self.assertions):
            try:
                self.tracker.mark_asserting(
                    AssetType.MODEL, self.train.get_name(), assertion=assertion
                )
                assertion.function(model)
            except Exception:
                failed_assertions.append(assertion)
        if len(failed_assertions) > 0:
            self.tracker.mark_failed_assertions(
                AssetType.MODEL, self.train.get_name(), failed_assertions
            )
            raise LayerFailedAssertionsException(failed_assertions)
        else:
            self.tracker.mark_asserted(AssetType.MODEL, self.train.get_name())

    @exception_handler(stage="Training run")
    def _train(
        self, callback: Optional[Callable[[str, Exception], None]] = None
    ) -> Any:
        model_name = self.train.get_name()
        train_model_func = self.function
        project_full_name = self.train.get_project_full_name()
        with self.train:
            asset_path = AssetPath(
                account_name=project_full_name.account_name,
                project_name=project_full_name.project_name,
                asset_type=AssetType.MODEL,
                asset_name=model_name,
            )
            with Context(
                url=self.url,
                client=self.client,
                asset_path=asset_path,
                model_train=self.model_train,
                tracker=self.tracker,
                logged_data_destination=self.logged_data_destination,
            ) as ctx:
                self._update_train_status(
                    ModelTrainStatus.IN_PROGRESS,
                )
                logger.info("Executing the train_model_func")
                work_dir = ctx.get_working_directory()
                os.chdir(work_dir)
                with SystemMetrics(logger):
                    model = train_model_func(*self.args, **self.kwargs)
                self.tracker.mark_done(
                    AssetType.MODEL,
                    model_name,
                    tag=f"{self.train.get_version()}.{self.train.get_train_index()}",
                )
                logger.info("Executed train_model_func successfully")
                if model:
                    self._run_assertions(model)
                    logger.info(f"Saving model artifact {model} to model registry")
                    ctx.save_model(model)
                    logger.info(
                        f"Saved model artifact {model} to model registry successfully"
                    )
                self._update_train_status(
                    ModelTrainStatus.SUCCESSFUL,
                )
                return model

    def _report_failure(self, stage: str, failure_exc: Exception) -> None:
        # Check to only keep cause for inner most exception, as __exit__
        # catches SystemExit exceptions, thus without this if a chain of status updates
        # could be triggered with the outer most exception message overriding inner ones
        existing_status = self._get_train_status()
        if existing_status != ModelTrainStatus.FAILED:
            report: ExecutionStatusReport
            if isinstance(failure_exc, LayerFailedAssertionsException):
                report = failure_exc.to_status_report()
            else:
                report = PythonExecutionStatusReport.from_exception(failure_exc, None)
            self._update_train_status(
                ModelTrainStatus.FAILED,
                info=ExecutionStatusReportFactory.to_json(report),
            )

    def _get_train_status(self) -> ModelTrainStatus:
        return self.client.model_catalog.get_model_train(self.model_train.id).status

    def _update_train_status(
        self,
        train_status: ModelTrainStatus,
        info: str = "",
    ) -> None:
        train_id = self.train.get_id()
        try:
            self.client.model_catalog.update_model_train_status(
                train_id, train_status, info
            )
        except Exception as e:
            reason = getattr(e, "message", repr(e))
            logger.error(
                f"Failure while trying to update the status of train ID {str(train_id)}: {reason}"
            )


RUNNER = make_runner(_run)
