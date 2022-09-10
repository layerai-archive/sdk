import logging
from typing import Any, Dict, Tuple

from layer import Context
from layer.contracts.models import ModelTrain, ModelTrainStatus
from layer.contracts.runs import TaskType
from layer.exceptions.exceptions import LayerFailedAssertionsException
from layer.exceptions.status_report import (
    ExecutionStatusReport,
    ExecutionStatusReportFactory,
    PythonExecutionStatusReport,
)

from .common import FunctionRunner


logger = logging.getLogger(__name__)


class ModelRunner(FunctionRunner):
    train: ModelTrain

    def _create_asset(self) -> Tuple[str, Dict[str, Any]]:
        model_version = self.client.model_catalog.create_model_version(
            self.definition.asset_path,
            self.definition.description,
            self.definition.source_code_digest,
            self.fabric.value,
        )
        train_id = self.client.model_catalog.create_model_train_from_version_id(
            model_version.id
        )
        self.train = self.client.model_catalog.get_model_train(train_id)
        if self.run_id:
            self.client.flow_manager.update_run_metadata(
                run_id=self.run_id,
                task_id=self.definition.asset_path.path(),
                task_type=TaskType.MODEL_TRAIN,
                key="train-id",
                value=str(train_id),
            )

        return self.train.tag, {"model_train": self.train}

    def _mark_start(self) -> None:
        self.client.model_catalog.start_model_train(
            train_id=self.train.id,
        )
        self._update_train_status(ModelTrainStatus.IN_PROGRESS)

    def _mark_success(self) -> str:
        self._update_train_status(ModelTrainStatus.SUCCESSFUL)
        return self.train.tag

    def _mark_failure(self, failure_exc: Exception) -> str:
        self._update_train_status(ModelTrainStatus.FAILED)
        report: ExecutionStatusReport
        if isinstance(failure_exc, LayerFailedAssertionsException):
            report = failure_exc.to_status_report()
        else:
            report = PythonExecutionStatusReport.from_exception(failure_exc, None)
        self._update_train_status(
            ModelTrainStatus.FAILED,
            info=ExecutionStatusReportFactory.to_json(report),
        )
        return self.train.tag

    def _save_artifact(self, ctx: Context, artifact: Any) -> None:
        logger.info(f"Saving model artifact {artifact} to model registry")
        ctx.save_model(artifact)
        logger.info(f"Saved model artifact {artifact} to model registry successfully")

    def _update_train_status(
        self, train_status: ModelTrainStatus, info: str = ""
    ) -> None:
        try:
            self.client.model_catalog.update_model_train_status(
                train_id=self.train.id,
                train_status=train_status,
                info=info,
            )
        except Exception as e:
            reason = getattr(e, "message", repr(e))
            logger.error(
                f"Failure while trying to update the status of train ID {str(self.train.id)}: {reason}"
            )
