import logging
import uuid
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

from layer.api.entity.model_train_status_pb2 import ModelTrainStatus
from layer.api.ids_pb2 import ModelTrainId
from layer.client import ModelCatalogClient
from layer.common import LayerClient
from layer.exceptions.status_report import (
    ExecutionStatusReportFactory,
    PythonExecutionStatusReport,
)
from layer.training.runtime import ModelTrainFailureReporter


logger = logging.getLogger(__name__)


class TestModelTrainFailureReporter:
    def test_given_model_failure_already_reported_when_failed_then_does_not_reports_again(
        self,
    ) -> None:
        # given
        model_source_folder = Path(__file__)
        train_id = uuid.uuid4()
        client_mock = MagicMock(
            spec_set=LayerClient, model_catalog=MagicMock(spec_set=ModelCatalogClient)
        )
        failure_reporter = ModelTrainFailureReporter(
            client_mock.model_catalog, logger, train_id, model_source_folder
        )
        train_status = PropertyMock(return_value=ModelTrainStatus.TRAIN_STATUS_FAILED)
        type(
            client_mock.model_catalog.get_model_train().train_status
        ).train_status = train_status

        # when
        failure_reporter.report_failure("test", Exception("test"))

        # then
        client_mock.model_catalog.update_model_train_status.assert_not_called()

    def test_given_model_train_in_progress_when_model_failed_then_reports_failure(
        self,
    ) -> None:
        # given
        model_source_folder = Path(__file__)
        train_id = uuid.uuid4()
        client_mock = MagicMock(
            spec_set=LayerClient, model_catalog=MagicMock(spec_set=ModelCatalogClient)
        )
        failure_reporter = ModelTrainFailureReporter(
            client_mock.model_catalog, logger, train_id, model_source_folder
        )
        train_status = PropertyMock(
            return_value=ModelTrainStatus.TRAIN_STATUS_IN_PROGRESS
        )
        type(
            client_mock.model_catalog.get_model_train().train_status
        ).train_status = train_status
        exception = Exception("test")

        # when
        failure_reporter.report_failure("test", exception)

        # then
        expected_info = ExecutionStatusReportFactory.to_json(
            PythonExecutionStatusReport.from_exception(exception, model_source_folder)
        )
        client_mock.model_catalog.update_model_train_status.assert_any_call(
            ModelTrainId(value=str(train_id)),
            ModelTrainStatus(
                train_status=ModelTrainStatus.TRAIN_STATUS_FAILED, info=expected_info
            ),
        )
