import uuid
from unittest.mock import MagicMock

from layerapi.api.entity.logged_model_metric_pb2 import LoggedModelMetric
from layerapi.api.ids_pb2 import LoggedDataId, ModelMetricId, ModelTrainId
from layerapi.api.service.logged_data.logged_data_api_pb2 import (
    LogDataRequest,
    LogDataResponse,
    LogModelMetricRequest,
    LogModelMetricResponse,
)
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType

from layer.clients.logged_data_service import ModelMetricPoint

from .util import get_logged_data_service_client_with_mocks


def test_given_tag_not_exists_when_log_text_then_calls_log_data_with_text_type():  # noqa
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000")
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    train_id = uuid.uuid4()
    tag = "test-tag"
    text = "test-text"

    # when
    logged_data_client.log_text_data(train_id=train_id, tag=tag, data=text)

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=LoggedDataType.LOGGED_DATA_TYPE_TEXT,
            text=text,
        )
    )


def test_given_tag_not_exists_when_log_table_then_calls_log_data_with_table_type():  # noqa
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000")
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    train_id = uuid.uuid4()
    tag = "table-test-tag"
    data = "{}"

    # when
    logged_data_client.log_table_data(train_id=train_id, tag=tag, data=data)

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=LoggedDataType.LOGGED_DATA_TYPE_TABLE,
            text=data,
        )
    )


def test_given_tag_not_exists_when_log_binary_then_calls_log_data_with_blob_type():  # noqa
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000"),
        s3_path="path/to/upload",
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    train_id = uuid.uuid4()
    tag = "blob-test-tag"

    # when
    s3_path = logged_data_client.log_binary_data(train_id=train_id, tag=tag)

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=LoggedDataType.LOGGED_DATA_TYPE_BLOB,
            text=None,
        )
    )
    assert s3_path == "path/to/upload"


def test_given_tag_not_exists_when_log_model_metric_then_calls_log_metric():  # noqa
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogModelMetric.return_value = LogModelMetricResponse(
        model_metric_id=ModelMetricId(value="00000000-0000-0000-0000-000000000000"),
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    train_id = uuid.uuid4()
    tag = "blob-test-tag"
    points = [ModelMetricPoint(epoch=1, value=1), ModelMetricPoint(epoch=2, value=2)]

    # when
    logged_data_client.log_model_metric(train_id=train_id, tag=tag, points=points)

    # then
    mock_logged_data_api.LogModelMetric.assert_called_with(
        request=LogModelMetricRequest(
            model_train_id=ModelTrainId(value=str(train_id)),
            metric=LoggedModelMetric(
                unique_tag=tag,
                points=[
                    LoggedModelMetric.ModelMetricPoint(epoch=1, value=1),
                    LoggedModelMetric.ModelMetricPoint(epoch=2, value=2),
                ],
            ),
        )
    )
