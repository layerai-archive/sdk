import uuid
from unittest.mock import MagicMock

import pytest
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


@pytest.mark.parametrize(
    ("tag", "text", "fnc_name", "log_type"),
    [
        (
            "table-test-tag",
            "{}",
            "log_table_data",
            LoggedDataType.LOGGED_DATA_TYPE_TABLE,
        ),
        ("text-test-tag", "abc", "log_text_data", LoggedDataType.LOGGED_DATA_TYPE_TEXT),
        (
            "num-test-tag",
            "123",
            "log_numeric_data",
            LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
        ),
        (
            "bool-test-tag",
            "True",
            "log_boolean_data",
            LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN,
        ),
    ],
)
def test_given_tag_not_exists_when_log_x_then_calls_log_data_with_x_type(
    tag, text, fnc_name, log_type
):  # noqa
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000")
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    train_id = uuid.uuid4()
    # tag = "table-test-tag"
    # data = "{}"

    # when
    fnc = getattr(logged_data_client, fnc_name)
    fnc(train_id=train_id, tag=tag, data=text)

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=log_type,
            text=text,
            epoch=-1,
        )
    )


def test_given_tag_not_exists_when_log_binary_then_calls_log_data_with_image_type():  # noqa
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
    tag = "image-test-tag"
    epoch = 123

    # when
    s3_path = logged_data_client.log_binary_data(
        train_id=train_id,
        tag=tag,
        logged_data_type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        epoch=epoch,
    )

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
            text=None,
            epoch=epoch,
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
    tag = "foo-test-tag"
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
