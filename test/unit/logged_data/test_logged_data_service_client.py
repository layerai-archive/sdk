import uuid
from unittest.mock import MagicMock

import pytest
from layerapi.api.ids_pb2 import LoggedDataId, ModelTrainId, RunId
from layerapi.api.service.logged_data.logged_data_api_pb2 import (
    LogDataRequest,
    LogDataResponse,
)
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType
from layerapi.api.value.logged_data_x_coordinate_type_pb2 import (
    LoggedDataXCoordinateType,
)

from layer.contracts.logged_data import XCoordinateType

from .util import get_logged_data_service_client_with_mocks


@pytest.mark.parametrize(
    ("tag", "value", "log_type"),
    [
        (
            "table-test-tag",
            "{}",
            LoggedDataType.LOGGED_DATA_TYPE_TABLE,
        ),
        ("text-test-tag", "abc", LoggedDataType.LOGGED_DATA_TYPE_TEXT),
        (
            "num-test-tag",
            "123",
            LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
        ),
        (
            "bool-test-tag",
            "True",
            LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN,
        ),
    ],
)
def test_given_tag_not_exists_when_log_x_then_calls_log_data_with_x_type(
    tag: str, value: str, log_type: "LoggedDataType.V"
) -> None:
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000")
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    run_id = uuid.uuid4()
    train_id = uuid.uuid4()
    # tag = "table-test-tag"
    # data = "{}"

    # when
    logged_data_client.log_data(
        run_id=run_id, train_id=train_id, type=log_type, tag=tag, value=value
    )

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            run_id=RunId(value=str(run_id)),
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=log_type,
            value=value,
            x_coordinate=-1,
        )
    )


def test_given_tag_not_exists_when_log_binary_then_calls_log_data_with_image_type() -> None:
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000"),
        s3_path="path/to/upload",
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    run_id = uuid.uuid4()
    train_id = uuid.uuid4()
    tag = "image-test-tag"
    x_coord = 123

    # when
    s3_path = logged_data_client.log_data(
        run_id=run_id,
        train_id=train_id,
        tag=tag,
        type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
        x_coordinate=x_coord,
        x_coordinate_type=XCoordinateType.TIME,
    ).s3_path

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            run_id=RunId(value=str(run_id)),
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=LoggedDataType.LOGGED_DATA_TYPE_IMAGE,
            x_coordinate=x_coord,
            x_coordinate_type=LoggedDataXCoordinateType.LOGGED_DATA_X_COORDINATE_TYPE_TIME,
        )
    )
    assert s3_path == "path/to/upload"


def test_given_tag_not_exists_when_log_number_then_calls_log_numeric_data() -> None:
    # given
    mock_logged_data_api = MagicMock()
    mock_logged_data_api.LogData.return_value = LogDataResponse(
        logged_data_id=LoggedDataId(value="00000000-0000-0000-0000-000000000000"),
    )
    logged_data_client = get_logged_data_service_client_with_mocks(
        logged_data_api_stub=mock_logged_data_api
    )
    run_id = uuid.uuid4()
    train_id = uuid.uuid4()
    tag = "foo-test-tag"

    # when
    logged_data_client.log_data(
        run_id=run_id,
        train_id=train_id,
        tag=tag,
        type=LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
        x_coordinate=1,
        x_coordinate_type=XCoordinateType.STEP,
        value="123",
    )

    # then
    mock_logged_data_api.LogData.assert_called_with(
        request=LogDataRequest(
            run_id=RunId(value=str(run_id)),
            model_train_id=ModelTrainId(value=str(train_id)),
            unique_tag=tag,
            type=LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
            x_coordinate=1,
            x_coordinate_type=LoggedDataXCoordinateType.LOGGED_DATA_X_COORDINATE_TYPE_STEP,
            value="123",
        )
    )
