from typing import TYPE_CHECKING, Optional
from uuid import UUID

from layerapi.api.ids_pb2 import DatasetBuildId, ModelTrainId
from layerapi.api.service.logged_data.logged_data_api_pb2 import (
    GetLoggedDataRequest,
    LogDataRequest,
    LogDataResponse,
)
from layerapi.api.service.logged_data.logged_data_api_pb2_grpc import LoggedDataAPIStub
from layerapi.api.value.logged_data_x_coordinate_type_pb2 import (
    LoggedDataXCoordinateType,
)

from layer.config import ClientConfig
from layer.contracts.logged_data import LoggedData
from layer.contracts.logged_data import LoggedDataType as LDType
from layer.contracts.logged_data import XCoordinateType
from layer.utils.grpc.channel import get_grpc_channel


if TYPE_CHECKING:
    from layerapi.api.value.logged_data_type_pb2 import LoggedDataType


X_COORDINATE_TYPE_PROTO_MAP = {
    XCoordinateType.STEP: LoggedDataXCoordinateType.LOGGED_DATA_X_COORDINATE_TYPE_STEP,
    XCoordinateType.TIME: LoggedDataXCoordinateType.LOGGED_DATA_X_COORDINATE_TYPE_TIME,
}


class LoggedDataClient:
    _service: LoggedDataAPIStub

    @staticmethod
    def create(config: ClientConfig) -> "LoggedDataClient":
        client = LoggedDataClient()
        channel = get_grpc_channel(config)
        client._service = LoggedDataAPIStub(channel)  # pylint: disable=protected-access
        return client

    def get_logged_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LoggedData:
        request = GetLoggedDataRequest(
            unique_tag=tag,
            model_train_id=ModelTrainId(value=str(train_id))
            if train_id is not None
            else None,
            dataset_build_id=DatasetBuildId(value=str(dataset_build_id))
            if dataset_build_id is not None
            else None,
        )
        logged_data_pb = self._service.GetLoggedData(request=request).data
        return LoggedData(
            tag=logged_data_pb.unique_tag,
            logged_data_type=LDType(logged_data_pb.type),
            value=logged_data_pb.value,
            values_with_coordinates={
                coord.x: coord.value for coord in logged_data_pb.values_with_coordinates
            },
        )

    def log_data(
        self,
        tag: str,
        type: "LoggedDataType.V",
        value: str = "",
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        group_tag: Optional[str] = None,
        category: Optional[str] = None,
        x_coordinate: Optional[int] = None,
        x_coordinate_type: Optional[XCoordinateType] = None,
    ) -> LogDataResponse:
        request = LogDataRequest(
            unique_tag=tag,
            category=category if category else "",
            group_tag=group_tag if group_tag else "",
            type=type,
            value=value,
            # Given that protocol buffers do not allow differentiating between default value (0) or the actual value 0
            # We use the convention that a minus value represents the absence of an epoch.
            # See https://developers.google.com/protocol-buffers/docs/proto3#default for more information.
            x_coordinate=x_coordinate if x_coordinate is not None else -1,
            x_coordinate_type=X_COORDINATE_TYPE_PROTO_MAP.get(x_coordinate_type)
            if x_coordinate_type
            else None,
            model_train_id=ModelTrainId(value=str(train_id))
            if train_id is not None
            else None,
            dataset_build_id=DatasetBuildId(value=str(dataset_build_id))
            if dataset_build_id is not None
            else None,
        )
        return self._service.LogData(request=request)
