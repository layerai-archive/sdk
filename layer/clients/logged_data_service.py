from typing import List, Optional, cast
from uuid import UUID

from layerapi.api.entity.logged_model_metric_pb2 import LoggedModelMetric
from layerapi.api.ids_pb2 import (
    DatasetBuildId,
    LoggedMetricGroupId,
    ModelMetricId,
    ModelTrainId,
)
from layerapi.api.service.logged_data.logged_data_api_pb2 import (
    GetLoggedDataRequest,
    LogDataRequest,
    LogDataResponse,
    LogModelMetricRequest,
)
from layerapi.api.service.logged_data.logged_data_api_pb2_grpc import LoggedDataAPIStub
from layerapi.api.value.logged_data_type_pb2 import LoggedDataType

from layer.config import ClientConfig
from layer.contracts.logged_data import LoggedData
from layer.contracts.logged_data import LoggedDataType as LDType
from layer.contracts.logged_data import ModelMetricPoint
from layer.utils.grpc.channel import get_grpc_channel


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
            data=logged_data_pb.text,
            logged_data_type=LDType(logged_data_pb.type),
            tag=logged_data_pb.unique_tag,
            epoched_data=logged_data_pb.epoched_data,
        )

    def log_model_metric(
        self,
        train_id: UUID,
        tag: str,
        points: List[ModelMetricPoint],
        metric_group_id: UUID,
        category: Optional[str] = None,
    ) -> ModelMetricId:
        metric = LoggedModelMetric(
            unique_tag=tag,
            points=[
                LoggedModelMetric.ModelMetricPoint(epoch=p.epoch, value=p.value)
                for p in points
            ],
            group_id=LoggedMetricGroupId(value=str(metric_group_id)),
            category=category if category is not None else "",
        )
        request = LogModelMetricRequest(
            model_train_id=ModelTrainId(value=str(train_id)), metric=metric
        )
        response = self._service.LogModelMetric(request=request)
        return response.model_metric_id

    def log_binary_data(
        self,
        *,
        tag: str,
        logged_data_type: "LoggedDataType.V",
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        epoch: Optional[int] = None,
        category: Optional[str] = None,
    ) -> str:
        return self._log_data(
            tag,
            logged_data_type,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            epoch=epoch,
            category=category,
        ).s3_path

    def log_text_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        category: Optional[str] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_TEXT,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            category=category,
        )

    def log_markdown_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        category: Optional[str] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_MARKDOWN,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            category=category,
        )

    def log_numeric_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        category: Optional[str] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            category=category,
        )

    def log_boolean_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        category: Optional[str] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            category=category,
        )

    def log_table_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        category: Optional[str] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_TABLE,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            category=category,
        )

    def _log_data(
        self,
        tag: str,
        type: "LoggedDataType.V",
        data: Optional[str] = None,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        epoch: Optional[int] = None,
        category: Optional[str] = None,
    ) -> LogDataResponse:
        text = cast(str, data)
        request = LogDataRequest(
            unique_tag=tag,
            type=type,
            text=text,
            # Given that protocol buffers do not allow differentiating between default value (0) or the actual value 0
            # We use the convention that a minus value represents the absence of an epoch.
            # See https://developers.google.com/protocol-buffers/docs/proto3#default for more information.
            epoch=epoch if epoch is not None else -1,
            model_train_id=ModelTrainId(value=str(train_id))
            if train_id is not None
            else None,
            dataset_build_id=DatasetBuildId(value=str(dataset_build_id))
            if dataset_build_id is not None
            else None,
            category=category if category is not None else "",
        )
        return self._service.LogData(request=request)
