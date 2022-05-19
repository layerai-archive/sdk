from contextlib import contextmanager
from logging import Logger
from typing import Iterator, List, Optional, cast
from uuid import UUID

from layerapi.api.entity.logged_model_metric_pb2 import LoggedModelMetric
from layerapi.api.ids_pb2 import DatasetBuildId, ModelMetricId, ModelTrainId
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
from layer.utils.grpc import create_grpc_channel


class LoggedDataClient:
    _service: LoggedDataAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["LoggedDataClient"]:
        with create_grpc_channel(
            self._config.grpc_gateway_address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = LoggedDataAPIStub(channel=channel)
            yield self

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
        )

    def log_model_metric(
        self, train_id: UUID, tag: str, points: List[ModelMetricPoint]
    ) -> ModelMetricId:
        metric = LoggedModelMetric(
            unique_tag=tag,
            points=[
                LoggedModelMetric.ModelMetricPoint(epoch=p.epoch, value=p.value)
                for p in points
            ],
        )
        request = LogModelMetricRequest(
            model_train_id=ModelTrainId(value=str(train_id)), metric=metric
        )
        response = self._service.LogModelMetric(request=request)
        return response.model_metric_id

    def log_binary_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> str:
        return self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_BLOB,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
        ).s3_path

    def log_text_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_TEXT,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
        )

    def log_numeric_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_NUMBER,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
        )

    def log_boolean_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_BOOLEAN,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
        )

    def log_table_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        self._log_data(
            tag,
            LoggedDataType.LOGGED_DATA_TYPE_TABLE,
            data=data,
            train_id=train_id,
            dataset_build_id=dataset_build_id,
        )

    def _log_data(
        self,
        tag: str,
        type: "LoggedDataType.V",
        data: Optional[str] = None,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LogDataResponse:
        text = cast(str, data)
        request = LogDataRequest(
            unique_tag=tag,
            type=type,
            text=text,
            model_train_id=ModelTrainId(value=str(train_id))
            if train_id is not None
            else None,
            dataset_build_id=DatasetBuildId(value=str(dataset_build_id))
            if dataset_build_id is not None
            else None,
        )
        return self._service.LogData(request=request)
