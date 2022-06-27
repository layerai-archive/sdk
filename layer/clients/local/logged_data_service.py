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
    def init(self):
        return self

    def get_logged_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LoggedData:
        pass

    def log_model_metric(
        self, train_id: UUID, tag: str, points: List[ModelMetricPoint]
    ) -> ModelMetricId:
        pass

    def log_binary_data(
        self,
        *,
        tag: str,
        logged_data_type: "LoggedDataType.V",
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        epoch: Optional[int] = None,
    ) -> str:
        pass

    def log_text_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        pass

    def log_markdown_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        pass

    def log_numeric_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        print(tag, data, train_id)

    def log_boolean_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        pass

    def log_table_data(
        self,
        tag: str,
        data: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> None:
        pass

    def _log_data(
        self,
        tag: str,
        type: "LoggedDataType.V",
        data: Optional[str] = None,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
        epoch: Optional[int] = None,
    ) -> LogDataResponse:
        pass
