from contextlib import contextmanager
from logging import Logger
from typing import Iterator, Optional

from layer.config import ClientConfig
from layer.utils.grpc.channel import get_grpc_channel

from .account_service import AccountServiceClient
from .data_catalog import DataCatalogClient
from .executor_service import ExecutorClient
from .flow_manager import FlowManagerClient
from .label import LabelClient
from .logged_data_service import LoggedDataClient
from .model_catalog import ModelCatalogClient
from .project_service import ProjectServiceClient
from .run_service import RunServiceClient
from .user_logs_service import UserLogsClient


class LayerClient:
    def __init__(self, config: ClientConfig, logger: Logger):
        self._config = config
        self._logger = logger
        self._data_catalog: Optional[DataCatalogClient] = None
        self._model_catalog: Optional[ModelCatalogClient] = None
        self._account: Optional[AccountServiceClient] = None
        self._flow_manager: Optional[FlowManagerClient] = None
        self._user_logs: Optional[UserLogsClient] = None
        self._project_service_client: Optional[ProjectServiceClient] = None
        self._run_service_client: Optional[RunServiceClient] = None
        self._logged_data_client: Optional[LoggedDataClient] = None
        self._label_client: Optional[LabelClient] = None
        self._executor_client: Optional[ExecutorClient] = None

    @contextmanager
    def init(self) -> Iterator["LayerClient"]:
        # kept for backwards compatibility only, remove in future version:
        # https://linear.app/layer/issue/LAY-3547/remove-layerclientconfigclient-loggerinit-method
        yield self

    @property
    def data_catalog(self) -> DataCatalogClient:
        if self._data_catalog is None:
            self._data_catalog = DataCatalogClient.create(self._config, self._logger)
        return self._data_catalog

    @property
    def model_catalog(self) -> ModelCatalogClient:
        if self._model_catalog is None:
            self._model_catalog = ModelCatalogClient.create(self._config, self._logger)
        return self._model_catalog

    @property
    def account(self) -> AccountServiceClient:
        if self._account is None:
            self._account = AccountServiceClient.create(self._config)
        return self._account

    @property
    def flow_manager(self) -> FlowManagerClient:
        if self._flow_manager is None:
            self._flow_manager = FlowManagerClient.create(self._config)
        return self._flow_manager

    @property
    def user_logs(self) -> UserLogsClient:
        if self._user_logs is None:
            self._user_logs = UserLogsClient.create(self._config)
        return self._user_logs

    @property
    def project_service_client(self) -> ProjectServiceClient:
        if self._project_service_client is None:
            self._project_service_client = ProjectServiceClient.create(self._config)
        return self._project_service_client

    @property
    def run_service_client(self) -> RunServiceClient:
        if self._run_service_client is None:
            self._run_service_client = RunServiceClient.create(self._config)
        return self._run_service_client

    @property
    def logged_data_service_client(self) -> LoggedDataClient:
        if self._logged_data_client is None:
            self._logged_data_client = LoggedDataClient.create(self._config)
        return self._logged_data_client

    @property
    def label_service_client(self) -> LabelClient:
        if self._label_client is None:
            self._label_client = LabelClient.create(self._config)
        return self._label_client

    @property
    def executor_service_client(self) -> ExecutorClient:
        if self._executor_client is None:
            self._executor_client = ExecutorClient.create(self._config)
        return self._executor_client

    def close(self) -> None:
        channel = get_grpc_channel(self._config, closing=True)
        if channel is not None:
            channel.close()
