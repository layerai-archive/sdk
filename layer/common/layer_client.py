from contextlib import ExitStack, contextmanager
from logging import Logger
from typing import Iterator

from layer.client import (
    DataCatalogClient,
    FlowManagerClient,
    ModelCatalogClient,
    ModelTrainingClient,
    UserLogsClient,
)
from layer.clients import AccountClient
from layer.config import ClientConfig
from layer.logged_data import LoggedDataClient
from layer.mlmodels.service import MLModelService
from layer.projects import ProjectServiceClient


class LayerClient:
    def __init__(self, config: ClientConfig, logger: Logger):
        self._config = config
        self._data_catalog = DataCatalogClient(config, logger)
        self._model_catalog = ModelCatalogClient(
            config,
            MLModelService(logger, s3_endpoint_url=config.s3.endpoint_url),
            logger,
        )
        self._model_training = ModelTrainingClient(config, logger)
        self._account = AccountClient(config, logger)
        self._flow_manager = FlowManagerClient(config, logger)
        self._user_logs = UserLogsClient(config, logger)
        self._project_service_client = ProjectServiceClient(config, logger)
        self._logged_data_client = LoggedDataClient(config, logger)

    @contextmanager
    def init(self) -> Iterator["LayerClient"]:
        with ExitStack() as exit_stack:
            # TODO(emin): remove unused config objects
            if self._config.data_catalog.is_enabled:
                exit_stack.enter_context(self._data_catalog.init())
            if self._config.model_catalog.is_enabled:
                exit_stack.enter_context(self._model_catalog.init())
            if self._config.model_training.is_enabled:
                exit_stack.enter_context(self._model_training.init())
            if self._config.account_service.is_enabled:
                exit_stack.enter_context(self._account.init())
            if self._config.flow_manager.is_enabled:
                exit_stack.enter_context(self._flow_manager.init())
            if self._config.user_logs.is_enabled:
                exit_stack.enter_context(self._user_logs.init())
            if self._config.project_service.is_enabled:
                exit_stack.enter_context(self.project_service_client.init())
            exit_stack.enter_context(self._logged_data_client.init())
            yield self

    @property
    def data_catalog(self) -> DataCatalogClient:
        return self._data_catalog

    @property
    def model_catalog(self) -> ModelCatalogClient:
        return self._model_catalog

    @property
    def model_training(self) -> ModelTrainingClient:
        return self._model_training

    @property
    def account(self) -> AccountClient:
        return self._account

    @property
    def flow_manager(self) -> FlowManagerClient:
        return self._flow_manager

    @property
    def user_logs(self) -> UserLogsClient:
        return self._user_logs

    @property
    def project_service_client(self) -> ProjectServiceClient:
        return self._project_service_client

    @property
    def logged_data_service_client(self) -> LoggedDataClient:
        return self._logged_data_client
