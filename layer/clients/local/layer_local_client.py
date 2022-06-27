from logging import Logger

from layer.clients.local.data_catalog import DataCatalogClient
from layer.clients.local.logged_data_service import LoggedDataClient
from layer.clients.local.model_catalog import ModelCatalogClient
from layer.clients.local.project import ProjectServiceClient
from layer.clients.model_training_service import ModelTrainingClient
from layer.config import ClientConfig


class LayerLocalClient:
    def __init__(self, config: ClientConfig, logger: Logger):
        self._config = config
        self._model_catalog = ModelCatalogClient(config, logger)
        self._model_training = ModelTrainingClient(config, logger)
        self._data_catalog = DataCatalogClient(config, logger)
        self._project_service_client = ProjectServiceClient(config, logger)
        self._logged_data_client = LoggedDataClient(config, logger)

    def init(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        pass

    @staticmethod
    def get_project_name():
        from layer import current_project_name

        project_name = current_project_name()
        if project_name is None:
            project_name = "experiment"
        return project_name

    @property
    def model_catalog(self) -> ModelCatalogClient:
        return self._model_catalog

    @property
    def data_catalog(self) -> DataCatalogClient:
        return self._data_catalog

    @property
    def model_training(self) -> ModelTrainingClient:
        return self._model_training

    @property
    def project_service_client(self) -> ProjectServiceClient:
        return self._project_service_client

    @property
    def logged_data_service_client(self) -> LoggedDataClient:
        return self._logged_data_client
