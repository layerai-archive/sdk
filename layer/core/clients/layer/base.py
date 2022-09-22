from abc import ABC, abstractmethod
from typing import Optional

from .account_api import AccountAPIClient
from .data_catalog_api import DataCatalogAPIClient
from .executor_api import ExecutorAPIClient
from .flow_manager_api import FlowManagerAPIClient
from .label_api import LabelAPIClient
from .logged_data_api import LoggedDataAPIClient
from .model_catalog_api import ModelCatalogAPIClient
from .project_api import ProjectAPIClient
from .run_api import RunAPIClient
from .user_logs_api import UserLogsAPIClient


class LayerAPIClient(ABC):
    # general
    _account_api_client: Optional[AccountAPIClient] = None
    _project_api_client: Optional[ProjectAPIClient] = None
    # metadata
    _run_api_client: Optional[RunAPIClient] = None
    _logged_data_api_client: Optional[LoggedDataAPIClient] = None
    _label_api_client: Optional[LabelAPIClient] = None
    # registry
    _data_catalog_api_client: Optional[DataCatalogAPIClient] = None
    _model_catalog_api_client: Optional[ModelCatalogAPIClient] = None
    # execution
    _flow_manager_api_client: Optional[FlowManagerAPIClient] = None
    _executor_api_client: Optional[ExecutorAPIClient] = None
    _user_logs_api_client: Optional[UserLogsAPIClient] = None

    @abstractmethod
    def get_account_api_client(self) -> AccountAPIClient:
        ...

    @property
    def account_api_client(self) -> AccountAPIClient:
        if self._account_api_client is None:
            self._account_api_client = self.get_account_api_client()
        return self._account_api_client

    @abstractmethod
    def get_project_api_client(self) -> ProjectAPIClient:
        ...

    @property
    def project_api_client(self) -> ProjectAPIClient:
        if self._project_api_client is None:
            self._project_api_client = self.get_project_api_client()
        return self._project_api_client

    @abstractmethod
    def get_run_api_client(self) -> RunAPIClient:
        ...

    @property
    def run_api_client(self) -> RunAPIClient:
        if self._run_api_client is None:
            self._run_api_client = self.get_run_api_client()
        return self._run_api_client

    @abstractmethod
    def get_logged_data_api_client(self) -> LoggedDataAPIClient:
        ...

    @property
    def logged_data_api_client(self) -> LoggedDataAPIClient:
        if self._logged_data_api_client is None:
            self._logged_data_api_client = self.get_logged_data_api_client()
        return self._logged_data_api_client

    @abstractmethod
    def get_label_api_client(self) -> LabelAPIClient:
        ...

    @property
    def label_api_client(self) -> LabelAPIClient:
        if self._label_api_client is None:
            self._label_api_client = self.get_label_api_client()
        return self._label_api_client

    @abstractmethod
    def get_data_catalog_api_client(self) -> DataCatalogAPIClient:
        ...

    @property
    def data_catalog_api_client(self) -> DataCatalogAPIClient:
        if self._data_catalog_api_client is None:
            self._data_catalog_api_client = self.get_data_catalog_api_client()
        return self._data_catalog_api_client

    @abstractmethod
    def get_model_catalog_api_client(self) -> ModelCatalogAPIClient:
        ...

    @property
    def model_catalog_api_client(self) -> ModelCatalogAPIClient:
        if self._model_catalog_api_client is None:
            self._model_catalog_api_client = self.get_model_catalog_api_client()
        return self._model_catalog_api_client

    @abstractmethod
    def get_flow_manager_api_client(self) -> FlowManagerAPIClient:
        ...

    @property
    def flow_manager_api_client(self) -> FlowManagerAPIClient:
        if self._flow_manager_api_client is None:
            self._flow_manager_api_client = self.get_flow_manager_api_client()
        return self._flow_manager_api_client

    @abstractmethod
    def get_executor_api_client(self) -> ExecutorAPIClient:
        ...

    @property
    def executor_api_client(self) -> ExecutorAPIClient:
        if self._executor_api_client is None:
            self._executor_api_client = self.get_executor_api_client()
        return self._executor_api_client

    @abstractmethod
    def get_user_logs_api_client(self) -> UserLogsAPIClient:
        ...

    @property
    def user_logs_api_client(self) -> UserLogsAPIClient:
        if self._user_logs_api_client is None:
            self._user_logs_api_client = self.get_user_logs_api_client()
        return self._user_logs_api_client
