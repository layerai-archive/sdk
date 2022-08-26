from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from uuid import UUID

from layer.clients.logged_data_service import LoggedDataClient
from layer.contracts.logged_data import LoggedData


class LoggedDataDestination(ABC):
    def __init__(self, client: LoggedDataClient) -> None:
        self.logged_data_client = client

    def get_logged_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LoggedData:
        return self.logged_data_client.get_logged_data(
            tag=tag, train_id=train_id, dataset_build_id=dataset_build_id
        )

    @abstractmethod
    def receive(
        self,
        func: Callable[[LoggedDataClient], Optional[Any]],
        data: Optional[Any] = None,
    ) -> None:
        pass

    @abstractmethod
    def close_and_get_errors(self) -> Optional[str]:
        pass
