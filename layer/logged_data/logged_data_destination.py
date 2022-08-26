from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from uuid import UUID

from layer.clients.logged_data_service import LoggedDataClient
from layer.contracts.logged_data import LoggedData


class LoggedDataDestination(ABC):
    @abstractmethod
    def get_logged_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LoggedData:
        pass

    @abstractmethod
    def receive(
        self,
        func: Callable[[LoggedDataClient], Optional[Any]],
        data: Optional[Any] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_logging_errors(self) -> Optional[str]:
        pass
