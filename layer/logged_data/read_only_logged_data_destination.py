from typing import Any, Callable, Optional
from uuid import UUID

from layer.clients.logged_data_service import LoggedDataClient
from layer.contracts.logged_data import LoggedData
from layer.logged_data.logged_data_destination import LoggedDataDestination


class ReadOnlyLoggedDataDestination(LoggedDataDestination):
    def __init__(self, client: LoggedDataClient) -> None:
        self._log_data_client = client

    def get_logged_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LoggedData:
        return self._log_data_client.get_logged_data(
            tag=tag, train_id=train_id, dataset_build_id=dataset_build_id
        )

    def receive(
        self,
        func: Callable[[LoggedDataClient], Optional[Any]],
        data: Optional[Any] = None,
    ) -> None:
        raise RuntimeError("Readonly logged data destination can only be read.")

    def get_logging_errors(self) -> None:
        raise RuntimeError("Readonly logged data destination can only be read.")
