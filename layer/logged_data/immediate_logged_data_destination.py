from types import TracebackType
from typing import Any, Callable, Optional
from uuid import UUID

from layer.clients.logged_data_service import LoggedDataClient
from layer.contracts.logged_data import LoggedData
from layer.logged_data.data_logging_request import DataLoggingRequest
from layer.logged_data.file_uploader import FileUploader
from layer.logged_data.logged_data_destination import LoggedDataDestination


class ImmediateLoggedDataDestination(LoggedDataDestination):
    def get_logged_data(
        self,
        tag: str,
        train_id: Optional[UUID] = None,
        dataset_build_id: Optional[UUID] = None,
    ) -> LoggedData:
        return self._log_data_client.get_logged_data(
            tag=tag, train_id=train_id, dataset_build_id=dataset_build_id
        )

    def __init__(self, client: LoggedDataClient) -> None:
        self._log_data_client = client
        self._files_storage = FileUploader()

    def __enter__(self) -> LoggedDataDestination:
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._files_storage.close()

    def receive(
        self,
        func: Callable[[LoggedDataClient], Optional[Any]],
        data: Optional[Any] = None,
    ) -> None:
        DataLoggingRequest(
            self._files_storage, lambda: func(self._log_data_client), data
        ).execute()

    def get_logging_errors(self) -> None:
        pass
