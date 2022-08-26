from types import TracebackType
from typing import Any, Callable, Optional

from layer.clients.logged_data_service import LoggedDataClient
from layer.logged_data.data_logging_request import DataLoggingRequest
from layer.logged_data.file_uploader import FileUploader
from layer.logged_data.logged_data_destination import LoggedDataDestination


class ImmediateLoggedDataDestination(LoggedDataDestination):
    def __init__(self, client: LoggedDataClient) -> None:
        super().__init__(client)
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
            self._files_storage, lambda: func(self.logged_data_client), data
        ).execute()

    def close_and_get_errors(self) -> Optional[str]:
        pass
