import pathlib
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
        data_path: Optional[pathlib.Path] = None,
    ) -> None:
        if data is not None and data_path is not None:
            raise ValueError(
                "Invalid arguments: 'data' and 'data_path' cannot both be given"
            )
        DataLoggingRequest(
            files_storage=self._files_storage,
            queued_operation_func=lambda: func(self.logged_data_client),
            data=data,
            data_path=data_path,
        ).execute()

    def close_and_get_errors(self) -> Optional[str]:
        pass
