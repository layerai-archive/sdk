from typing import Any, Callable, Optional

from layerapi.api.service.logged_data.logged_data_api_pb2 import LogDataResponse

from layer.logged_data.file_uploader import FileUploader


class DataLoggingRequest:
    def __init__(
        self,
        files_storage: FileUploader,
        queued_operation_func: Callable[[], Optional[LogDataResponse]],
        data: Optional[Any] = None,
    ) -> None:
        self._queued_operation_func = queued_operation_func
        self._data = data
        self._files_storage = files_storage

    def execute(self) -> None:
        data_logging_response = self._queued_operation_func()
        if (
            self._data is not None
            and data_logging_response is not None
            and data_logging_response.s3_path is not None
        ):
            self._files_storage.upload(data_logging_response.s3_path, self._data)
