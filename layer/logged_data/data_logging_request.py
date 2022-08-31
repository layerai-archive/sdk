from typing import Any, Callable, Optional

from layer.logged_data.file_uploader import FileUploader


class DataLoggingRequest:
    def __init__(
        self,
        files_storage: FileUploader,
        queued_operation_func: Callable[[], Optional[str]],
        data: Optional[Any] = None,
    ) -> None:
        self._queued_operation_func = queued_operation_func
        self._data = data
        self._files_storage = files_storage

    def execute(self) -> None:
        maybe_s3_path = self._queued_operation_func()
        if self._data is not None and maybe_s3_path is not None:
            self._files_storage.upload(maybe_s3_path, self._data)
