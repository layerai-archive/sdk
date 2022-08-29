import queue
import threading
from queue import Queue
from types import TracebackType
from typing import Any, Callable, Optional

from layer.clients.logged_data_service import LoggedDataClient
from layer.logged_data.data_logging_request import DataLoggingRequest
from layer.logged_data.file_uploader import FileUploader
from layer.logged_data.logged_data_destination import LoggedDataDestination


WAIT_INTERVAL_SECONDS = 1
LOGGING_TIMEOUT = 300


class QueueingLoggedDataDestination(LoggedDataDestination):
    def __init__(self, client: LoggedDataClient) -> None:
        super().__init__(client)
        self._files_storage = FileUploader()

        self._sending_errors: str = ""
        self._stop_reading = False
        self._local_queue: Queue[DataLoggingRequest] = Queue()
        self._reading_thread = threading.Thread(target=self._execute_elem_from_queue)

    def __enter__(self) -> "QueueingLoggedDataDestination":
        self._reading_thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._exit()

    def receive(
        self,
        func: Callable[[LoggedDataClient], Optional[Any]],
        data: Optional[Any] = None,
    ) -> None:
        self._local_queue.put_nowait(
            DataLoggingRequest(
                files_storage=self._files_storage,
                queued_operation_func=lambda: func(self.logged_data_client),
                data=data,
            )
        )

    def close_and_get_errors(self) -> Optional[str]:
        self._exit()
        return (
            None
            if len(self._sending_errors) == 0
            else f"WARNING: Layer was unable to log requested data because of the following errors:\n{self._sending_errors}"
        )

    def _exit(self) -> None:
        if not self._stop_reading:
            self._stop_reading = True
            self._reading_thread.join(timeout=LOGGING_TIMEOUT)
            if self._reading_thread.is_alive():
                # this means that there is some active logging request that is being executed for more than
                # LOGGING_TIMEOUT
                self._sending_errors = (
                    self._sending_errors + f"Requested data could not be logged within "
                    f"{LOGGING_TIMEOUT}s after finished execution,"
                    f" giving up.\n"
                )
            else:  # in the case the thread was stopped, but there is still something unprocessed in the qeueue
                self._flush_queue()
            self._files_storage.close()

    def _execute_elem_from_queue(self) -> None:
        while not self._stop_reading:
            try:
                execution_item = self._local_queue.get(
                    block=True, timeout=WAIT_INTERVAL_SECONDS
                )
                execution_item.execute()
            except queue.Empty:
                # no item in the queue, wait for another
                pass
            except Exception as ex:
                self._append_to_error_message(ex)

    def _flush_queue(self) -> None:
        while not self._local_queue.empty():
            execution_item = self._local_queue.get(block=False)
            try:
                execution_item.execute()
            except Exception as ex:
                self._append_to_error_message(ex)
        return

    def _append_to_error_message(self, ex: Exception) -> None:
        self._sending_errors = self._sending_errors + f"Exception: {ex}\n"
