import threading
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from layerapi.api.service.logged_data.logged_data_api_pb2 import LogDataResponse

from layer.clients.logged_data_service import LoggedDataClient
from layer.logged_data.queuing_logged_data_destination import (
    QueueingLoggedDataDestination,
)


def test_given_errors_on_upload_and_grpc_when_get_errors_then_returns_all_errors() -> None:
    # given
    error_message = "Something happened!"
    other_error_message = "Something else happened!"

    def queued_failed_execution(__: LoggedDataClient):
        raise Exception(error_message)

    def queued_successful_execution(__: LoggedDataClient):
        response = LogDataResponse()
        response.s3_path = "https://valid_url"
        return response

    def failed_upload(_: str, __: Optional[Any] = None):
        raise Exception(other_error_message)

    logged_data_client = MagicMock()

    with patch(
        "layer.logged_data.file_uploader.FileUploader.upload", side_effect=failed_upload
    ), QueueingLoggedDataDestination(logged_data_client) as destination:

        destination.receive(queued_failed_execution)
        destination.receive(queued_successful_execution, data="12345")

        # when
        errors = destination.close_and_get_errors()

        # then
        assert (
            errors
            == f"WARNING: Layer was unable to log requested data because of the following errors:\nException: {error_message}\nException: {other_error_message}\n"
        )


def test_given_logging_successful_then_no_errors() -> None:
    logged_data_client = MagicMock()

    def queued_ok_execution(__: LoggedDataClient):
        pass

    with QueueingLoggedDataDestination(logged_data_client) as destination:
        destination.receive(queued_ok_execution)
        assert destination.close_and_get_errors() is None
        assert destination.close_and_get_errors() is None


def test_even_when_reading_thread_terminated_then_queue_flushed() -> None:
    execution_happened = False
    mock_thread_used_as_reading_thread = False

    # use a fake reading thread that terminates right after is starts and doesn't read any messages
    def set_and_terminate() -> None:
        nonlocal mock_thread_used_as_reading_thread
        mock_thread_used_as_reading_thread = True
        raise RuntimeError()

    def queued_ok_execution(__: LoggedDataClient) -> None:
        nonlocal execution_happened
        execution_happened = True

    idle_thread = threading.Thread(target=set_and_terminate)
    with patch("threading.Thread", return_value=idle_thread):
        logged_data_client = MagicMock()
        with QueueingLoggedDataDestination(logged_data_client) as destination:
            idle_thread.join(1.0)  # wait a moment for fake reading thread to terminate
            assert idle_thread.is_alive() is False
            destination.receive(queued_ok_execution)
            assert execution_happened is False

    assert (
        mock_thread_used_as_reading_thread is True
    )  # make sure we used fake thread and not having false positive
    assert (
        execution_happened is True
    )  # make sure that request was executed anyway upon __exit__ of the destination
