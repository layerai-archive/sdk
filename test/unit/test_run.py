import os
import uuid
from typing import Any, Optional
from unittest.mock import MagicMock

import grpc
import pytest
from tests.unit.grpc_test_utils import new_client_call_details, rpc_error

from layer.api.entity.operations_pb2 import ExecutionPlan
from layer.api.entity.run_metadata_pb2 import RunMetadata
from layer.api.ids_pb2 import RunId
from layer.api.service.flowmanager.flow_manager_api_pb2 import (
    GetRunHistoryAndMetadataRequest,
    GetRunHistoryAndMetadataResponse,
    StartRunV2Request,
    StartRunV2Response,
)
from layer.exceptions.exceptions import ProjectRunnerError
from layer.grpc_utils.interceptors import (
    GRPCErrorClientInterceptor,
    RequestIdInterceptor,
)
from layer.projects.project_runner import ProjectRunner
from layer.projects.tracker.remote_execution_project_progress_tracker import (
    RemoteExecutionProjectProgressTracker,
)
from layer.run import (
    _ENV_KEY_LAYER_DEBUG,
    _ENV_KEY_REQUEST_ID,
    UserSessionId,
    is_layer_debug_on,
)


class TestLayerDebug:
    @pytest.fixture()
    def _cleanup_after(self) -> None:
        yield
        if _ENV_KEY_LAYER_DEBUG in os.environ:
            del os.environ[_ENV_KEY_LAYER_DEBUG]

    @pytest.mark.usefixtures("_cleanup_after")
    @pytest.mark.parametrize(
        ("test_input", "expected"),
        [
            ("False", False),
            ("false", False),
            ("0", False),
            ("", False),
            ("random", False),
            ("True", True),
            ("true", True),
            ("1", True),
        ],
    )
    def test_detects_layer_debug_correctly(self, test_input, expected):
        os.environ[_ENV_KEY_LAYER_DEBUG] = test_input

        is_debug = is_layer_debug_on()

        assert is_debug == expected


class TestUserSessionIdSingleton:
    def test_user_session_id_singleton(self) -> None:
        assert str(UserSessionId()) == str(UserSessionId())

    def test_user_session_id_value_uuid(self) -> None:
        uuid.UUID(str(UserSessionId()))


class TestRequestIdInterceptor:
    @pytest.fixture()
    def _cleanup_after(self) -> None:
        yield
        RequestIdInterceptor._clear_instance()
        if _ENV_KEY_REQUEST_ID in os.environ:
            del os.environ[_ENV_KEY_REQUEST_ID]

    @pytest.mark.usefixtures("_cleanup_after")
    def test_is_singleton(self):
        assert RequestIdInterceptor() == RequestIdInterceptor()

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_adds_random_request_id_with_valid_uuid_when_not_set_from_env(
        self,
    ):
        # given
        client_call_details = _client_call_details("/api.FlowManagerAPI/StartRun")
        run_id = RunId(value=str(uuid.uuid4()))
        mock_response = _mock_response(StartRunV2Response(run_id=run_id))
        continuation = MagicMock(return_value=mock_response)

        # when
        interceptor = RequestIdInterceptor()
        request = StartRunV2Request(project_name="lala")
        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=client_call_details,
            request=request,
        )

        # then
        continuation.assert_called_with(client_call_details, request)
        assert isinstance(response.result(), StartRunV2Response)
        request_id_found = _get_request_id_from(client_call_details)
        assert request_id_found is not None
        try:
            uuid.UUID(request_id_found)
        except Exception as e:
            pytest.fail(f"unexpected exception raised: '{e}'")

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_adds_request_id_from_env_if_valid(self):
        # given
        client_call_details = _client_call_details("/api.FlowManagerAPI/StartRun")
        run_id = RunId(value=str(uuid.uuid4()))
        mock_response = _mock_response(StartRunV2Response(run_id=run_id))
        continuation = MagicMock(return_value=mock_response)

        # when
        request_id_from_env = str(uuid.uuid4())
        os.environ[_ENV_KEY_REQUEST_ID] = request_id_from_env
        interceptor = RequestIdInterceptor()
        request = StartRunV2Request(project_name="lala")
        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=client_call_details,
            request=request,
        )

        # then
        continuation.assert_called_with(client_call_details, request)
        assert isinstance(response.result(), StartRunV2Response)
        request_id_found = _get_request_id_from(client_call_details)
        assert request_id_found == request_id_from_env

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_raises_exception_when_request_id_from_env_is_invalid(self):
        request_id_from_env = "invalid"
        os.environ[_ENV_KEY_REQUEST_ID] = request_id_from_env

        with pytest.raises(ValueError, match=".*invalid REQUEST_ID.*"):
            RequestIdInterceptor()

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_adds_different_request_id_for_each_request(self):
        # given
        run_id = RunId(value=str(uuid.uuid4()))
        client_call_details_1 = _client_call_details(
            "/api.FlowManagerAPI/GetRunHistoryAndMetadata"
        )
        client_call_details_2 = _client_call_details(
            "/api.FlowManagerAPI/GetRunHistoryAndMetadata"
        )
        mock_response_1 = _mock_response(
            GetRunHistoryAndMetadataResponse(run_metadata=RunMetadata())
        )
        continuation_1 = MagicMock(return_value=mock_response_1)
        mock_response_2 = _mock_response(
            GetRunHistoryAndMetadataResponse(run_metadata=RunMetadata())
        )
        continuation_2 = MagicMock(return_value=mock_response_2)

        # when
        interceptor = RequestIdInterceptor()
        request_1 = GetRunHistoryAndMetadataRequest(run_id=run_id)
        request_2 = GetRunHistoryAndMetadataRequest(run_id=run_id)
        interceptor.intercept_unary_unary(
            continuation=continuation_1,
            client_call_details=client_call_details_1,
            request=request_1,
        )
        interceptor.intercept_unary_unary(
            continuation=continuation_2,
            client_call_details=client_call_details_2,
            request=request_2,
        )

        # then
        request_id_1 = _get_request_id_from(client_call_details_1)
        request_id_2 = _get_request_id_from(client_call_details_2)
        assert request_id_1 is not None
        assert request_id_2 is not None
        assert request_id_1 != request_id_2


class TestProjectRun:
    def test_project_run_fails_when_max_active_run_exceeds(self) -> None:
        runner = ProjectRunner(
            config=MagicMock(),
            project_progress_tracker_factory=RemoteExecutionProjectProgressTracker,
        )
        error = rpc_error(
            metadata=(),
            code=grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        layer_client_exception = (
            GRPCErrorClientInterceptor._convert_rpc_error_to_client_exception(error)
        )

        client = MagicMock()
        client.flow_manager.start_run.side_effect = layer_client_exception
        project = MagicMock()
        project.name.return_value = "test"
        with pytest.raises(ProjectRunnerError, match=".*RESOURCE_EXHAUSTED.*"):
            runner._run(client=client, project=project, execution_plan=ExecutionPlan())


def _get_request_id_from(client_call_details: grpc.ClientCallDetails) -> Optional[str]:
    for pair in client_call_details.metadata:
        if pair[0] == "x-request-id":
            return pair[1]

    return None


def _mock_response(return_value: Any) -> MagicMock:
    mock_response = MagicMock()
    mock_response.result = MagicMock(return_value=return_value)
    return mock_response


def _client_call_details(method: str) -> grpc.ClientCallDetails:
    client_call_details = new_client_call_details(
        method=method,
        metadata=[
            ("Authorization", f"Bearer {str(uuid.uuid4())}"),
        ],
    )
    return client_call_details
