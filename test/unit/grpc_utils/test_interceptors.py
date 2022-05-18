import json
import os
import uuid
from collections import namedtuple
from pathlib import Path
from test.unit.grpc_test_utils import new_client_call_details, rpc_error
from typing import Any, Optional, Tuple
from unittest.mock import MagicMock

import grpc
import pytest
from google.protobuf.json_format import ParseDict
from grpc._cython.cygrpc import _Metadatum
from layerapi.api.ids_pb2 import RunId
from layerapi.api.service.flowmanager.flow_manager_api_pb2 import (
    StartRunV2Request,
    StartRunV2Response,
    TerminateRunRequest,
    TerminateRunResponse,
)

from layer.exceptions.exceptions import (
    LayerClientResourceAlreadyExistsException,
    LayerClientResourceNotFoundException,
    LayerClientTimeoutException,
    LayerResourceExhaustedException,
)
from layer.utils.grpc.interceptors import (
    _OBFUSCATED_VALUE,
    GRPCErrorClientInterceptor,
    LogRpcCallsInterceptor,
)
from layer.utils.session import _ENV_KEY_LAYER_DEBUG


class TestLogRpcCallsInterceptor:
    @pytest.fixture()
    def _cleanup_after(self) -> None:
        yield
        LogRpcCallsInterceptor._clear_instance()
        if _ENV_KEY_LAYER_DEBUG in os.environ:
            del os.environ[_ENV_KEY_LAYER_DEBUG]

    @pytest.mark.usefixtures("_cleanup_after")
    def test_is_singleton(self, tmp_path: Path):
        logs_file_path = Path(tmp_path) / "test.log"

        assert LogRpcCallsInterceptor(logs_file_path) == LogRpcCallsInterceptor(
            logs_file_path
        )

    @pytest.mark.usefixtures("_cleanup_after")
    def test_raises_value_error_for_different_logs_file_paths(self, tmp_path: Path):
        logs_dir_1 = Path(tmp_path) / "1.log"
        logs_dir_2 = Path(tmp_path) / "2.log"

        LogRpcCallsInterceptor(logs_dir_1)

        with pytest.raises(
            ValueError, match=r".*already exists with different file path.*"
        ):
            LogRpcCallsInterceptor(logs_dir_2)

    def test__obfuscate_dict_values_works_recursively(self):
        raw = {
            "a": {
                "a.a": {
                    "a.a.a": "raw",
                    "a.a.b": {"a.a.b.a": "raw"},
                },
                "a.b": 3,
            },
            "b": 4.5,
        }

        obfuscated = LogRpcCallsInterceptor._obfuscate_dict_values(raw)

        assert obfuscated["a"]["a.a"]["a.a.a"] == _OBFUSCATED_VALUE
        assert obfuscated["a"]["a.a"]["a.a.b"]["a.a.b.a"] == _OBFUSCATED_VALUE
        assert obfuscated["a"]["a.b"] == _OBFUSCATED_VALUE
        assert obfuscated["b"] == _OBFUSCATED_VALUE

    @pytest.mark.usefixtures("_cleanup_after")
    def test_when_layer_debug_interceptor_logs_rpc_call_with_clear_response_and_opt_in_metadata_visible(
        self, tmp_path: Path
    ):
        # given
        request_id = uuid.uuid4()
        client_call_details = _client_call_details_with_request_id_and_auth_metadata(
            "/api.FlowManagerAPI/TerminateRun", request_id
        )
        run_id = RunId(value=str(uuid.uuid4()))
        request = TerminateRunRequest(run_id=run_id)
        mock_response = _mock_response(TerminateRunResponse(run_id=run_id))
        continuation = MagicMock(return_value=mock_response)

        # when
        os.environ[_ENV_KEY_LAYER_DEBUG] = "1"
        logs_file_path = Path(tmp_path) / "logs" / "test.log"
        interceptor = LogRpcCallsInterceptor(logs_file_path)
        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=client_call_details,
            request=request,
        )
        interceptor.close()

        # then
        continuation.assert_called_with(client_call_details, request)
        assert isinstance(response.result(), TerminateRunResponse)

        assert logs_file_path.parent.is_dir()
        with open(logs_file_path, "r") as f:
            content = f.read()
            deserialized = json.loads(content)
            assert deserialized["method"] == "/api.FlowManagerAPI/TerminateRun"
            assert deserialized["request_metadata"]["x-request-id"] == str(request_id)
            assert (
                deserialized["request_metadata"]["Authorization"] == _OBFUSCATED_VALUE
            )
            request_message = TerminateRunRequest()
            request_message = ParseDict(deserialized["request"], request_message)
            assert request_message.run_id == request.run_id
            response_message = TerminateRunResponse()
            response_message = ParseDict(deserialized["result"], response_message)
            assert response_message.run_id == response.result().run_id

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_logs_rpc_call_with_obfuscated_response_and_opt_in_metadata_visible(
        self, tmp_path: Path
    ):
        # given
        request_id = uuid.uuid4()
        client_call_details = _client_call_details_with_request_id_and_auth_metadata(
            "/api.FlowManagerAPI/TerminateRun", request_id
        )
        run_id = RunId(value=str(uuid.uuid4()))
        request = TerminateRunRequest(run_id=run_id)
        mock_response = _mock_response(TerminateRunResponse(run_id=run_id))
        continuation = MagicMock(return_value=mock_response)

        # when
        logs_file_path = Path(tmp_path) / "logs" / "test.log"
        interceptor = LogRpcCallsInterceptor(logs_file_path)
        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=client_call_details,
            request=request,
        )
        interceptor.close()

        # then
        continuation.assert_called_with(client_call_details, request)
        assert isinstance(response.result(), TerminateRunResponse)

        assert logs_file_path.parent.is_dir()
        with open(logs_file_path, "r") as f:
            content = f.read()
            deserialized = json.loads(content)
            assert deserialized["method"] == "/api.FlowManagerAPI/TerminateRun"
            assert deserialized["request_metadata"]["x-request-id"] == str(request_id)
            assert (
                deserialized["request_metadata"]["Authorization"] == _OBFUSCATED_VALUE
            )
            request_message = TerminateRunRequest()
            request_message = ParseDict(deserialized["request"], request_message)
            assert request_message.run_id == request.run_id
            response_message = TerminateRunResponse()
            response_message = ParseDict(deserialized["result"], response_message)
            assert response_message.run_id.value == _OBFUSCATED_VALUE

    @pytest.mark.usefixtures("_cleanup_after")
    def when_layer_debug_test_interceptor_logs_rpc_call_with_exception_and_metadata_if_rpc_error_from_outcome(
        self, tmp_path: Path
    ):
        # given
        request_id = uuid.uuid4()
        client_call_details = _client_call_details_with_request_id_and_auth_metadata(
            "/api.FlowManagerAPI/TerminateRun", request_id
        )
        run_id = RunId(value=str(uuid.uuid4()))
        request = TerminateRunRequest(run_id=run_id)
        rpc_error = _new_rpc_error(
            grpc.StatusCode.UNAVAILABLE,
            "oh no",
            (
                _Metadatum(key="x-request-id", value=str(request_id)),
                _Metadatum(key="x-random", value="should be obfuscated"),
            ),
        )
        mock_response = _mock_grpc_exception(rpc_error)
        continuation = MagicMock(return_value=mock_response)

        # when
        os.environ[_ENV_KEY_LAYER_DEBUG] = "1"
        logs_file_path = Path(tmp_path) / "logs" / "test.log"
        interceptor = LogRpcCallsInterceptor(logs_file_path)
        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=client_call_details,
            request=request,
        )
        interceptor.close()

        # then
        continuation.assert_called_with(client_call_details, request)
        assert response.result() is None
        assert isinstance(response.exception(), grpc.RpcError)
        with open(logs_file_path, "r") as f:
            content = f.read()
            deserialized = json.loads(content)
            assert deserialized["method"] == "/api.FlowManagerAPI/TerminateRun"
            assert deserialized["request_metadata"]["x-request-id"] == str(request_id)
            assert (
                deserialized["request_metadata"]["Authorization"] == _OBFUSCATED_VALUE
            )
            request_message = TerminateRunRequest()
            request_message = ParseDict(deserialized["request"], request_message)
            assert request_message.run_id == request.run_id
            assert "result" not in deserialized
            assert deserialized["exception"]["type"] == "RpcError"
            assert deserialized["exception"]["code"] == "UNAVAILABLE"
            assert deserialized["exception"]["details"] == "oh no"
            assert deserialized["result_metadata"]["x-request-id"] == str(request_id)
            assert deserialized["result_metadata"]["x-random"] == _OBFUSCATED_VALUE

    @pytest.mark.usefixtures("_cleanup_after")
    def when_not_layer_debug_test_interceptor_logs_rpc_call_with_obfuscated_exception_if_exception_from_outcome(
        self, tmp_path: Path
    ):
        # given
        request_id = uuid.uuid4()
        client_call_details = _client_call_details_with_request_id_and_auth_metadata(
            "/api.FlowManagerAPI/TerminateRun", request_id
        )
        run_id = RunId(value=str(uuid.uuid4()))
        request = TerminateRunRequest(run_id=run_id)
        exception = ValueError("oh no")
        mock_response = _mock_grpc_exception(exception)
        continuation = MagicMock(return_value=mock_response)

        # when
        logs_file_path = Path(tmp_path) / "logs" / "test.log"
        interceptor = LogRpcCallsInterceptor(logs_file_path)
        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=client_call_details,
            request=request,
        )
        interceptor.close()

        # then
        continuation.assert_called_with(client_call_details, request)
        assert response.result() is None
        assert isinstance(response.exception(), ValueError)
        with open(logs_file_path, "r") as f:
            content = f.read()
            deserialized = json.loads(content)
            assert deserialized["method"] == "/api.FlowManagerAPI/TerminateRun"
            assert deserialized["request_metadata"]["x-request-id"] == str(request_id)
            assert (
                deserialized["request_metadata"]["Authorization"] == _OBFUSCATED_VALUE
            )
            request_message = TerminateRunRequest()
            request_message = ParseDict(deserialized["request"], request_message)
            assert request_message.run_id == request.run_id

            assert "result" not in deserialized
            assert deserialized["exception"]["type"] == "ValueError"
            assert "code" not in deserialized["exception"]
            assert deserialized["exception"]["details"] == _OBFUSCATED_VALUE

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_logs_rpc_call_with_exception_if_rpc_error_raised_and_rethrows(
        self, tmp_path: Path
    ):
        # given
        request_id = uuid.uuid4()
        client_call_details = _client_call_details_with_request_id_and_auth_metadata(
            "/api.FlowManagerAPI/TerminateRun", request_id
        )
        run_id = RunId(value=str(uuid.uuid4()))
        request = TerminateRunRequest(run_id=run_id)
        rpc_error_without_trailing_metadata = _new_rpc_error(
            grpc.StatusCode.UNAVAILABLE, "oh no"
        )
        continuation = MagicMock()
        continuation.side_effect = rpc_error_without_trailing_metadata

        # when
        logs_file_path = Path(tmp_path) / "logs" / "test.log"
        interceptor = LogRpcCallsInterceptor(logs_file_path)
        with pytest.raises(grpc.RpcError, match=".*"):
            interceptor.intercept_unary_unary(
                continuation=continuation,
                client_call_details=client_call_details,
                request=request,
            )
            interceptor.close()

        # then
        continuation.assert_called_with(client_call_details, request)
        with open(logs_file_path, "r") as f:
            content = f.read()
            deserialized = json.loads(content)
            assert deserialized["method"] == "/api.FlowManagerAPI/TerminateRun"
            assert deserialized["request_metadata"]["x-request-id"] == str(request_id)
            assert (
                deserialized["request_metadata"]["Authorization"] == _OBFUSCATED_VALUE
            )
            request_message = TerminateRunRequest()
            request_message = ParseDict(deserialized["request"], request_message)
            assert request_message.run_id == request.run_id

            assert "result" not in deserialized
            assert deserialized["exception"]["type"] == "RpcError"
            assert deserialized["exception"]["code"] == "UNAVAILABLE"
            assert deserialized["exception"]["details"] == _OBFUSCATED_VALUE
            assert deserialized["result_metadata"] is None

    @pytest.mark.usefixtures("_cleanup_after")
    def test_interceptor_logs_all_rpc_calls(self, tmp_path: Path):
        # given
        logs_file_path = Path(tmp_path) / "test.log"
        interceptor = LogRpcCallsInterceptor(logs_file_path)

        # when
        # call 1
        run_id = RunId(value=str(uuid.uuid4()))
        _intercept_call_for_method(
            interceptor,
            "/api.FlowManagerAPI/StartRun",
            request=StartRunV2Request(project_name="lala"),
            result=StartRunV2Response(run_id=run_id),
        )
        # call 2
        _intercept_call_for_method(
            interceptor,
            "/api.FlowManagerAPI/TerminateRun",
            request=TerminateRunRequest(run_id=run_id),
            result=TerminateRunResponse(run_id=run_id),
        )
        interceptor.close()

        # then
        with open(logs_file_path, "r") as f:
            assert "/api.FlowManagerAPI/StartRun" in f.readline()
            assert "/api.FlowManagerAPI/TerminateRun" in f.readline()


class TestGRPCErrorClientInterceptor:
    def test_convert_rpc_error_to_client_exception_without_x_request_id(self):
        error = rpc_error(metadata=())
        layer_client_exception = (
            GRPCErrorClientInterceptor._convert_rpc_error_to_client_exception(error)
        )
        assert (
            str(layer_client_exception) == "code: INTERNAL, details: there was an error"
        )

    def test_convert_rpc_error_to_client_exception(self):
        _Metadatum = namedtuple("_Metadatum", ["key", "value"])
        error = rpc_error((_Metadatum(key="x-request-id", value="xyz-123"),))
        layer_client_exception = (
            GRPCErrorClientInterceptor._convert_rpc_error_to_client_exception(error)
        )
        assert (
            str(layer_client_exception)
            == "error id: xyz-123, code: INTERNAL, details: there was an error"
        )

    def test_convert_deadline_exceeded_to_client_timeout_exception(self):
        _Metadatum = namedtuple("_Metadatum", ["key", "value"])
        error = rpc_error(
            (_Metadatum(key="x-request-id", value="xyz-123"),),
            grpc.StatusCode.DEADLINE_EXCEEDED,
        )
        layer_client_exception = (
            GRPCErrorClientInterceptor._convert_rpc_error_to_client_exception(error)
        )
        assert isinstance(layer_client_exception, LayerClientTimeoutException)
        assert (
            str(layer_client_exception)
            == "error id: xyz-123, code: DEADLINE_EXCEEDED, details: there was an error"
        )

    @pytest.mark.parametrize(
        ("status_code", "expected_client_exception_type"),
        [
            (
                grpc.StatusCode.ALREADY_EXISTS,
                LayerClientResourceAlreadyExistsException,
            ),
            (
                grpc.StatusCode.NOT_FOUND,
                LayerClientResourceNotFoundException,
            ),
        ],
    )
    def test_convert_already_exists_to_client_already_exists_exception(
        self,
        status_code: grpc.StatusCode,
        expected_client_exception_type: type,
    ):
        _Metadatum = namedtuple("_Metadatum", ["key", "value"])
        error = rpc_error(
            (_Metadatum(key="x-request-id", value="xyz-123"),),
            status_code,
            "resource lala",
        )
        layer_client_exception = (
            GRPCErrorClientInterceptor._convert_rpc_error_to_client_exception(error)
        )
        assert isinstance(layer_client_exception, expected_client_exception_type)
        assert str(layer_client_exception) == "resource lala"

    def test_convert_resource_exhausted_to_client_max_active_runs_exceeded_exception(
        self,
    ):
        _Metadatum = namedtuple("_Metadatum", ["key", "value"])
        error = rpc_error(
            (_Metadatum(key="x-request-id", value="xyz-123"),),
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        )
        layer_client_exception = (
            GRPCErrorClientInterceptor._convert_rpc_error_to_client_exception(error)
        )
        assert isinstance(layer_client_exception, LayerResourceExhaustedException)
        assert (
            str(layer_client_exception)
            == "error id: xyz-123, code: RESOURCE_EXHAUSTED, details: there was an error"
        )


def _client_call_details_with_request_id_and_auth_metadata(
    method: str, request_id: Optional[uuid.UUID] = None
) -> grpc.ClientCallDetails:
    if request_id is None:
        request_id = uuid.uuid4()
    client_call_details = new_client_call_details(
        method=method,
        metadata=[
            ("x-request-id", str(request_id)),
            ("Authorization", f"Bearer {str(uuid.uuid4())}"),
        ],
    )
    return client_call_details


def _intercept_call_for_method(
    interceptor: LogRpcCallsInterceptor, method: str, request: Any, result: Any
) -> None:
    client_call_details = _client_call_details_with_request_id_and_auth_metadata(method)
    mock_response = _mock_response(result)
    continuation = MagicMock(return_value=mock_response)

    # when
    interceptor.intercept_unary_unary(
        continuation=continuation,
        client_call_details=client_call_details,
        request=request,
    )

    continuation.assert_called_with(client_call_details, request)


def _new_rpc_error(
    code: grpc.StatusCode,
    details: Optional[str] = None,
    trailing_metadata: Optional[Tuple[_Metadatum, ...]] = None,
) -> grpc.RpcError:
    rpc_error = grpc.RpcError()
    rpc_error.code = lambda: code
    rpc_error.details = lambda: details
    rpc_error.trailing_metadata = lambda: trailing_metadata
    return rpc_error


def _mock_response(return_value: Any) -> MagicMock:
    mock_response = MagicMock()
    mock_response.result = MagicMock(return_value=return_value)
    return mock_response


def _mock_grpc_exception(ex: Exception) -> MagicMock:
    mock_response = MagicMock()
    mock_response.result = lambda: None
    mock_response.exception = MagicMock(return_value=ex)
    return mock_response
