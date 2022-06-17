import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import grpc
from google.protobuf.json_format import MessageToDict
from grpc._cython.cygrpc import _Metadatum  # type: ignore

from layer.exceptions.exceptions import (
    LayerClientException,
    LayerClientResourceAlreadyExistsException,
    LayerClientResourceNotFoundException,
    LayerClientServiceUnavailableException,
    LayerClientTimeoutException,
    LayerResourceExhaustedException,
)
from layer.utils.session import _ENV_KEY_REQUEST_ID, UserSessionId, is_layer_debug_on


_OBFUSCATED_VALUE = "***"
_NON_OBFUSCATED_REQUEST_METADATA = [
    "x-request-id",
]


class GRPCErrorClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        error: Optional[grpc.RpcError] = None
        try:
            outcome = continuation(client_call_details, request)
        except grpc.RpcError as err:
            error = err
        if outcome.exception() and isinstance(outcome.exception(), grpc.RpcError):
            error = outcome.exception()

        if error:
            if "Exception deserializing response" in str(error.details()):
                raise LayerClientException(
                    "Please make sure that you have the latest layer sdk version installed."
                )
            else:
                raise self._convert_rpc_error_to_client_exception(error)
        return outcome

    @staticmethod
    def _convert_rpc_error_to_client_exception(
        error: grpc.RpcError,
    ) -> Union[LayerClientException, grpc.RpcError]:
        error_details = str(error.details())
        request_id = ""
        for metadata in error.trailing_metadata():
            if metadata.key == "x-request-id":
                request_id = str(metadata.value)
                break
        client_error_message = f"code: {error.code().name}, details: {error_details}"
        if len(request_id) > 0:
            client_error_message = f"error id: {request_id}, {client_error_message}"

        if error.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
            return LayerClientTimeoutException(client_error_message)
        elif error.code() is grpc.StatusCode.RESOURCE_EXHAUSTED:
            return LayerResourceExhaustedException(client_error_message)
        elif error.code() is grpc.StatusCode.UNAVAILABLE:
            return LayerClientServiceUnavailableException(error_details)
        elif error.code() is grpc.StatusCode.NOT_FOUND:
            return LayerClientResourceNotFoundException(error_details)
        elif error.code() is grpc.StatusCode.ALREADY_EXISTS:
            return LayerClientResourceAlreadyExistsException(error_details)
        return LayerClientException(client_error_message)


class LogRpcCallsInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """
    _LogRpcCallsInterceptor will log all gRPC calls with obfuscated responses to
    the session logs.
    Metadata keys specified in _NON_OBFUSCATED_REQUEST_METADATA are not obfuscated.

    With LAYER_DEBUG environment variable set to truthy values ("True", "1"), responses
    will NOT be obfuscated.
    """

    _instance = None

    def __new__(cls, logs_file_path: Path) -> Any:
        if cls._instance is None:
            import atexit

            cls._instance = super(LogRpcCallsInterceptor, cls).__new__(cls)
            atexit.register(cls._instance.close)

            user_session_id = str(UserSessionId())
            cls._instance._user_session_id = user_session_id  # type: ignore
            cls.should_obfuscate_responses = not is_layer_debug_on()
            cls._instance._logs_file_path = logs_file_path  # type: ignore
            logs_file_path.parent.mkdir(parents=True, exist_ok=True)
            cls._instance._log_file = open(logs_file_path, "w")  # type: ignore
        elif logs_file_path != cls._instance._logs_file_path:  # type: ignore
            raise ValueError(
                "LogRpcCallsInterceptor instance already exists with different file path"
            )

        return cls._instance

    @classmethod
    def _clear_instance(cls) -> None:
        cls._instance = None

    def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        started_utc = datetime.datetime.now(datetime.timezone.utc).timestamp()
        log_entry = {
            "started_utc": started_utc,
            "method": client_call_details.method,
            "request": MessageToDict(request),
        }

        if client_call_details.metadata is not None:
            obfuscated_metadata = self._obfuscate_metadata(client_call_details.metadata)
            log_entry["request_metadata"] = obfuscated_metadata

        try:
            outcome = continuation(client_call_details, request)
            if outcome.exception():
                err = outcome.exception()
                self._add_exception_to_log_entry(err, log_entry)
            if outcome.result():
                self._add_result_to_log_entry(outcome.result(), log_entry)
            log_entry["duration"] = (
                datetime.datetime.now(datetime.timezone.utc).timestamp() - started_utc
            )
            self._print_json_as_new_line(log_entry)
            return outcome
        except Exception as err:
            self._add_exception_to_log_entry(err, log_entry)
            log_entry["duration"] = (
                datetime.datetime.now(datetime.timezone.utc).timestamp() - started_utc
            )
            self._print_json_as_new_line(log_entry)
            raise err

    @staticmethod
    def _obfuscate_metadata(
        metadata: Tuple[Tuple[str, Union[str, bytes]], ...]
    ) -> Dict[str, Union[str, bytes]]:

        return {
            k: v if k in _NON_OBFUSCATED_REQUEST_METADATA else _OBFUSCATED_VALUE
            for k, v in metadata
        }

    @staticmethod
    def _obfuscate_trailing_metadata(
        metadata: Tuple[_Metadatum, ...]  # pytype: disable=invalid-annotation
    ) -> Dict[str, Union[str, bytes]]:
        return {
            m.key: m.value
            if m.key in _NON_OBFUSCATED_REQUEST_METADATA
            else _OBFUSCATED_VALUE
            for m in metadata
        }

    def _add_result_to_log_entry(self, result: Any, log_entry: Dict[str, Any]) -> None:
        result_dict = MessageToDict(result)
        if self.should_obfuscate_responses:
            log_entry["result"] = self._obfuscate_dict_values(result_dict)
        else:
            log_entry["result"] = result_dict

    @staticmethod
    def _obfuscate_dict_values(raw: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in raw.items():
            out[k] = (
                LogRpcCallsInterceptor._obfuscate_dict_values(v)
                if isinstance(v, dict)
                else _OBFUSCATED_VALUE
            )
        return out

    def _add_exception_to_log_entry(
        self,
        exception: Exception,
        log_entry: Dict[str, Any],
    ) -> None:
        exception_as_dict = {
            "type": exception.__class__.__name__,
            "details": _OBFUSCATED_VALUE,
        }
        metadata: Optional[Dict[str, Any]] = None

        if isinstance(exception, grpc.RpcError):
            exception_as_dict[
                "code"
            ] = exception.code().name  # pytype: disable=attribute-error

            if not self.should_obfuscate_responses:
                exception_as_dict[
                    "details"
                ] = exception.details()  # pytype: disable=attribute-error

            if exception.trailing_metadata():  # pytype: disable=attribute-error
                metadata = LogRpcCallsInterceptor._obfuscate_trailing_metadata(
                    exception.trailing_metadata()  # pytype: disable=attribute-error
                )
        elif not self.should_obfuscate_responses:
            exception_as_dict["details"] = str(exception)

        log_entry["exception"] = exception_as_dict
        log_entry["result_metadata"] = metadata

    def _print_json_as_new_line(self, entry: Dict[str, Any]) -> None:
        print(json.dumps(entry), file=self._log_file)  # type: ignore
        self._flush()

    def close(self) -> None:
        if self._log_file:  # type: ignore
            self._log_file.close()  # type: ignore

    def _flush(self) -> None:
        if self._log_file:  # type: ignore
            self._log_file.flush()  # type: ignore


class RequestIdInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    _instance = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super(RequestIdInterceptor, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        # REQUEST_ID set by invoker if wants to use same request id for all requests
        request_id_candidate = os.getenv(_ENV_KEY_REQUEST_ID, default=None)
        if request_id_candidate is None:
            self._request_id = None
            return

        try:
            self._request_id = uuid.UUID(request_id_candidate)
        except Exception as e:
            raise ValueError(
                f"invalid {_ENV_KEY_REQUEST_ID} environment variable. Error message: {e}"
            )

    @classmethod
    def _clear_instance(cls) -> None:
        cls._instance = None

    def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Any,
    ) -> Any:
        metadata: List[Tuple[str, str]] = (
            client_call_details.metadata if client_call_details.metadata else []  # type: ignore
        )
        if self._request_id is not None:
            request_id = self._request_id
        else:
            request_id = uuid.uuid4()
        metadata.append(("x-request-id", str(request_id)))
        client_call_details = client_call_details._replace(metadata=metadata)  # type: ignore
        return continuation(client_call_details, request)
