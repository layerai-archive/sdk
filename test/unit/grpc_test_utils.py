import collections
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import grpc


def new_client_call_details(
    method: str,
    metadata: List[Tuple[str, str]],
    timeout: float = 1.0,
    credentials: Any = None,
) -> grpc.ClientCallDetails:
    return _ClientCallDetails(method, metadata, timeout, credentials)


def rpc_error(
    metadata: Tuple[Any, ...],
    code: Optional[grpc.StatusCode] = grpc.StatusCode.INTERNAL,
    message: str = "there was an error",
) -> grpc.RpcError:
    error = grpc.RpcError()
    error.details = MagicMock(return_value=message)
    error.code = MagicMock(return_value=code)
    error.trailing_metadata = MagicMock(return_value=metadata)
    return error


class _ClientCallDetails(
    collections.namedtuple(
        "_ClientCallDetails", ("method", "metadata", "timeout", "credentials")
    ),
    grpc.ClientCallDetails,
):
    pass
