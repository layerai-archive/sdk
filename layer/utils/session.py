import os
import uuid
from typing import Any


_ENV_KEY_LAYER_DEBUG = "LAYER_DEBUG"
_ENV_KEY_REQUEST_ID = "REQUEST_ID"

_TRUTHY_LAYER_DEBUG_VALUES = [
    "True",
    "true",
    "1",
    1,
]


def is_layer_debug_on() -> bool:
    if _ENV_KEY_LAYER_DEBUG not in os.environ:
        return False

    return os.environ[_ENV_KEY_LAYER_DEBUG] in _TRUTHY_LAYER_DEBUG_VALUES


class UserSessionId:
    """
    UserSessionId is a singleton (only one instance per process,
    shared across threads) used to link all user requests by
    the same id.

    It will be used in grpc headers for remote interactions and
    propagated to downstream backend services to serve different
    purposes:
    - debugging a user journey: search DataDog logs by this id
    - reliably rely on retry mechanisms via idempotent requests
    """

    _instance = None

    def __new__(cls) -> Any:
        if cls._instance is None:
            id_ = uuid.uuid4()
            cls._instance = super(UserSessionId, cls).__new__(cls)
            cls._instance._value = id_  # type: ignore
        return cls._instance

    def __str__(self) -> str:
        return str(self._value)  # type: ignore
