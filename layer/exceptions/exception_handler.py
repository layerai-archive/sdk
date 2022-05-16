import logging
import sys
import traceback
from logging import Logger
from typing import Callable

import grpc
import wrapt  # type: ignore

from .exceptions import RuntimeMemoryException


default_logger = logging.getLogger(__name__)


def exception_handler(stage: str, callback=None, logger: Logger = default_logger) -> Callable:  # type: ignore
    """
    Common utility to handle exceptions and report them to their respective backends via callbacks.

    :param stage: at which stage exception has been occurred.
    :param callback: a callback which would be called in case of exception to report it to a backend.
    If none provided and decorated function has callback keyword argument it would be used.
    :param logger: logger to log exceptions in case of structured logging configured for a module.
    Otherwise the default logger would be used.
    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore
        callback_override = callback if callback else kwargs.pop("callback", None)
        try:
            return wrapped(*args, **kwargs)
        except grpc.RpcError as rpc_error_call:
            failure_exc = _transform_grpc_exception(rpc_error_call)
            _log_exception_from_execution(logger)
        except Exception as e:
            _log_exception_from_execution(logger)
            failure_exc = e
            if isinstance(e, MemoryError):
                failure_exc = _transform_memory_error_exception(e)
        except SystemExit as e:
            _log_exception_from_execution(logger)
            failure_exc = _transform_system_exit_error(e)
        try:
            if callback_override:
                callback_override(stage, failure_exc)
        except Exception as e:
            reason = getattr(e, "message", repr(e))
            logger.error(
                f"Failure during reporting the failure: ({failure_exc}) : {reason}",
            )
            _log_exception_from_execution(logger)
        finally:
            sys.exit(1)

    return wrapper


def _transform_system_exit_error(e: SystemExit) -> Exception:
    # To populate stacktrace
    try:
        raise Exception(f"System exit with code:{e.code}")
    except Exception as ee:
        return ee


def _transform_memory_error_exception(e: MemoryError) -> Exception:
    # To populate stacktrace
    try:
        raise RuntimeMemoryException(str(e))
    except Exception as ee:
        return ee


def _transform_grpc_exception(exc: grpc.RpcError) -> Exception:
    """
    Done to populate sys.exc_info() with an exception containing the stacktrace
    of the input param exception, so that traceback.print_exc(), which we use
    to print debug logs, would print the complete exception stacktrace as intended.
    :param exc: The exception, whose error message we want to extract + retain its stacktrace
    """
    failure_reason = exc.details()
    try:
        raise Exception(failure_reason) from exc
    except Exception as e:
        return e


def _log_exception_from_execution(logger: Logger) -> None:
    logger.warning("Caught exception stacktrace:", exc_info=True)
    traceback.print_exc(file=sys.stdout)
