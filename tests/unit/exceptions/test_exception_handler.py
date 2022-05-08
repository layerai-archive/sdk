import re
import sys

import pytest

from layer.exceptions import exception_handler
from layer.exceptions.exceptions import RuntimeMemoryException, SparkRuntimeException


class CallbackClass:
    def __init__(self):
        self.callback_called = False
        self.stage = ""
        self.reason = ""

    def callback(self, stage: str, reason: str):
        self.callback_called = True
        self.stage = stage
        self.reason = reason


callback_called = False
stage = ""
reason = ""


def callback(stage_param: str, reason_param: str) -> None:
    global callback_called, stage, reason
    callback_called = True
    stage = stage_param
    reason = reason_param


def test_exception_handler_with_function_callback() -> None:
    @exception_handler("test_1", callback=callback)
    def fut() -> None:
        raise Exception("failure")

    with pytest.raises(SystemExit):
        fut()

    assert callback_called
    assert stage == "test_1"
    assert str(reason) == str(Exception("failure"))


def test_exception_handler_with_method_callback() -> None:
    callback_instance = CallbackClass()

    @exception_handler("test_2", callback=callback_instance.callback)
    def fut() -> None:
        raise Exception("failure_2")

    with pytest.raises(SystemExit):
        fut()

    assert callback_instance.callback_called
    assert callback_instance.stage == "test_2"
    assert str(callback_instance.reason) == str(Exception("failure_2"))


def test_exception_handler_with_system_exit() -> None:
    callback_instance = CallbackClass()

    @exception_handler("test_2", callback=callback_instance.callback)
    def fut() -> None:
        sys.exit(11)

    with pytest.raises(SystemExit):
        fut()

    assert callback_instance.callback_called
    assert callback_instance.stage == "test_2"
    assert str(callback_instance.reason) == str(Exception("System exit with code:11"))


def test_exception_handler_decorates_method() -> None:
    callback_instance = CallbackClass()

    class Fut:
        @exception_handler("test", callback=callback_instance.callback)
        def fut(self) -> None:
            raise Exception("failure")

    with pytest.raises(SystemExit):
        Fut().fut()

    assert callback_instance.callback_called
    assert callback_instance.stage == "test"
    assert str(callback_instance.reason) == str(Exception("failure"))


class Py4JJavaError(Exception):
    pass


def test_exception_dataset_not_found() -> None:
    callback_instance = CallbackClass()
    dataset_name = "missingdatasetinvalid"
    example_spark_error = f"""Error: Train d9b48611-782f-42bf-a6a4-c54ea8de7ada failed with "An error occurred while calling o97.load.
    : com.layer.shadow.arrow.flight.FlightRuntimeException: NOT_FOUND: Dataset not found. Name: '{dataset_name}'
        at com.layer.shadow.arrow.flight.CallStatus.toRuntimeException(CallStatus.java:131)
    """

    @exception_handler("test_2", callback=callback_instance.callback)
    def fut() -> None:
        raise Py4JJavaError(example_spark_error)

    with pytest.raises(SystemExit):
        fut()

    assert callback_instance.callback_called
    assert callback_instance.stage == "test_2"
    assert str(callback_instance.reason) == str(
        SparkRuntimeException(f"Missing dataset `{dataset_name}`")
    )


def test_exception_memory_error_thrown() -> None:
    callback_instance = CallbackClass()
    exc_message = "Out of available memory"

    @exception_handler("test_3", callback=callback_instance.callback)
    def fut() -> None:
        raise MemoryError(exc_message)

    with pytest.raises(SystemExit):
        fut()

    assert callback_instance.callback_called
    assert callback_instance.stage == "test_3"
    assert str(callback_instance.reason) == str(
        RuntimeMemoryException(str(MemoryError(exc_message)))
    )


def test_exception_missing_dependency_error_thrown() -> None:
    callback_instance = CallbackClass()
    exc_message_pattern = re.compile(".*No module named 'missing_dep'.*")

    @exception_handler("test_4", callback=callback_instance.callback)
    def fut() -> None:
        import importlib

        importlib.import_module("missing_dep")
        return

    with pytest.raises(SystemExit):
        fut()

    assert callback_instance.callback_called
    assert callback_instance.stage == "test_4"
    assert exc_message_pattern.match(str(callback_instance.reason))
