import json
import logging
from traceback import FrameSummary

from layer.exceptions.status_report import (
    ExecutionStatusReportFactory,
    GenericExecutionStatusReport,
    PythonExecutionStatusReport,
)


logger = logging.getLogger(__name__)


def test_generic_status_report_to_json() -> None:
    message = "Test msg"
    expected_dict = {"message": message, "type": "Generic"}
    actual_json = ExecutionStatusReportFactory.to_json(
        GenericExecutionStatusReport(message)
    )
    actual_dict = json.loads(actual_json)
    assert expected_dict == actual_dict


def test_generic_status_report_from_json() -> None:
    message = "Test msg"
    json = f'{{"message": "{message}", "type": "Generic"}}'
    expected = GenericExecutionStatusReport(message)
    actual = ExecutionStatusReportFactory.from_json(json)
    assert expected == actual


def test_generic_status_report_from_incorrect_json() -> None:
    message = "foo bar"
    json = "foo bar"
    expected = GenericExecutionStatusReport(message)
    actual = ExecutionStatusReportFactory.from_json(json)
    assert expected == actual


def test_python_status_report_to_json() -> None:
    message = "Test msg"
    frames = [
        FrameSummary("file1", 1, "name1", line="line1"),
        FrameSummary("file2", 2, "name2", line="line2"),
    ]
    expected_dict = {
        "message": message,
        "type": "Python",
        "frames": [
            {"file": "file1", "line": "line1", "lineno": 1, "name": "name1"},
            {"file": "file2", "line": "line2", "lineno": 2, "name": "name2"},
        ],
    }

    actual_json = ExecutionStatusReportFactory.to_json(
        PythonExecutionStatusReport(message, frames)
    )
    actual_dict = json.loads(actual_json)
    assert expected_dict == actual_dict


def test_python_status_report_from_json() -> None:
    message = "Test msg"
    frames = [
        FrameSummary("file1", 1, "name1", line="line1"),
        FrameSummary("file2", 2, "name2", line="line2"),
    ]
    json = (
        f'{{"message": "{message}",'
        f'"type": "Python",'
        f'"frames": ['
        f'{{"file": "file1", "line": "line1", "lineno": 1, "name": "name1"}},'
        f'{{"file": "file2", "line": "line2", "lineno": 2, "name": "name2"}}'
        f"]}}"
    )
    expected = PythonExecutionStatusReport(message, frames)
    actual = ExecutionStatusReportFactory.from_json(json)
    assert expected == actual
