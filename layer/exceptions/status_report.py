import json
import traceback
from abc import ABC
from os import path
from pathlib import Path
from traceback import FrameSummary
from typing import Any, List, Optional

from layer.contracts.assertions import Assertion


class ExecutionStatusReport(ABC):
    @property
    def message(self) -> str:
        return ""

    @property
    def cause(self) -> str:
        return ""


class GenericExecutionStatusReport(ExecutionStatusReport):
    def __init__(self, message: str):
        self._message = message

    @property
    def message(self) -> str:
        return self._message

    @property
    def cause(self) -> str:
        return ""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GenericExecutionStatusReport):
            return False
        return self.message == other.message


class PythonExecutionStatusReport(ExecutionStatusReport):
    def __init__(
        self,
        message: str,
        frames: List[FrameSummary],
        source_dir: Optional[Path] = None,
    ):
        if source_dir is not None:
            expanded_source_dir = path.expanduser(source_dir)
            self._message = self._strip_source_dir(message, expanded_source_dir)
            self._frames = [
                FrameSummary(
                    self._strip_source_dir(frame.filename, expanded_source_dir),
                    frame.lineno,
                    frame.name,
                    line=frame.line,
                )
                for frame in frames
            ]
        else:
            self._message = message
            self._frames = frames
        self._source_dir = source_dir

    @property
    def message(self) -> str:
        return self._message

    @property
    def frames(self) -> List[FrameSummary]:
        return self._frames

    @property
    def cause(self) -> str:
        return self._format_frame(self._frames[-1])

    @staticmethod
    def _format_frame(frame: FrameSummary) -> str:
        return f'File "{frame.filename}" on line {frame.lineno}: "{frame.line}"'

    @staticmethod
    def from_exception(
        exc: Exception, source_dir: Optional[Path] = None
    ) -> "PythonExecutionStatusReport":
        return PythonExecutionStatusReport(
            message=str(exc),
            frames=traceback.extract_tb(exc.__traceback__),
            source_dir=source_dir,
        )

    @staticmethod
    def _strip_source_dir(string: str, source_dir: str) -> str:
        return string.replace(source_dir + path.sep, "")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PythonExecutionStatusReport):
            return False
        return self.message == other.message and self.frames == other.frames


class AssertionFailureStatusReport(ExecutionStatusReport):
    def __init__(
        self,
        failed_assertions: Optional[List[Assertion]] = None,
        message: Optional[str] = None,
    ):
        if failed_assertions:
            stringified = ", ".join([str(assertion) for assertion in failed_assertions])
            self._message = f"failed: {stringified}"
        else:
            assert message
            self._message = message

    @property
    def message(self) -> str:
        return self._message

    @property
    def cause(self) -> str:
        return ""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AssertionFailureStatusReport):
            return False
        return self._message == other._message


class ExecutionStatusReportFactory:
    @staticmethod
    def from_json(json_str: str) -> ExecutionStatusReport:
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            return GenericExecutionStatusReport(json_str)
        report_type = payload["type"].lower()
        if report_type == "python":
            message: str = payload["message"]
            frames: List[FrameSummary] = [
                FrameSummary(
                    filename=f["file"],
                    lineno=f["lineno"],
                    name=f["name"],
                    line=f["line"],
                )
                for f in payload["frames"]
            ]
            return PythonExecutionStatusReport(
                message=message,
                frames=frames,
            )
        elif report_type == "generic":
            return GenericExecutionStatusReport(payload["message"])
        elif report_type == "assertion":
            return AssertionFailureStatusReport(message=payload["message"])
        else:
            raise Exception("Invalid status report type")

    @staticmethod
    def to_json(report: ExecutionStatusReport) -> str:
        if isinstance(report, PythonExecutionStatusReport):
            dic_py = {
                "type": "Python",
                "message": report.message,
                "frames": [
                    {
                        "file": f.filename,
                        "line": f.line,
                        "lineno": f.lineno,
                        "name": f.name,
                    }
                    for f in report.frames
                ],
            }
            return json.dumps(dic_py)
        elif isinstance(report, GenericExecutionStatusReport):
            dic_generic = {"type": "Generic", "message": report.message}
            return json.dumps(dic_generic)
        elif isinstance(report, AssertionFailureStatusReport):
            dic_generic = {"type": "Assertion", "message": report.message}
            return json.dumps(dic_generic)
        else:
            raise Exception("Unsupported report type")
