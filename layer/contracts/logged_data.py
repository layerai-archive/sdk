from dataclasses import dataclass
from enum import Enum, unique


@unique
class LoggedDataType(Enum):
    INVALID = 0
    TEXT = 1
    TABLE = 2
    BLOB = 3
    NUMBER = 4
    BOOLEAN = 5
    IMAGE = 6
    VIDEO = 7
    MARKDOWN = 8


@dataclass(frozen=True)
class LoggedData:
    logged_data_type: LoggedDataType
    tag: str
    data: str


@dataclass(frozen=True)
class ModelMetricPoint:
    epoch: int
    value: float


@dataclass(frozen=True)
class Markdown:
    """
    Encapsulates a markdown text to be displayed correctly in Layer UI as part of logging.

    .. code-block:: python

        layer.log(layer.Markdown("## Hello world!"))

    """

    data: str
