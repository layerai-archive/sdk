import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from layerapi.api.entity.run_pb2 import Run as PBRun


GetRunsFunction = Callable[[], List[PBRun]]


@dataclass(frozen=True)
class Run:
    """
    Provides access to project runs stored in Layer.

    You can retrieve an instance of this object with :code:`layer.run()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Runs the current project with the given functions
        layer.run([build_dataset, train_model])

    """

    project_name: str
    run_id: uuid.UUID = field(repr=False)


class ResourceTransferState:
    def __init__(self, name: Optional[str] = None) -> None:
        self._name: Optional[str] = name
        self._total_num_files: int = 0
        self._transferred_num_files: int = 0
        self._total_resource_size_bytes: int = 0
        self._transferred_resource_size_bytes: int = 0
        self._timestamp_to_bytes_sent: Dict[Any, int] = defaultdict(int)
        self._how_many_previous_seconds = 20

    def increment_num_transferred_files(self, increment_with: int) -> None:
        self._transferred_num_files += increment_with

    def increment_transferred_resource_size_bytes(self, increment_with: int) -> None:
        self._transferred_resource_size_bytes += increment_with
        self._timestamp_to_bytes_sent[self._get_current_timestamp()] += increment_with

    def get_bandwidth_in_previous_seconds(self) -> int:
        curr_timestamp = self._get_current_timestamp()
        valid_seconds = 0
        bytes_transferred = 0
        # Exclude start and current sec as otherwise result would not account for data that will be sent
        # within current second after the call of this func
        for i in range(1, self._how_many_previous_seconds + 1):
            look_at = curr_timestamp - i
            if look_at >= 0 and look_at in self._timestamp_to_bytes_sent:
                valid_seconds += 1
                bytes_transferred += self._timestamp_to_bytes_sent[look_at]
        if valid_seconds == 0:
            return 0
        return round(bytes_transferred / valid_seconds)

    def get_eta_seconds(self) -> int:
        bandwidth = self.get_bandwidth_in_previous_seconds()
        if bandwidth == 0:
            return 0
        return round(
            (self._total_resource_size_bytes - self._transferred_resource_size_bytes)
            / bandwidth
        )

    @staticmethod
    def _get_current_timestamp() -> int:
        return round(time.time())

    @property
    def transferred_num_files(self) -> int:
        return self._transferred_num_files

    @property
    def total_num_files(self) -> int:
        return self._total_num_files

    @total_num_files.setter
    def total_num_files(self, value: int) -> None:
        self._total_num_files = value

    @property
    def total_resource_size_bytes(self) -> int:
        return self._total_resource_size_bytes

    @total_resource_size_bytes.setter
    def total_resource_size_bytes(self, value: int) -> None:
        self._total_resource_size_bytes = value

    @property
    def transferred_resource_size_bytes(self) -> int:
        return self._transferred_resource_size_bytes

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return str(self.__dict__)


class DatasetTransferState:
    def __init__(self, total_num_rows: int, name: Optional[str] = None) -> None:
        self._name: Optional[str] = name
        self._total_num_rows: int = total_num_rows
        self._transferred_num_rows: int = 0
        self._timestamp_to_rows_sent: Dict[Any, int] = defaultdict(int)
        self._how_many_previous_seconds = 20

    def increment_num_transferred_rows(self, increment_with: int) -> None:
        self._transferred_num_rows += increment_with
        self._timestamp_to_rows_sent[self._get_current_timestamp()] += increment_with

    def _get_rows_per_sec(self) -> int:
        curr_timestamp = self._get_current_timestamp()
        valid_seconds = 0
        rows_transferred = 0
        # Exclude start and current sec as otherwise result would not account for data that will be sent
        # within current second after the call of this func
        for i in range(1, self._how_many_previous_seconds + 1):
            look_at = curr_timestamp - i
            if look_at >= 0 and look_at in self._timestamp_to_rows_sent:
                valid_seconds += 1
                rows_transferred += self._timestamp_to_rows_sent[look_at]
        if valid_seconds == 0:
            return 0
        return round(rows_transferred / valid_seconds)

    def get_eta_seconds(self) -> int:
        rows_per_sec = self._get_rows_per_sec()
        if rows_per_sec == 0:
            return 0
        return round((self._total_num_rows - self._transferred_num_rows) / rows_per_sec)

    @staticmethod
    def _get_current_timestamp() -> int:
        return round(time.time())

    @property
    def transferred_num_rows(self) -> int:
        return self._transferred_num_rows

    @property
    def total_num_rows(self) -> int:
        return self._total_num_rows

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def __str__(self) -> str:
        return str(self.__dict__)
