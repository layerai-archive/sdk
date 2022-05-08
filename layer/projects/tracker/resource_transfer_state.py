import time
from collections import defaultdict
from typing import Any, Dict


class ResourceTransferState:
    def __init__(self) -> None:
        self._total_num_files: int = 0
        self._transferred_num_files: int = 0
        self._total_resource_size_bytes: int = 0
        self._transferred_resource_size_bytes: int = 0
        self._timestamp_to_bytes_sent: Dict[Any, int] = defaultdict(int)
        self._how_many_previous_seconds = 2

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
            if look_at > 0:
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

    def __str__(self) -> str:
        return str(self.__dict__)
