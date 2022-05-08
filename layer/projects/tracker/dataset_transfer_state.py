import time
from collections import defaultdict
from typing import Any, Dict


class DatasetTransferState:
    def __init__(self, total_num_rows: int) -> None:
        self._total_num_rows: int = total_num_rows
        self._transferred_num_rows: int = 0
        self._timestamp_to_rows_sent: Dict[Any, int] = defaultdict(int)
        self._how_many_previous_seconds = 4

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
            if look_at > 0:
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

    def __str__(self) -> str:
        return str(self.__dict__)
