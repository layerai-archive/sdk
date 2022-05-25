import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import Any, Callable, List, Tuple

from layerapi.api.entity.user_log_line_pb2 import UserLogLine as PBUserLogLine
from layerapi.api.ids_pb2 import RunId

from layer.clients.layer import LayerClient


POLLING_INTERVAL_SEC = 3
LOGS_BUFFER_INTERVAL = 20  # Minimum amount of time to have as a buffer for logs


@unique
class EntityType(Enum):
    MODEL_TRAIN = 2
    DATASET_BUILD = 4


@dataclass(frozen=True)
class UserLogLine:
    entity_name: str = field(default_factory=str)
    type: str = field(default_factory=str)
    host_name: str = field(default_factory=str)
    time: datetime = datetime.utcfromtimestamp(0)
    log: str = field(default_factory=str)


def __convert_log_line(line: PBUserLogLine) -> UserLogLine:
    return UserLogLine(
        type=__convert_entity(line.type).name,
        entity_name=line.entity_name,
        host_name=line.host_name,
        time=line.time.ToDatetime(),
        log=line.log,
    )


def __convert_entity(entity: PBUserLogLine.TaskType) -> EntityType:
    if entity == PBUserLogLine.TASK_TYPE_MODEL_TRAIN:
        return EntityType.MODEL_TRAIN
    elif entity == PBUserLogLine.TASK_TYPE_DATASET_BUILD:
        return EntityType.DATASET_BUILD
    else:
        raise Exception(f"Unable to convert {entity}")


def __get_lines(
    pipeline_run_id: uuid.UUID, continuation_token: str, client: LayerClient
) -> Tuple[List[UserLogLine], str]:
    proto_lines, continuation_token = client.user_logs.get_pipeline_run_logs(
        pipeline_run_id, continuation_token
    )
    converted_lines = [__convert_log_line(line) for line in proto_lines]
    return converted_lines, continuation_token


def __format_line(line: UserLogLine) -> str:
    time_str = line.time.strftime("%H:%M:%S")
    return f"\033[0;33m{time_str} \033[1;32m{line.entity_name}\033[0m: {line.log}"


def show_pipeline_run_logs(
    client: LayerClient,
    pipeline_run_id: str,
    follow: bool,
    polling_interval_sec: float = POLLING_INTERVAL_SEC,
    printer: Callable[[str], Any] = print,
    evaluate_until_callback: Callable[[], bool] = lambda: True,
) -> None:
    curr_token = ""
    while evaluate_until_callback():
        lines, next_token = __get_lines(uuid.UUID(pipeline_run_id), curr_token, client)
        for line in lines:
            formatted = __format_line(line)
            printer(formatted)
        prev_token = curr_token
        curr_token = next_token
        if prev_token == curr_token and follow is False:
            break
        time.sleep(polling_interval_sec)


def get_pipeline_run_logs(
    client: LayerClient,
    pipeline_run_id: RunId,
    polling_interval_sec: float = POLLING_INTERVAL_SEC,
) -> List[UserLogLine]:
    lines: List[UserLogLine] = []
    curr_token = ""
    while True:
        curr_lines, next_token = __get_lines(
            uuid.UUID(pipeline_run_id.value), curr_token, client
        )
        lines.extend(curr_lines)
        prev_token = curr_token
        curr_token = next_token
        if prev_token == curr_token:
            break
        time.sleep(polling_interval_sec)
    return lines
