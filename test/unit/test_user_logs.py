from datetime import datetime

from google.protobuf.timestamp_pb2 import Timestamp
from layerapi.api.entity.user_log_line_pb2 import UserLogLine as PBUserLogLine

from layer.user_logs import EntityType, __convert_log_line


def test_convert_pb_log_line() -> None:
    now = datetime.now()
    timestamp = Timestamp()
    timestamp.FromDatetime(dt=now)
    pb_user_log_line = PBUserLogLine(
        entity_name="test",
        host_name="127.0.0.1",
        time=timestamp,
        log="test-log-line",
        type=PBUserLogLine.TASK_TYPE_DATASET_BUILD,
    )

    line = __convert_log_line(pb_user_log_line)
    assert line.entity_name == "test"
    assert line.host_name == "127.0.0.1"
    assert line.time == now
    assert line.log == "test-log-line"
    assert line.type == EntityType.DATASET_BUILD.name
