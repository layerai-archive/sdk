import uuid

from layerapi.api.entity.task_pb2 import Task as PBTask
from layerapi.api.ids_pb2 import RunId

from layer.contracts.runs import TaskType


def to_run_id(id: uuid.UUID) -> RunId:
    return RunId(value=str(id))


def from_run_id(id: RunId) -> uuid.UUID:
    return uuid.UUID(id.value)


TASK_TYPE_TO_PROTO_MAP = {
    TaskType.MODEL_TRAIN: PBTask.Type.TYPE_MODEL_TRAIN,
    TaskType.DATASET_BUILD: PBTask.Type.TYPE_DATASET_BUILD,
}

TASK_TYPE_FROM_PROTO_MAP = {v: k for k, v in TASK_TYPE_TO_PROTO_MAP.items()}
