import uuid

from layerapi.api.entity.run_pb2 import Run as PBRun
from layerapi.api.entity.task_pb2 import Task as PBTask
from layerapi.api.ids_pb2 import RunId
from layerapi.api.value.run_status_pb2 import RunStatus as PBRunStatus

from layer.contracts.runs import Run, RunStatus, TaskType


def to_run_id(id: uuid.UUID) -> RunId:
    return RunId(value=str(id))


def from_run_id(id: RunId) -> uuid.UUID:
    return uuid.UUID(id.value)


def from_run(run: PBRun) -> Run:
    return Run(
        id=from_run_id(run.id),
        name=run.name,
    )


def to_run_status(val: RunStatus) -> "PBRunStatus.V":
    return RUN_STATUS_TO_PROTO_MAP[val]


def from_run_status(val: "PBRunStatus.V") -> RunStatus:
    return RUN_STATUS_FROM_PROTO_MAP[val]


TASK_TYPE_TO_PROTO_MAP = {
    TaskType.MODEL_TRAIN: PBTask.Type.TYPE_MODEL_TRAIN,
    TaskType.DATASET_BUILD: PBTask.Type.TYPE_DATASET_BUILD,
}

TASK_TYPE_FROM_PROTO_MAP = {v: k for k, v in TASK_TYPE_TO_PROTO_MAP.items()}

RUN_STATUS_TO_PROTO_MAP = {
    RunStatus.RUNNING: PBRunStatus.RUN_STATUS_RUNNING,
    RunStatus.SUCCEEDED: PBRunStatus.RUN_STATUS_SUCCEEDED,
    RunStatus.FAILED: PBRunStatus.RUN_STATUS_FAILED,
}

RUN_STATUS_FROM_PROTO_MAP = {v: k for k, v in RUN_STATUS_TO_PROTO_MAP.items()}
