from layerapi import api

from layer.contracts import ids
from layer.contracts.runs import Run, RunStatus, TaskType


def to_run_id(id: ids.RunId) -> api.RunId:
    return api.RunId(value=str(id))


def from_run_id(id: api.RunId) -> ids.RunId:
    return ids.RunId(id.value)


def from_run(run: api.Run) -> Run:
    return Run(
        id=from_run_id(run.id),
        name=run.name,
    )


def to_run_status(val: RunStatus) -> api.RunStatus:
    return RUN_STATUS_TO_PROTO_MAP[val]


def from_run_status(val: api.RunStatus) -> RunStatus:
    return RUN_STATUS_FROM_PROTO_MAP[val]


TASK_TYPE_TO_PROTO_MAP = {
    TaskType.MODEL_TRAIN: api.TaskType.TYPE_MODEL_TRAIN,
    TaskType.DATASET_BUILD: api.TaskType.TYPE_DATASET_BUILD,
}

TASK_TYPE_FROM_PROTO_MAP = {v: k for k, v in TASK_TYPE_TO_PROTO_MAP.items()}

RUN_STATUS_TO_PROTO_MAP = {
    RunStatus.RUNNING: api.RunStatus.RUN_STATUS_RUNNING,
    RunStatus.SUCCEEDED: api.RunStatus.RUN_STATUS_SUCCEEDED,
    RunStatus.FAILED: api.RunStatus.RUN_STATUS_FAILED,
}

RUN_STATUS_FROM_PROTO_MAP = {v: k for k, v in RUN_STATUS_TO_PROTO_MAP.items()}
