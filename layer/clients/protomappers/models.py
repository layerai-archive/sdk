import uuid

from layerapi.api.entity.model_train_pb2 import ModelTrain as PBModelTrain
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion as PBModelVersion
from layerapi.api.ids_pb2 import ModelTrainId, ModelVersionId

from layer.contracts.models import ModelTrain, ModelTrainStatus, ModelVersion


MODEL_TRAIN_STATUS_TO_PROTO_MAP = {
    ModelTrainStatus.INVALID: PBModelTrainStatus.TRAIN_STATUS_INVALID,
    ModelTrainStatus.PENDING: PBModelTrainStatus.TRAIN_STATUS_PENDING,
    ModelTrainStatus.INITIALIZING: PBModelTrainStatus.TRAIN_STATUS_INITIALIZING,
    ModelTrainStatus.FETCHING_FEATURES: PBModelTrainStatus.TRAIN_STATUS_FETCHING_FEATURES,
    ModelTrainStatus.IN_PROGRESS: PBModelTrainStatus.TRAIN_STATUS_IN_PROGRESS,
    ModelTrainStatus.SUCCESSFUL: PBModelTrainStatus.TRAIN_STATUS_SUCCESSFUL,
    ModelTrainStatus.FAILED: PBModelTrainStatus.TRAIN_STATUS_FAILED,
    ModelTrainStatus.CANCEL_REQUESTED: PBModelTrainStatus.TRAIN_STATUS_CANCEL_REQUESTED,
    ModelTrainStatus.CANCELED: PBModelTrainStatus.TRAIN_STATUS_CANCELED,
}

MODEL_TRAIN_STATUS_FROM_PROTO_MAP = {
    v: k for k, v in MODEL_TRAIN_STATUS_TO_PROTO_MAP.items()
}


def to_model_train_id(id: uuid.UUID) -> ModelTrainId:
    return ModelTrainId(value=str(id))


def from_model_train_id(id: ModelTrainId) -> uuid.UUID:
    return uuid.UUID(id.value)


def to_model_version_id(id: uuid.UUID) -> ModelVersionId:
    return ModelVersionId(value=str(id))


def from_model_version_id(id: ModelVersionId) -> uuid.UUID:
    return uuid.UUID(id.value)


def from_model_version(model_version: PBModelVersion) -> ModelVersion:
    return ModelVersion(
        id=from_model_version_id(model_version.id),
        name=model_version.name,
    )


def from_model_train(
    model_train: PBModelTrain, model_version: PBModelVersion
) -> ModelTrain:
    return ModelTrain(
        id=from_model_train_id(model_train.id),
        index=model_train.index,
        status=MODEL_TRAIN_STATUS_FROM_PROTO_MAP[model_train.train_status.train_status],
        tag=f"{model_version.name}.{model_train.index}",
    )


def to_model_train_status(status: ModelTrainStatus, info: str) -> PBModelTrainStatus:
    return PBModelTrainStatus(
        train_status=MODEL_TRAIN_STATUS_TO_PROTO_MAP[status], info=info
    )
