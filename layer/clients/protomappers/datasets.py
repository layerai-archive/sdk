import uuid
from typing import Optional

from layerapi.api.entity.dataset_build_pb2 import DatasetBuild as PBDatasetBuild
from layerapi.api.entity.dataset_version_pb2 import DatasetVersion as PBDatasetVersion
from layerapi.api.ids_pb2 import DatasetBuildId, DatasetVersionId

from layer.contracts.datasets import DatasetBuild, DatasetBuildStatus


DATASET_BUILD_STATUS_TO_PROTO_MAP = {
    DatasetBuildStatus.INVALID: PBDatasetBuild.BUILD_STATUS_INVALID,
    DatasetBuildStatus.STARTED: PBDatasetBuild.BUILD_STATUS_STARTED,
    DatasetBuildStatus.FAILED: PBDatasetBuild.BUILD_STATUS_FAILED,
    DatasetBuildStatus.COMPLETED: PBDatasetBuild.BUILD_STATUS_COMPLETED,
}

DATASET_BUILD_STATUS_FROM_PROTO_MAP = {
    v: k for k, v in DATASET_BUILD_STATUS_TO_PROTO_MAP.items()
}


def to_dataset_build_id(id: uuid.UUID) -> DatasetBuildId:
    return DatasetBuildId(value=str(id))


def from_dataset_build_id(id: DatasetBuildId) -> uuid.UUID:
    return uuid.UUID(id.value)


def to_dataset_version_id(id: uuid.UUID) -> DatasetVersionId:
    return DatasetVersionId(value=str(id))


def from_dataset_version_id(id: DatasetVersionId) -> uuid.UUID:
    return uuid.UUID(id.value)


def from_dataset_build(
    dataset_build: PBDatasetBuild, dataset_version: Optional[PBDatasetVersion]
) -> DatasetBuild:
    return DatasetBuild(
        id=from_dataset_build_id(dataset_build.id),
        status=DATASET_BUILD_STATUS_FROM_PROTO_MAP[dataset_build.build_info.status],
        tag=f"{dataset_version.name}.{dataset_build.index}" if dataset_version else "",
    )
