from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Union

from yarl import URL

from .runs import DatasetTransferState, ResourceTransferState


@unique
class EntityStatus(str, Enum):
    PENDING = "pending"
    SAVING = "saving"
    BUILDING = "building"
    TRAINING = "training"
    DONE = "done"
    ERROR = "error"
    ASSERTING = "asserting"
    RESOURCE_UPLOADING = "uploading resources"
    RESULT_UPLOADING = "uploading result"
    ENTITY_DOWNLOADING = "downloading entity"
    ENTITY_FROM_CACHE = "entity from cache"
    ENTITY_LOADED = "entity loaded"

    @property
    def is_running(self) -> bool:
        return self in (
            EntityStatus.SAVING,
            EntityStatus.BUILDING,
            EntityStatus.TRAINING,
            EntityStatus.ASSERTING,
            EntityStatus.RESOURCE_UPLOADING,
            EntityStatus.RESULT_UPLOADING,
            EntityStatus.ENTITY_DOWNLOADING,
            EntityStatus.ENTITY_FROM_CACHE,
        )

    @property
    def is_finished(self) -> bool:
        return self in (
            EntityStatus.DONE,
            EntityStatus.ERROR,
            EntityStatus.ENTITY_LOADED,
        )


@unique
class EntityType(str, Enum):
    RAW_DATASET = "raw_dataset"
    DERIVED_DATASET = "derived_dataset"
    MODEL = "model"


@dataclass
class Entity:
    type: EntityType
    name: str
    status: EntityStatus = EntityStatus.PENDING
    base_url: Optional[URL] = None
    error_reason: str = ""
    resource_transfer_state: Optional[ResourceTransferState] = None
    dataset_transfer_state: Optional[DatasetTransferState] = None
    model_transfer_state: Optional[ResourceTransferState] = None
    entity_download_transfer_state: Optional[
        Union[ResourceTransferState, DatasetTransferState]
    ] = None
    loading_cache_entity: Optional[str] = None
    version: Optional[str] = None
    build_idx: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.status.is_running

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished
