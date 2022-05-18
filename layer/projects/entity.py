from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Optional, Type, Union

from yarl import URL

from layer.contracts.datasets import DerivedDataset
from layer.contracts.models import Model
from layer.projects.tracker.dataset_transfer_state import DatasetTransferState
from layer.projects.tracker.resource_transfer_state import ResourceTransferState


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

    def get_url(
        self, base_url: URL, project_name: str, account_name: str, **kwargs: Any
    ) -> URL:
        if self == EntityType.MODEL:
            version, train_index, name = (
                kwargs.get("version"),
                kwargs.get("train_index"),
                kwargs["name"],
            )

            if version and train_index:
                version_and_train_index = f"{version}.{train_index}"
                return (
                    base_url
                    / account_name
                    / project_name
                    / "models"
                    / name
                    % {"v": version_and_train_index}
                )
            else:
                return base_url / account_name / project_name / "models" / name
        if self == EntityType.DERIVED_DATASET:
            version, build_index, name = (
                kwargs.get("version"),
                kwargs.get("build_index"),
                kwargs["name"],
            )

            if version and build_index:
                version_and_build_index = f"{version}.{build_index}"

                return (
                    base_url
                    / account_name
                    / project_name
                    / "datasets"
                    / name
                    % {"v": version_and_build_index}
                )
            else:
                return base_url / account_name / project_name / "datasets" / name
        raise RuntimeError(f"Unsupported entity type: {self}")

    def get_factory(
        self,
    ) -> Union[Type[DerivedDataset], Type[Model]]:
        if self == EntityType.DERIVED_DATASET:
            return DerivedDataset
        elif self == EntityType.MODEL:
            return Model
        else:
            raise RuntimeError(f"Unsupported entity type: {self}")


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
