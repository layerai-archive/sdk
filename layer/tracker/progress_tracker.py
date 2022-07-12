import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, List, Optional, Union

from layerapi.api.ids_pb2 import RunId

from layer.contracts.assertions import Assertion
from layer.contracts.asset import AssetType
from layer.contracts.tracker import DatasetTransferState, ResourceTransferState
from layer.exceptions.exceptions import ProjectBaseException, ProjectRunnerError


class RunProgressTracker(ABC):
    @abstractmethod
    @contextmanager
    def track(self) -> Iterator["RunProgressTracker"]:
        pass

    @abstractmethod
    def add_asset(self, asset_type: AssetType, asset_name: str) -> None:
        pass

    @abstractmethod
    def mark_error_messages(
        self, exc: Union[ProjectBaseException, ProjectRunnerError]
    ) -> None:
        pass

    @abstractmethod
    def mark_start_running(self, run_id: RunId) -> None:
        pass

    @abstractmethod
    def mark_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        pass

    @abstractmethod
    def mark_dataset_building(
        self, name: str, version: Optional[str] = None, build_idx: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_failed(self, name: str, reason: str) -> None:
        pass

    @abstractmethod
    def mark_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_model_saving(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_model_saved(
        self,
        name: str,
        version: Optional[str] = None,
        train_index: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_model_training(
        self, name: str, version: Optional[str] = None, train_idx: Optional[str] = None
    ) -> None:
        pass

    @abstractmethod
    def mark_model_trained(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        train_index: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_model_train_failed(
        self, name: str, reason: str
    ) -> None:  # TODO(volkan) check that reason is always given
        pass

    @abstractmethod
    def update_dataset_saving_progress(
        self, name: str, cur_step: int, total_steps: int
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_resources_uploaded(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_model_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        pass

    @abstractmethod
    def mark_model_resources_uploaded(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_model_running_assertions(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_model_running_assertion(self, name: str, assertion: Assertion) -> None:
        pass

    @abstractmethod
    def mark_model_completed_assertions(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_model_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_running_assertions(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_dataset_running_assertion(self, name: str, assertion: Assertion) -> None:
        pass

    @abstractmethod
    def mark_dataset_completed_assertions(self, name: str) -> None:
        pass

    @abstractmethod
    def mark_dataset_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_saving_result(
        self, name: str, state: DatasetTransferState
    ) -> None:
        pass

    @abstractmethod
    def mark_model_saving_result(self, name: str, state: ResourceTransferState) -> None:
        pass

    @abstractmethod
    def mark_model_getting_model(
        self,
        name: str,
        getting_asset_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        pass

    @abstractmethod
    def mark_model_getting_dataset(
        self, name: str, getting_asset_name: str, from_cache: bool
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_getting_model(
        self,
        name: str,
        getting_asset_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_getting_dataset(
        self, name: str, getting_asset_name: str, from_cache: bool
    ) -> None:
        pass

    @abstractmethod
    def mark_model_loaded(
        self,
        name: str,
    ) -> None:
        pass

    @abstractmethod
    def mark_dataset_loaded(
        self,
        name: str,
    ) -> None:
        pass
