from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, List, Optional, Union

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
    def mark_running(
        self,
        asset_type: AssetType,
        name: str,
        *,
        tag: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_asserting(
        self,
        asset_type: AssetType,
        name: str,
        *,
        assertion: Optional[Assertion] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_failed_assertions(
        self, asset_type: AssetType, name: str, assertions: List[Assertion]
    ) -> None:
        pass

    @abstractmethod
    def mark_asserted(self, asset_type: AssetType, name: str) -> None:
        pass

    @abstractmethod
    def mark_asset_uploading(
        self,
        asset_type: AssetType,
        name: str,
        *,
        dataset_transfer_state: Optional[DatasetTransferState] = None,
        model_transfer_state: Optional[ResourceTransferState] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_done(
        self,
        asset_type: AssetType,
        name: str,
        *,
        warnings: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_failed(
        self,
        asset_type: AssetType,
        name: str,
        *,
        reason: str,
        tag: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def mark_asset_downloading(
        self,
        asset_type: AssetType,
        name: str,
        getting_asset_name: str,
        state: Optional[ResourceTransferState],
        from_cache: bool,
    ) -> None:
        pass

    @abstractmethod
    def mark_asset_downloaded(
        self,
        asset_type: AssetType,
        name: str,
    ) -> None:
        pass
