import uuid
from contextlib import contextmanager
from typing import Iterator, List, Optional, Union

from layer.api.ids_pb2 import RunId
from layer.assertion_utils import Assertion
from layer.exceptions.exceptions import ProjectBaseException, ProjectRunnerError
from layer.projects.tracker.dataset_transfer_state import DatasetTransferState
from layer.projects.tracker.resource_transfer_state import ResourceTransferState


class ProjectProgressTracker:
    @contextmanager
    def track(self) -> Iterator["ProjectProgressTracker"]:
        yield self

    def mark_error_messages(
        self, exc: Union[ProjectBaseException, ProjectRunnerError]
    ) -> None:
        pass

    def mark_start_running(self, run_id: RunId) -> None:
        pass

    def mark_raw_dataset_saved(self, name: str) -> None:
        pass

    def mark_raw_dataset_save_failed(self, name: str, reason: str) -> None:
        pass

    def mark_derived_dataset_saved(self, name: str, *, id_: uuid.UUID) -> None:
        pass

    def mark_derived_dataset_building(self, name: str) -> None:
        pass

    def mark_derived_dataset_failed(self, name: str, reason: str) -> None:
        pass

    def mark_derived_dataset_built(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        build_index: Optional[str] = None,
    ) -> None:
        pass

    def mark_model_saving(self, name: str) -> None:
        pass

    def mark_model_saved(self, name: str) -> None:
        pass

    def mark_model_training(self, name: str) -> None:
        pass

    def mark_model_trained(
        self,
        name: str,
        *,
        version: Optional[str] = None,
        train_index: Optional[str] = None,
    ) -> None:
        pass

    def mark_model_train_failed(self, name: str, reason: str) -> None:
        pass

    def update_derived_dataset_saving_progress(
        self, name: str, cur_step: int, total_steps: int
    ) -> None:
        pass

    def mark_model_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        pass

    def mark_model_resources_uploaded(self, name: str) -> None:
        pass

    def mark_dataset_resources_uploading(
        self, name: str, state: ResourceTransferState
    ) -> None:
        pass

    def mark_dataset_resources_uploaded(self, name: str) -> None:
        pass

    def mark_model_running_assertions(self, name: str) -> None:
        pass

    def mark_model_running_assertion(self, name: str, assertion: Assertion) -> None:
        pass

    def mark_model_completed_assertions(self, name: str) -> None:
        pass

    def mark_model_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        pass

    def mark_dataset_running_assertions(self, name: str) -> None:
        pass

    def mark_dataset_running_assertion(self, name: str, assertion: Assertion) -> None:
        pass

    def mark_dataset_completed_assertions(self, name: str) -> None:
        pass

    def mark_dataset_failed_assertions(
        self, name: str, assertions: List[Assertion]
    ) -> None:
        pass

    def mark_dataset_saving_result(
        self, name: str, state: DatasetTransferState
    ) -> None:
        pass

    def mark_model_saving_result(self, name: str, state: ResourceTransferState) -> None:
        pass
