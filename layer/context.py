from types import TracebackType
from typing import Optional

from layer.contracts.datasets import DatasetBuild
from layer.contracts.entities import EntityType
from layer.tracker.project_progress_tracker import ProjectProgressTracker
from layer.training.base_train import BaseTrain


class Context:
    """
    Provides access to variables within the pipeline execution.

    An instance of this class is automatically passed in as an argument to the user-defined functions.
    It must always be listed as the first argument in the user-defined functions.

    This class should not be initialized by end-users.
    """

    def __init__(
        self,
        train: Optional[BaseTrain] = None,
        dataset_build: Optional[DatasetBuild] = None,
        tracker: Optional[ProjectProgressTracker] = None,
        entity_name: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
    ) -> None:
        self._train: Optional[BaseTrain] = train
        self._dataset_build: Optional[DatasetBuild] = dataset_build
        self._tracker: Optional[ProjectProgressTracker] = tracker
        self._entity_name = entity_name
        self._entity_type = entity_type

    def train(self) -> Optional[BaseTrain]:
        """
        Retrieves the active Layer train object.

        :return: Represents the current train of the model, passed by Layer when the training of the model starts.

        .. code-block:: python

            # Get train object from Layer Context.
            train = context.train()
        """
        return self._train

    def dataset_build(self) -> Optional[DatasetBuild]:
        """
        Retrieves the active Layer dataset build object.

        :return: Represents the current dataset build of the dataset, passed by Layer when building of the dataset starts.
        """
        return self._dataset_build

    def with_train(self, train: Optional[BaseTrain]) -> None:
        self._train = train

    def with_dataset_build(self, dataset_build: Optional[DatasetBuild]) -> None:
        self._dataset_build = dataset_build

    def with_tracker(self, tracker: ProjectProgressTracker) -> None:
        self._tracker = tracker

    def with_entity_name(self, entity_name: str) -> None:
        self._entity_name = entity_name

    def with_entity_type(self, entity_type: EntityType) -> None:
        self._entity_type = entity_type

    def tracker(self) -> Optional[ProjectProgressTracker]:
        return self._tracker

    def entity_name(self) -> Optional[str]:
        return self._entity_name

    def entity_type(self) -> EntityType:
        if self._entity_type:
            return self._entity_type
        elif self.train():
            return EntityType.MODEL
        elif self.dataset_build():
            return EntityType.DERIVED_DATASET
        else:
            raise Exception("Unsupported entity type")

    def close(self) -> None:
        pass

    def __enter__(self) -> "Context":
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
