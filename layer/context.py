from types import TracebackType
from typing import Optional

from layer.contracts.assets import AssetType
from layer.contracts.datasets import DatasetBuild
from layer.tracker.ui_progress_tracker import RunProgressTracker
from layer.training.base_train import BaseTrain


class Context:
    """
    Provides access to variables within the pipeline execution.

    This class should not be initialized by end-users.
    """

    def __init__(
        self,
        train: Optional[BaseTrain] = None,
        dataset_build: Optional[DatasetBuild] = None,
        tracker: Optional[RunProgressTracker] = None,
        asset_name: Optional[str] = None,
        asset_type: Optional[AssetType] = None,
    ) -> None:
        self._train: Optional[BaseTrain] = train
        self._dataset_build: Optional[DatasetBuild] = dataset_build
        self._tracker: Optional[RunProgressTracker] = tracker
        self._asset_name = asset_name
        self._asset_type = asset_type

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

    def with_tracker(self, tracker: RunProgressTracker) -> None:
        self._tracker = tracker

    def with_asset_name(self, asset_name: str) -> None:
        self._asset_name = asset_name

    def with_asset_type(self, asset_type: AssetType) -> None:
        self._asset_type = asset_type

    def tracker(self) -> Optional[RunProgressTracker]:
        return self._tracker

    def asset_name(self) -> Optional[str]:
        return self._asset_name

    def asset_type(self) -> AssetType:
        if self._asset_type:
            return self._asset_type
        elif self.train():
            return AssetType.MODEL
        elif self.dataset_build():
            return AssetType.DATASET
        else:
            raise Exception("Unsupported asset type")

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
