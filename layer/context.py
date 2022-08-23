from types import TracebackType
from typing import Optional

from layer.contracts.assets import AssetType
from layer.contracts.datasets import DatasetBuild
from layer.tracker.ui_progress_tracker import RunProgressTracker
from layer.training.base_train import BaseTrain


# We store the active context temporarily so it can be used within
# either a @layer decorated function or soon via `with layer.model() as context:`
_ACTIVE_CONTEXT: Optional["Context"] = None


def set_active_context(context: "Context") -> None:
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context


def reset_active_context() -> None:
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = None


def get_active_context() -> Optional["Context"]:
    """
    Returns the active context object set from the active computation.
    Used in local mode to identify which context to log resources to.

    @return:  active context object
    """
    return _ACTIVE_CONTEXT


class Context:
    """
    Provides access to variables within a given execution, for example during a model train.

    This class should not be initialized by end-users.
    """

    def __init__(
        self,
        asset_name: str,
        asset_type: AssetType,
        tracker: Optional[RunProgressTracker] = None,
        train: Optional[BaseTrain] = None,
        dataset_build: Optional[DatasetBuild] = None,
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

        :return: Represents the current build of the dataset, passed by Layer when building of the dataset starts.
        """
        return self._dataset_build

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
