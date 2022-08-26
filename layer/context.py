from types import TracebackType
from typing import Optional

from yarl import URL

from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.datasets import DatasetBuild
from layer.logged_data.logged_data_destination import LoggedDataDestination
from layer.tracker.ui_progress_tracker import RunProgressTracker
from layer.training.base_train import BaseTrain


# We store the active context temporarily so it can be used within
# either a @layer decorated function or soon via `with layer.model() as context:`
_ACTIVE_CONTEXT: Optional["Context"] = None


def _set_active_context(context: "Context") -> None:
    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = context


def _reset_active_context() -> None:
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
        url: URL,
        asset_path: AssetPath,
        logged_data_destination: Optional[LoggedDataDestination] = None,
        tracker: Optional[RunProgressTracker] = None,
        train: Optional[BaseTrain] = None,
        dataset_build: Optional[DatasetBuild] = None,
    ) -> None:
        if train is not None and dataset_build is not None:
            raise Exception(
                "Context cannot hold model train and dataset build at the same time"
            )
        if train is not None and asset_path.asset_type is not AssetType.MODEL:
            raise Exception("Wrong asset type for model train context")
        if dataset_build is not None and asset_path.asset_type is not AssetType.DATASET:
            raise Exception("Wrong asset type for dataset build context")
        self._url = url
        self._project_full_name = asset_path.project_full_name()
        self._asset_name = asset_path.asset_name
        self._asset_type = asset_path.asset_type
        self._train: Optional[BaseTrain] = train
        self._dataset_build: Optional[DatasetBuild] = dataset_build
        self._logged_data_destination: Optional[
            LoggedDataDestination
        ] = logged_data_destination
        self._tracker: Optional[RunProgressTracker] = tracker

    def url(self) -> URL:
        """
        Returns the URL of the current active context.

        For example, during model training, it returns the current
        model train URL.
        """
        p = AssetPath(
            account_name=self._project_full_name.account_name,
            project_name=self._project_full_name.project_name,
            asset_type=self.asset_type(),
            asset_name=self.asset_name(),
        )
        if self.train() is not None:
            train = self.train()
            assert train is not None
            p = p.with_version_and_build(
                train.get_version(), int(train.get_train_index())
            )

        return p.url(self._url)

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

    def asset_name(self) -> str:
        return self._asset_name

    def logged_data_destination(self) -> Optional[LoggedDataDestination]:
        return self._logged_data_destination

    def asset_type(self) -> AssetType:
        return self._asset_type

    def close(self) -> None:
        _reset_active_context()

    def __enter__(self) -> "Context":
        _set_active_context(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
