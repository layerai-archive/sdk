import os
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

import pandas as pd
from yarl import URL

from layer.clients.layer import LayerClient
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.datasets import DatasetBuild
from layer.contracts.tracker import DatasetTransferState, ResourceTransferState
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
        client: Optional[LayerClient] = None,
    ) -> None:
        if train is not None and dataset_build is not None:
            raise Exception(
                "Context cannot hold model train and dataset build at the same time"
            )
        if train is not None and asset_path.asset_type is not AssetType.MODEL:
            raise Exception("Wrong asset type for model train context")
        if dataset_build is not None and asset_path.asset_type is not AssetType.DATASET:
            raise Exception("Wrong asset type for dataset build context")
        if dataset_build is not None and client is None:
            raise Exception("Context with dataset build also needs a layer client")
        self._url = url
        self._client = client
        self._project_full_name = asset_path.project_full_name()
        self._asset_name = asset_path.asset_name
        self._asset_type = asset_path.asset_type
        self._train: Optional[BaseTrain] = train
        self._dataset_build: Optional[DatasetBuild] = dataset_build
        self._logged_data_destination: Optional[
            LoggedDataDestination
        ] = logged_data_destination
        self._tracker: Optional[RunProgressTracker] = tracker
        self._initial_cwd: Optional[Path] = None

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

    def save_model(self, model: Any) -> None:
        train = self._train
        if not train:
            raise RuntimeError(
                "Saving model only allowed inside when context is for model training"
            )
        model_name = self.asset_name()
        tracker = self._tracker
        if tracker:
            tracker.mark_model_saving(model_name)
        transfer_state = ResourceTransferState()
        train.save_model(model, transfer_state=transfer_state)
        if tracker:
            tracker.mark_model_saving_result(model_name, transfer_state)

    def dataset_build(self) -> Optional[DatasetBuild]:
        """
        Retrieves the active Layer dataset build object.

        :return: Represents the current build of the dataset, passed by Layer when building of the dataset starts.
        """
        return self._dataset_build

    def save_dataset(self, ds: pd.DataFrame) -> None:
        build = self._dataset_build
        if not build:
            raise RuntimeError(
                "Saving dataset only allowed inside when context is for dataset building"
            )
        dataset_name = self.asset_name()
        assert self._client is not None
        transfer_state = DatasetTransferState(len(ds))
        if self._tracker:
            self._tracker.mark_dataset_saving_result(dataset_name, transfer_state)
        # this call would store the resulting dataset, extract the schema and complete the build from remote
        self._client.data_catalog.store_dataset(
            data=ds,
            build_id=build.id,
            progress_callback=transfer_state.increment_num_transferred_rows,
        )

    def tracker(self) -> Optional[RunProgressTracker]:
        return self._tracker

    def asset_name(self) -> str:
        return self._asset_name

    def logged_data_destination(self) -> Optional[LoggedDataDestination]:
        return self._logged_data_destination

    def asset_type(self) -> AssetType:
        return self._asset_type

    def close(self) -> None:
        assert self._initial_cwd
        os.chdir(
            self._initial_cwd
        )  # Important for local execution to have no such side effect
        _reset_active_context()

    def get_working_directory(self) -> Path:
        assert self._initial_cwd
        return Path(self._initial_cwd)

    def __enter__(self) -> "Context":
        self._initial_cwd = Path(os.getcwd())
        _set_active_context(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()
