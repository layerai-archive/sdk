import os
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, Set

import pandas as pd
from yarl import URL

from layer.clients.layer import LayerClient
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.datasets import DatasetBuild
from layer.contracts.models import ModelTrain
from layer.contracts.tracker import DatasetTransferState, ResourceTransferState
from layer.logged_data.logged_data_destination import LoggedDataDestination
from layer.tracker.ui_progress_tracker import RunProgressTracker


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
        dataset_build: Optional[DatasetBuild] = None,
        model_train: Optional[ModelTrain] = None,
        client: Optional[LayerClient] = None,
    ) -> None:
        if model_train is not None and dataset_build is not None:
            raise Exception(
                "Context cannot hold model train and dataset build at the same time"
            )
        if model_train is not None and asset_path.asset_type is not AssetType.MODEL:
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
        self._logged_data_destination = logged_data_destination
        self._tracker = tracker
        self._model_train = model_train
        self._dataset_build = dataset_build
        self._initial_cwd: Optional[Path] = None

    def _label_asset_with(self, label_names: Set[str]) -> None:
        """
        Temporary until we introduce global runs
        """
        assert self._client is not None
        self._client.label_service_client.add_labels_to(
            label_names,
            model_train_id=self._model_train.id
            if self._model_train is not None
            else None,
            dataset_build_id=self._dataset_build.id
            if self._dataset_build is not None
            else None,
        )

    def asset_path(self) -> AssetPath:
        """
        Returns the full path for the Layer asset.
        """
        path = AssetPath(
            account_name=self._project_full_name.account_name,
            project_name=self._project_full_name.project_name,
            asset_type=self.asset_type(),
            asset_name=self.asset_name(),
        )
        model_train = self.model_train()
        if model_train is not None:
            path = path.with_tag(model_train.tag)
        return path

    def url(self) -> URL:
        """
        Returns the URL of the current active context.

        For example, during model training, it returns the current
        model train URL.
        """
        return self.asset_path().url(self._url)

    def model_train(self) -> Optional[ModelTrain]:
        """
        Retrieves the active Layer model train object.

        :return: Represents the current train of the model, passed by Layer when training of the model starts.
        """
        return self._model_train

    def save_model(self, model: Any) -> None:
        assert self._client is not None
        model_train = self._model_train
        if not model_train:
            raise RuntimeError(
                "Saving model only allowed inside when context is for model training"
            )
        model_name = self.asset_name()
        if self._tracker:
            self._tracker.mark_asset_uploading(AssetType.MODEL, model_name)

        transfer_state = ResourceTransferState()

        self._client.model_catalog.save_model_object(
            model,
            model_train.id,
            transfer_state=transfer_state,
        )

        if self._tracker:
            self._tracker.mark_asset_uploading(
                AssetType.MODEL, model_name, model_transfer_state=transfer_state
            )

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
            self._tracker.mark_asset_uploading(
                AssetType.DATASET, dataset_name, dataset_transfer_state=transfer_state
            )
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
