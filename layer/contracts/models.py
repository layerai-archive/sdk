import uuid
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import pandas as pd
from layerapi.api.ids_pb2 import ModelTrainId
from layerapi.api.value.aws_credentials_pb2 import AwsCredentials
from layerapi.api.value.s3_path_pb2 import S3Path

from layer.exceptions.exceptions import LayerClientException
from layer.flavors.base import ModelFlavor, ModelRuntimeObjects
from layer.types import ModelObject

from .asset import AssetPath, AssetType, BaseAsset


@dataclass(frozen=True)
class TrainStorageConfiguration:
    train_id: ModelTrainId
    s3_path: S3Path
    credentials: AwsCredentials


class Model(BaseAsset):
    """
    Provides access to ML models trained and stored in Layer.

    You can retrieve an instance of this object with :code:`layer.get_model()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Fetches a specific version of this model
        layer.get_model("churn_model:1.2")

    """

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
        version_id: Optional[uuid.UUID] = None,
        description: str = "",
        flavor: Optional[ModelFlavor] = None,
        storage_config: Optional[TrainStorageConfiguration] = None,
        model_runtime_objects: Optional[ModelRuntimeObjects] = None,
    ):
        super().__init__(
            path=asset_path,
            id=id,
            dependencies=dependencies,
        )
        self._version_id = version_id
        self._description = description
        self._flavor = flavor
        self._storage_config = storage_config
        self._model_runtime_objects = model_runtime_objects

    @property
    def asset_type(self) -> AssetType:
        return AssetType.MODEL

    @property
    def version_id(self) -> uuid.UUID:
        if self._version_id is None:
            raise LayerClientException("Model version id is not initialized")
        return self._version_id

    @property
    def flavor(self) -> ModelFlavor:
        if self._flavor is None:
            raise LayerClientException("Model flavor is not initialized")
        return self._flavor

    @property
    def storage_config(self) -> TrainStorageConfiguration:
        if self._storage_config is None:
            raise LayerClientException("Model storage config is not initialized")
        return self._storage_config

    @property
    def model_object(self) -> ModelObject:
        if self._model_runtime_objects is None:
            raise LayerClientException("Model artifact is not yet fetched from storage")
        return self._model_runtime_objects.model_object

    def get_train(self) -> ModelObject:
        """
        Returns the trained and saved model artifact. For example, a scikit-learn or PyTorch model object.

        :return: The trained model artifact.
        """
        return self.model_object

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs prediction on the input dataframe data.
        :return: the predictions as a pd.DataFrame
        """
        if (
            self._model_runtime_objects is None
            or self._model_runtime_objects.prediction_function is None
        ):
            raise Exception("No predict function provided")
        return self._model_runtime_objects.prediction_function(input_df)

    def set_model_runtime_objects(
        self, model_runtime_objects: ModelRuntimeObjects
    ) -> "Model":
        self._model_runtime_objects = model_runtime_objects
        return self
