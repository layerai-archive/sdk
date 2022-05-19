import copy
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

from layerapi.api.ids_pb2 import ModelTrainId
from layerapi.api.value.aws_credentials_pb2 import AwsCredentials
from layerapi.api.value.s3_path_pb2 import S3Path

from layer.exceptions.exceptions import LayerClientException
from layer.flavors.base import ModelFlavor
from layer.types import ModelArtifact

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
        parameters: Optional[Dict[str, Any]] = None,
        model_artifact: Optional[ModelArtifact] = None,
    ):
        super().__init__(
            asset_type=AssetType.MODEL,
            path=asset_path,
            id=id,
            dependencies=dependencies,
        )
        self._version_id = version_id
        self._description = description
        self._flavor = flavor
        self._storage_config = storage_config
        self.parameters = parameters or {}
        self._model_artifact = model_artifact

    def set_parameters(self, parameters: Dict[str, Any]) -> "Model":
        self.parameters = parameters
        return self

    def set_artifact(self, model_artifact: ModelArtifact) -> "Model":
        self._model_artifact = model_artifact
        return self

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
    def artifact(self) -> ModelArtifact:
        if self._model_artifact is None:
            raise LayerClientException("Model artifact is not yet fetched from storage")
        return self._model_artifact

    def get_train(self) -> ModelArtifact:
        """
        Returns the trained and saved model artifact. For example, a scikit-learn or PyTorch model object.

        :return: The trained model artifact.
        """
        return self.artifact

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of the parameters of the model.

        :return: The mapping from a parameter that defines the model to the value of that parameter.

        You could enter this in a Jupyter Notebook:

        .. code-block:: python

            model = layer.get_model("survival_model")
            parameters = model.get_parameters()
            parameters

        .. code-block:: python

            {'test_size': '0.2', 'n_estimators': '100'}

        """
        return self.parameters

    def with_dependencies(self, dependencies: Sequence[BaseAsset]) -> "Model":
        new_model = copy.deepcopy(self)
        new_model._set_dependencies(dependencies)  # pylint: disable=protected-access
        return new_model

    def with_project_name(self, project_name: str) -> "Model":
        new_asset = super().with_project_name(project_name=project_name)
        new_model = copy.deepcopy(self)
        new_model._update_with(new_asset)  # pylint: disable=protected-access
        return new_model

    def drop_dependencies(self) -> "Model":
        return self.with_dependencies(())

    def __str__(self) -> str:
        return f"Model({self.name})"
