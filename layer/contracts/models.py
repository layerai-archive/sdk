import copy
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Sequence, Tuple, Union

from layerapi.api.ids_pb2 import ModelTrainId
from layerapi.api.value.aws_credentials_pb2 import AwsCredentials
from layerapi.api.value.s3_path_pb2 import S3Path

from .asset import AssetPath, AssetType, BaseAsset


TrainedModelObject = NewType("TrainedModelObject", object)


@dataclass(frozen=True)
class TrainStorageConfiguration:
    train_id: ModelTrainId
    s3_path: S3Path
    credentials: AwsCredentials


@dataclass(frozen=True)
class Parameter:
    name: str
    value: str


class ParameterType(Enum):
    INT = 1
    FLOAT = 2
    STRING = 3


@dataclass(frozen=True)
class ParameterValue:
    string_value: Optional[str] = None
    float_value: Optional[float] = None
    int_value: Optional[int] = None

    def with_string(self, string: str) -> "ParameterValue":
        return replace(self, string_value=string)

    def with_float(self, float: float) -> "ParameterValue":
        return replace(self, float_value=float)

    def with_int(self, int: int) -> "ParameterValue":
        return replace(self, int_value=int)


@dataclass(frozen=True)
class TypedParameter:
    name: str
    value: ParameterValue
    type: ParameterType


@dataclass(frozen=True)
class ParameterRange:
    name: str
    min: ParameterValue
    max: ParameterValue
    type: ParameterType


@dataclass(frozen=True)
class ParameterCategoricalRange:
    name: str
    values: List[ParameterValue]
    type: ParameterType


@dataclass(frozen=True)
class ParameterStepRange:
    name: str
    min: ParameterValue
    max: ParameterValue
    step: ParameterValue
    type: ParameterType


@dataclass(frozen=True)
class ManualSearch:
    parameters: List[List[TypedParameter]]


@dataclass(frozen=True)
class RandomSearch:
    max_jobs: int
    parameters: List[ParameterRange]
    parameters_categorical: List[ParameterCategoricalRange]


@dataclass(frozen=True)
class GridSearch:
    parameters: List[ParameterStepRange]


@dataclass(frozen=True)
class BayesianSearch:
    max_jobs: int
    parameters: List[ParameterRange]


@dataclass(frozen=True)
class HyperparameterTuning:
    strategy: str
    max_parallel_jobs: Optional[int]
    maximize: Optional[str]
    minimize: Optional[str]
    early_stop: Optional[bool]
    fixed_parameters: Dict[str, float]
    manual_search: Optional[ManualSearch]
    random_search: Optional[RandomSearch]
    grid_search: Optional[GridSearch]
    bayesian_search: Optional[BayesianSearch]


@dataclass(frozen=True)
class Train:
    name: str = ""
    description: str = ""
    entrypoint: str = ""
    environment: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    hyperparameter_tuning: Optional[HyperparameterTuning] = None
    fabric: str = ""


class Model(BaseAsset):
    """
    Provides access to ML models trained and stored in Layer.

    You can retrieve an instance of this object with :code:`layer.get_model()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Fetches a specific version of this model
        layer.get_model("churn_model:1.2")

    """

    local_path: Path
    description: str
    training: Train
    trained_model_object: Any
    training_files_digest: str
    parameters: Dict[str, Any]
    language_version: Tuple[int, int, int]

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
        description: str = "",
        local_path: Optional[Path] = None,
        training: Optional[Train] = None,
        trained_model_object: Any = None,
        training_files_digest: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        language_version: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            path=asset_path,
            asset_type=AssetType.MODEL,
            id=id,
            dependencies=dependencies,
        )
        if parameters is None:
            parameters = {}
        self.description = description
        if local_path is None:
            local_path = Path()
        self.local_path = local_path
        if training is None:
            training = Train()
        self.training = training
        self.trained_model_object = trained_model_object
        self.training_files_digest = training_files_digest
        self.parameters = parameters

    def get_train(self) -> Any:
        """
        Returns the trained and saved model object. For example, a scikit-learn or PyTorch model object.

        :return: The trained model object.

        """
        return self.trained_model_object

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

    def with_language_version(self, language_version: Tuple[int, int, int]) -> "Model":
        new_model = copy.deepcopy(self)
        new_model.language_version = language_version
        return new_model

    def drop_dependencies(self) -> "Model":
        return self.with_dependencies(())

    def __str__(self) -> str:
        return f"Model({self.name})"
