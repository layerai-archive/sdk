import abc
import copy
import enum
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas

from layer.api.ids_pb2 import ModelTrainId
from layer.api.value.aws_credentials_pb2 import AwsCredentials
from layer.api.value.s3_path_pb2 import S3Path
from layer.projects.asset import AssetPath, AssetType, BaseAsset


def _create_empty_data_frame() -> "pandas.DataFrame":
    return pandas.DataFrame()


class BaseDataset(BaseAsset):
    description: str

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        description: str = "",
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
    ):
        super().__init__(
            path=asset_path,
            asset_type=AssetType.DATASET,
            id=id,
            dependencies=dependencies,
        )
        self.description = description

    def with_project_name(self: "BaseDataset", project_name: str) -> "BaseDataset":
        new_asset = super().with_project_name(project_name=project_name)
        return BaseDataset(
            new_asset.path,
            self.description,
            self.id,
            self.dependencies,
        )


@enum.unique
class DatasetBuildStatus(enum.IntEnum):
    INVALID = 0
    STARTED = 1
    COMPLETED = 2
    FAILED = 3


@dataclass(frozen=True)
class DatasetBuild:
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    status: DatasetBuildStatus = DatasetBuildStatus.INVALID
    info: str = ""
    index: str = ""


class Dataset(BaseDataset, metaclass=abc.ABCMeta):
    """
    Provides access to datasets defined in Layer.

    You can retrieve an instance of this object with :code:`layer.get_dataset()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Fetches the `titanic` dataset
        Dataset("titanic")

    """

    schema: str
    uri: str
    build: DatasetBuild
    __pandas_df_factory: Optional[
        Callable[[], "pandas.DataFrame"]
    ]  # https://stackoverflow.com/questions/51811024/mypy-type-checking-on-callable-thinks-that-member-variable-is-a-method
    version: str

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        description: str = "",
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
        schema: str = "{}",
        uri: str = "",
        fabric: str = "",
        build: Optional[DatasetBuild] = None,
        _pandas_df_factory: Optional[Callable[[], "pandas.DataFrame"]] = None,
        version: Optional[str] = None,
    ):
        super().__init__(
            asset_path=asset_path,
            id=id,
            dependencies=dependencies,
            description=description,
        )
        self.schema = schema
        self.uri = uri
        if build is None:
            build = DatasetBuild()
        self.build = build
        if _pandas_df_factory is None:
            _pandas_df_factory = _create_empty_data_frame
        self.__pandas_df_factory = _pandas_df_factory
        if version is None:
            version = ""
        self.version = version
        self.fabric = fabric

    def with_id(self, id: uuid.UUID) -> "Dataset":
        new_ds = copy.deepcopy(self)
        new_ds._set_id(id)
        return new_ds

    def _pandas_df_factory(self) -> "pandas.DataFrame":
        assert self.__pandas_df_factory
        return self.__pandas_df_factory()

    @property
    def is_build_completed(self) -> bool:
        return self.build.status == DatasetBuildStatus.COMPLETED

    @property
    def build_info(self) -> str:
        return self.build.info

    def to_pandas(self) -> "pandas.DataFrame":
        """
        Fetches the dataset as a Pandas dataframe.

        :return: A Pandas dataframe containing your dataset.
        """
        return self._pandas_df_factory()

    def to_pytorch(
        self,
        transformer: Callable[[Any], Any],
        tensors: Optional[List[str]] = None,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Any = None,
        batch_sampler: Any = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[Any], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        generator: Any = None,
    ) -> Any:
        """
        Fetches the dataset as a Pytorch DataLoader.
        :param transformer: Function to apply the transformations to the data
        :param tensors: List of columns to fetch
        :param batch_size: how many samples per batch to load (default: 1).
        :param shuffle: set to True to have the data reshuffled at every epoch (default: False).
        :param sampler: defines the strategy to draw samples from the dataset. Can be any Iterable with __len__ implemented. If specified, shuffle must not be specified.
        :param batch_sampler: like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
        :param num_workers: how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
        :param collate_fn: merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them. If your data elements are a custom type, or your collate_fn returns a batch that is a custom type, see the example below.
        :param drop_last: set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
        :param timeout:  if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: 0)
        :param worker_init_fn: If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)
        :param prefetch_factor: Number of samples loaded in advance by each worker. 2 means there will be a total of 2 * num_workers samples prefetched across all workers. (default: 2)
        :param persistent_workers: If True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. (default: False)
        :param generator:  If not None, this RNG will be used by RandomSampler to generate random indexes and multiprocessing to generate base_seed for workers. (default: None)
        :return: torch.utils.data.DataLoader
        """

        # Check if `torch` is installed
        try:
            import torch  # noqa: F401
        except ImportError:
            raise Exception(
                "PyTorch needs to be installed to run `to_pytorch()`. Try `pip install torch`"
            )

        class PytorchDataset(torch.utils.data.Dataset[Any]):
            # TODO: Streaming data fetching for faster data access

            def __init__(self, df: pandas.DataFrame, transformer: Callable[[Any], Any]):
                self.df = df
                self.transformer = transformer

            def __getitem__(self, key: Union[slice, int]) -> Any:
                if isinstance(key, slice):
                    # get the start, stop, and step from the slice
                    return [self[ii] for ii in range(*key.indices(len(self)))]
                elif isinstance(key, int):
                    # handle negative indices
                    if key < 0:
                        key += len(self)
                    if key < 0 or key >= len(self):
                        raise IndexError("The index (%d) is out of range." % key)
                    # get the data from direct index
                    return self.transformer(self.df.iloc[key])
                else:
                    raise TypeError("Invalid argument type.")

            def __len__(self) -> int:
                return self.df.shape[0]

        df = self.to_pandas()
        if tensors:
            df = df[tensors]
        dataset = PytorchDataset(df, transformer)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def __str__(self) -> str:
        return f"Dataset({self.name})"


class RawDataset(Dataset):
    metadata: Mapping[str, str]

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        description: str = "",
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
        schema: str = "{}",
        uri: str = "",
        build: Optional[DatasetBuild] = None,
        _pandas_df_factory: Callable[[], "pandas.DataFrame"] = _create_empty_data_frame,
        metadata: Optional[Mapping[str, str]] = None,
        version: Optional[str] = None,
    ):
        super().__init__(
            asset_path=asset_path,
            id=id,
            dependencies=dependencies,
            description=description,
            schema=schema,
            uri=uri,
            build=build,
            _pandas_df_factory=_pandas_df_factory,
            version=version,
        )
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def with_table_name(self, name: str) -> "RawDataset":
        new_ds = copy.deepcopy(self)
        new_ds.metadata = {**self.metadata, **{"table": name}}
        return new_ds

    def with_metadata(self, metadata: Mapping[str, str]) -> "RawDataset":
        new_ds = copy.deepcopy(self)
        new_ds.metadata = metadata
        return new_ds

    def with_project_name(self, project_name: str) -> "RawDataset":
        new_asset_path = self._path.with_project_name(project_name=project_name)
        new_ds = copy.deepcopy(self)
        new_ds._path = new_asset_path
        return new_ds


class DerivedDataset(Dataset):
    """
    Provides access to derived datasets defined in Layer.

    You can retrieve an instance of this object with :code:`layer.get_dataset()`.

    This class should not be initialized by end-users.

    .. code-block:: python

        # Fetches the `titanic` derived dataset
        DerivedDataset("titanic")

    """

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        description: str = "",
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
        schema: str = "{}",
        uri: str = "",
        build: Optional[DatasetBuild] = None,
        _pandas_df_factory: Callable[[], "pandas.DataFrame"] = _create_empty_data_frame,
    ):
        super().__init__(
            asset_path=asset_path,
            id=id,
            dependencies=dependencies,
            description=description,
            schema=schema,
            uri=uri,
            build=build,
            _pandas_df_factory=_pandas_df_factory,
        )

    def with_dependencies(self, dependencies: Sequence[BaseAsset]) -> "DerivedDataset":
        new_ds = copy.deepcopy(self)
        new_ds._set_dependencies(dependencies)
        return new_ds

    def drop_dependencies(self) -> "DerivedDataset":
        return self.with_dependencies(())

    def with_project_name(self, project_name: str) -> "DerivedDataset":
        new_asset_path = self._path.with_project_name(project_name=project_name)
        new_ds = copy.deepcopy(self)
        new_ds._path = new_asset_path
        return new_ds


class PythonDataset(DerivedDataset):
    entrypoint: str
    entrypoint_path: Path
    entrypoint_content: str
    environment: str
    environment_path: Path
    language_version: Tuple[int, int, int]

    def __init__(
        self,
        asset_path: Union[str, AssetPath],
        description: str = "",
        id: Optional[uuid.UUID] = None,
        dependencies: Optional[Sequence[BaseAsset]] = None,
        schema: str = "{}",
        uri: str = "",
        build: Optional[DatasetBuild] = None,
        _pandas_df_factory: Callable[[], "pandas.DataFrame"] = _create_empty_data_frame,
        fabric: str = "",
        entrypoint: str = "",
        entrypoint_path: Optional[Path] = None,
        entrypoint_content: str = "",
        environment: str = "",
        environment_path: Optional[Path] = None,
        language_version: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(
            asset_path=asset_path,
            id=id,
            dependencies=dependencies,
            description=description,
            schema=schema,
            uri=uri,
            build=build,
            _pandas_df_factory=_pandas_df_factory,
        )
        self.fabric = fabric
        self.entrypoint = entrypoint
        if entrypoint_path is None:
            entrypoint_path = Path()
        self.entrypoint_path = entrypoint_path
        self.entrypoint_content = entrypoint_content
        self.environment = environment
        if environment_path is None:
            environment_path = Path()
        self.environment_path = environment_path
        self.language_version = language_version

    def with_language_version(
        self, language_version: Tuple[int, int, int]
    ) -> "PythonDataset":
        new_ds = copy.deepcopy(self)
        new_ds.language_version = language_version
        return new_ds


@dataclass(frozen=True)
class SortField:
    name: str
    descending: bool = False


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
        new_model._set_dependencies(dependencies)
        return new_model

    def with_project_name(self, project_name: str) -> "Model":
        new_asset = super().with_project_name(project_name=project_name)
        new_model = copy.deepcopy(self)
        new_model._update_with(new_asset)
        return new_model

    def with_language_version(self, language_version: Tuple[int, int, int]) -> "Model":
        new_model = copy.deepcopy(self)
        new_model.language_version = language_version
        return new_model

    def drop_dependencies(self) -> "Model":
        return self.with_dependencies(())

    def __str__(self) -> str:
        return f"Model({self.name})"


@dataclass(frozen=True)
class User(abc.ABC):
    name: str
    email: str
    first_name: str
    last_name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    account_id: Optional[uuid.UUID] = field(default_factory=uuid.uuid4)


@enum.unique
class Fabric(enum.Enum):
    F_LOCAL = "f-local"
    F_XXSMALL = "f-xxsmall"
    F_XSMALL = "f-xsmall"
    F_SMALL = "f-small"
    F_MEDIUM = "f-medium"
    F_GPU_SMALL = "f-gpu-small"

    @classmethod
    def has_member_key(cls, key: str) -> bool:
        try:
            cls.__new__(cls, key)
            return True
        except ValueError:
            return False

    @classmethod
    def default(cls) -> str:
        return Fabric.F_SMALL.value

    def is_gpu(self) -> bool:
        return "gpu" in self.value
