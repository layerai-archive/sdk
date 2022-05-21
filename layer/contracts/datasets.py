import abc
import copy
import enum
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas

from .asset import AssetPath, AssetType, BaseAsset


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
        new_ds._set_id(id)  # pylint: disable=protected-access
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
        new_ds._path = new_asset_path  # pylint: disable=protected-access
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
        new_ds._set_dependencies(dependencies)  # pylint: disable=protected-access
        return new_ds

    def drop_dependencies(self) -> "DerivedDataset":
        return self.with_dependencies(())

    def with_project_name(self, project_name: str) -> "DerivedDataset":
        new_asset_path = self._path.with_project_name(project_name=project_name)
        new_ds = copy.deepcopy(self)
        new_ds._path = new_asset_path  # pylint: disable=protected-access
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
