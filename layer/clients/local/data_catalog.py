import os
import uuid
from contextlib import contextmanager
from logging import Logger
from typing import Any, Callable, Generator, Iterator, List, Optional, Sequence, Tuple

import pandas
import pandas as pd
from layerapi.api.ids_pb2 import DatasetBuildId, DatasetId, DatasetVersionId, ProjectId
from layerapi.api.service.datacatalog.data_catalog_api_pb2 import (
    CompleteBuildResponse,
    GetPythonDatasetAccessCredentialsResponse,
    InitiateBuildResponse,
)
from layerapi.api.service.datacatalog.data_catalog_api_pb2_grpc import (
    DataCatalogAPIStub,
)

from layer.clients.local.asset_db import AssetDB
from layer.config import ClientConfig
from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.datasets import Dataset, SortField
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import FunctionDefinition


class DataCatalogClient:
    _service: DataCatalogAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.data_catalog
        self._logger = logger

    @contextmanager
    def init(self):
        return self

    def _get_python_dataset_access_credentials(
        self, dataset_path: AssetPath
    ) -> GetPythonDatasetAccessCredentialsResponse:
        pass

    def fetch_dataset(
        self, asset_path: AssetPath, no_cache: bool = False
    ) -> "pandas.DataFrame":
        asset = AssetDB.get_asset_by_name_and_version(
            asset_path.project_name,
            asset_path.asset_name,
            asset_path.asset_version,
            AssetType.DATASET,
        )
        df = pd.read_pickle(asset.path / "dataset" / "dataframe.pkl")
        return df

    def _get_dataset_writer(self, name: str, build_id: uuid.UUID, schema: Any) -> Any:
        pass

    def store_dataset(
        self,
        name: str,
        data: "pandas.DataFrame",
        build_id: uuid.UUID,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        from layer.clients.local.layer_local_client import LayerLocalClient

        project_name = LayerLocalClient.get_project_name()
        asset = AssetDB.get_asset_by_id(project_name, build_id.hex, AssetType.DATASET)
        dataset_path = asset.path / "dataset"
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        data.to_pickle(dataset_path / "dataframe.pkl")

    def initiate_build(
        self,
        dataset: FunctionDefinition,
        project_id: uuid.UUID,
        is_local: bool,
    ) -> InitiateBuildResponse:
        from layer.clients.local.layer_local_client import LayerLocalClient

        project_name = LayerLocalClient.get_project_name()
        db = AssetDB(project_name)
        build_id = uuid.uuid4().hex
        asset = db.get_asset(dataset.asset_name, build_id, AssetType.DATASET)
        return InitiateBuildResponse(id=DatasetBuildId(value=build_id))

    def complete_build(
        self,
        dataset_build_id: DatasetBuildId,
        dataset: FunctionDefinition,
        error: Optional[Exception] = None,
    ) -> CompleteBuildResponse:
        pass

    def add_dataset(
        self,
        project_id: uuid.UUID,
        dataset_definition: FunctionDefinition,
        is_local: bool,
    ) -> str:
        version = uuid.uuid4().hex
        return version

    def get_resource_paths(
        self, project_full_name: ProjectFullName, function_name: str, path: str = ""
    ) -> List[str]:
        return []

    def update_resource_paths_index(
        self, project_full_name: ProjectFullName, function_name: str, paths: List[str]
    ) -> None:
        pass

    def get_dataset_by_id(self, id_: uuid.UUID) -> Dataset:
        pass

    def get_dataset_by_name(
        self, project_id: uuid.UUID, name: str, version_name: str = ""
    ) -> Dataset:
        pass

    def get_dataset_by_build_id(self, id_: uuid.UUID) -> Dataset:
        pass

    def list_datasets(
        self,
        sort_fields: Sequence[SortField] = (),
        query_build: bool = True,
    ) -> Iterator[Dataset]:
        pass
