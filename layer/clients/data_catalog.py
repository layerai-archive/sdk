import math
import tempfile
import uuid
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence

import pandas
import pyarrow
from layerapi.api.entity.dataset_build_pb2 import DatasetBuild as PBDatasetBuild
from layerapi.api.entity.dataset_list_options_pb2 import (
    DatasetListOptions,
    DatasetSortField,
)
from layerapi.api.entity.dataset_pb2 import Dataset as PBDataset
from layerapi.api.entity.dataset_version_pb2 import DatasetVersion as PBDatasetVersion
from layerapi.api.ids_pb2 import DatasetBuildId, DatasetId, DatasetVersionId, ProjectId
from layerapi.api.service.datacatalog.data_catalog_api_pb2 import (
    CompleteBuildRequest,
    CompleteBuildResponse,
    GetBuildRequest,
    GetDatasetRequest,
    GetDatasetsRequest,
    GetLatestBuildRequest,
    GetPythonDatasetAccessCredentialsRequest,
    GetPythonDatasetAccessCredentialsResponse,
    GetResourcePathsRequest,
    GetVersionRequest,
    InitiateBuildRequest,
    InitiateBuildResponse,
    RegisterDatasetRequest,
    UpdateResourcePathsIndexRequest,
)
from layerapi.api.service.datacatalog.data_catalog_api_pb2_grpc import (
    DataCatalogAPIStub,
)
from layerapi.api.service.dataset.dataset_api_pb2 import (
    Command,
    DatasetQuery,
    DatasetSnapshot,
)
from layerapi.api.value.language_version_pb2 import LanguageVersion
from layerapi.api.value.python_dataset_pb2 import PythonDataset as PBPythonDataset
from layerapi.api.value.python_source_pb2 import PythonSource
from layerapi.api.value.s3_path_pb2 import S3Path
from layerapi.api.value.storage_location_pb2 import StorageLocation
from layerapi.api.value.ticket_pb2 import DatasetPathTicket, DataTicket
from pyarrow import flight

from layer.config import ClientConfig
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.datasets import (
    Dataset,
    DatasetBuild,
    DatasetBuildStatus,
    SortField,
)
from layer.contracts.runs import DatasetFunctionDefinition
from layer.exceptions.exceptions import LayerClientException
from layer.utils.file_utils import tar_directory
from layer.utils.grpc import create_grpc_channel, generate_client_error_from_grpc_error
from layer.utils.s3 import S3Util

from .dataset_service import DatasetClient, DatasetClientError


# Number of rows to send in a single chunk, but still bounded by the gRPC max message size.
# Allow to send rows on average up to 1MB, assuming default max gRPC message size is 4MB
_STORE_DATASET_MAX_CHUNK_SIZE = 4


class DataCatalogClient:
    _service: DataCatalogAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
        *,
        service_factory: Callable[..., DataCatalogAPIStub] = DataCatalogAPIStub,
        flight_client: flight.FlightClient = None,
        dataset_client: Optional["DatasetClient"] = None,
    ):
        self._config = config.data_catalog
        self._logger = logger
        self._service_factory = service_factory
        self._access_token = config.access_token
        self._call_metadata = [("authorization", f"Bearer {config.access_token}")]
        self._flight_client = flight_client
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path
        self._s3_endpoint_url = config.s3.endpoint_url
        self._dataset_client = (
            DatasetClient(
                address_and_port=config.grpc_gateway_address,
                access_token=config.access_token,
            )
            if dataset_client is None
            else dataset_client
        )

    @contextmanager
    def init(self) -> Iterator["DataCatalogClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = self._service_factory(channel)
            yield self

    def _get_python_dataset_access_credentials(
        self, dataset_path: AssetPath
    ) -> GetPythonDatasetAccessCredentialsResponse:
        return self._service.GetPythonDatasetAccessCredentials(
            GetPythonDatasetAccessCredentialsRequest(dataset_path=dataset_path.path()),
        )

    def fetch_dataset(
        self, asset_path: AssetPath, no_cache: bool = False
    ) -> "pandas.DataFrame":
        data_ticket = DataTicket(
            dataset_path_ticket=DatasetPathTicket(path=asset_path.path()),
        )
        command = Command(dataset_query=DatasetQuery(ticket=data_ticket))
        all_partition_data = []
        try:
            for partition_metadata in self._dataset_client.get_partitions_metadata(
                command
            ):
                partition = self._dataset_client.fetch_partition(
                    partition_metadata, no_cache=no_cache
                )
                all_partition_data.append(partition.to_pandas())
        except DatasetClientError as e:
            raise LayerClientException(str(e))

        if len(all_partition_data) > 0:
            df = pandas.concat(all_partition_data, ignore_index=True)
        else:
            df = pandas.DataFrame()

        return df

    def _get_dataset_writer(self, name: str, build_id: uuid.UUID, schema: Any) -> Any:
        dataset_snapshot = DatasetSnapshot(build_id=DatasetBuildId(value=str(build_id)))
        return self._dataset_client.get_dataset_writer(
            Command(dataset_snapshot=dataset_snapshot), schema
        )

    def store_dataset(
        self,
        name: str,
        data: "pandas.DataFrame",
        build_id: uuid.UUID,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        Store a dataset.

        :param name: the name of the dataset
            If empty, the data is written to internal storage
        :param data: dataset data
        :param build_id: dataset build id
        :param progress_callback: progress callback
        """

        # Creates a Record batch from the pandas dataframe
        batch = pyarrow.RecordBatch.from_pandas(data, preserve_index=False)
        try:
            writer, _ = self._get_dataset_writer(name, build_id, batch.schema)
            try:
                # We stream the batch in smaller chunks to be able to track the progress
                # Ref: https://arrow.apache.org/docs/python/ipc.html#efficiently-writing-and-reading-arrow-data
                for x in range(
                    math.ceil(batch.num_rows / _STORE_DATASET_MAX_CHUNK_SIZE)
                ):
                    start_index = x * _STORE_DATASET_MAX_CHUNK_SIZE
                    length = min(
                        _STORE_DATASET_MAX_CHUNK_SIZE, batch.num_rows - start_index
                    )
                    if progress_callback:
                        progress_callback(length)
                    writer.write_batch(batch.slice(start_index, length))
            finally:
                writer.close()
        except Exception as err:
            raise generate_client_error_from_grpc_error(
                err, "internal dataset store error"
            )

    def initiate_build(
        self,
        dataset: DatasetFunctionDefinition,
        project_id: uuid.UUID,
        is_local: bool,
    ) -> InitiateBuildResponse:
        self._logger.debug(
            "Initiating build for the dataset %r",
            dataset.name,
        )

        resp = self._service.InitiateBuild(
            InitiateBuildRequest(
                dataset_name=dataset.name,
                format="python",
                build_entity_type=PBDatasetBuild.BUILD_ENTITY_TYPE_DATASET,
                project_id=ProjectId(value=str(project_id)),
                fabric=dataset.get_fabric(is_local),
            )
        )

        return resp

    def complete_build(
        self,
        dataset_build_id: DatasetBuildId,
        dataset: DatasetFunctionDefinition,
        error: Optional[Exception] = None,
    ) -> CompleteBuildResponse:
        self._logger.debug(
            "Completing build for the dataset %r",
            dataset.name,
        )

        if error:
            max_error_length = 99_999
            placeholder = "..."
            raw_text = str.format("Dataset build failed with {}", error)
            status = DatasetBuildStatus.FAILED
            success = None
            failure = CompleteBuildRequest.BuildFailed(
                info=(raw_text[: (max_error_length - len(placeholder))] + placeholder)
                if len(raw_text) > max_error_length
                else raw_text
            )
        else:
            status = DatasetBuildStatus.COMPLETED
            success = CompleteBuildRequest.BuildSuccess(
                location=StorageLocation(uri=dataset.uri), schema="{}"
            )
            failure = None

        resp = self._service.CompleteBuild(
            CompleteBuildRequest(
                id=dataset_build_id,
                status=status,
                success=success,
                failure=failure,
            )
        )

        return resp

    def add_dataset(
        self,
        project_id: uuid.UUID,
        dataset_definition: DatasetFunctionDefinition,
        is_local: bool,
    ) -> DatasetFunctionDefinition:
        self._logger.debug(
            "Adding or updating a dataset with name %r",
            dataset_definition.name,
        )
        resp = self._service.RegisterDataset(
            RegisterDatasetRequest(
                name=dataset_definition.name,
                description=dataset_definition.description,
                python_dataset=self._get_pb_python_dataset(
                    dataset_definition, is_local
                ),
                project_id=ProjectId(value=str(project_id)),
            ),
        )
        return dataset_definition.with_repository_id(resp.dataset_id.value)

    def _get_pb_python_dataset(
        self,
        dataset: DatasetFunctionDefinition,
        is_local: bool,
    ) -> PBPythonDataset:
        s3_path = self._upload_dataset_source(dataset)
        return PBPythonDataset(
            s3_path=s3_path,
            python_source=PythonSource(
                content=dataset.func_source,
                entrypoint=dataset.entrypoint,
                environment=dataset.environment,
                language_version=LanguageVersion(
                    major=dataset.language_version[0],
                    minor=dataset.language_version[1],
                    micro=dataset.language_version[2],
                ),
            ),
            fabric=dataset.get_fabric(is_local),
        )

    def _upload_dataset_source(self, dataset: DatasetFunctionDefinition) -> S3Path:
        response = self._get_python_dataset_access_credentials(dataset.asset_path)
        archive_name = f"{dataset.name}.tgz"

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = f"{tmp_dir}/{archive_name}"
            tar_directory(archive_path, dataset.entity_path)
            S3Util.upload_dir(
                Path(tmp_dir),
                response.credentials,
                response.s3_path,
                endpoint_url=self._s3_endpoint_url,
            )
        return S3Path(
            bucket=response.s3_path.bucket,
            key=f"{response.s3_path.key}{archive_name}",
        )

    def get_resource_paths(
        self, project_name: str, function_name: str, path: str = ""
    ) -> List[str]:
        request = GetResourcePathsRequest(
            project_name=project_name, function_name=function_name, path=path
        )
        response = self._service.GetResourcePaths(request)
        return response.paths

    def update_resource_paths_index(
        self, project_name: str, function_name: str, paths: List[str]
    ) -> None:
        request = UpdateResourcePathsIndexRequest(
            project_name=project_name, function_name=function_name, paths=paths
        )
        self._service.UpdateResourcePathsIndex(request)

    def get_dataset_by_id(self, id_: uuid.UUID) -> Dataset:
        dataset = self._get_dataset_by_id(str(id_))
        build = self._get_build_by_id(dataset.default_build_id.value)
        version = self._get_version_by_id(build.dataset_version_id.value)
        return self._create_dataset(dataset, version, build)

    def get_dataset_by_name(
        self, project_id: uuid.UUID, name: str, version_name: str = ""
    ) -> Dataset:
        build = self._get_build_by_name(project_id, name, version_name)
        version = self._get_version_by_id(build.dataset_version_id.value)
        dataset = self._get_dataset_by_id(version.dataset_id.value)
        return self._create_dataset(dataset, version, build)

    def get_dataset_by_build_id(self, id_: uuid.UUID) -> Dataset:
        build = self._get_build_by_id(str(id_))
        version = self._get_version_by_id(build.dataset_version_id.value)
        dataset = self._get_dataset_by_id(version.dataset_id.value)
        return self._create_dataset(dataset, version, build)

    def list_datasets(
        self,
        sort_fields: Sequence[SortField] = (),
        query_build: bool = True,
    ) -> Iterator[Dataset]:
        for dataset in self._list_datasets(sort_fields):
            if query_build:
                build = self._get_build_by_id(dataset.default_build_id.value)
                version = self._get_version_by_id(build.dataset_version_id.value)
            else:
                build = PBDatasetBuild()
                version = PBDatasetVersion()
            yield self._create_dataset(dataset, version, build)

    def _create_dataset(
        self,
        dataset: PBDataset,
        version: PBDatasetVersion,
        build: PBDatasetBuild,
    ) -> Dataset:

        asset_path = AssetPath(
            asset_type=AssetType.DATASET,
            entity_name=dataset.name,
            entity_version=version.name,
            project_name=dataset.project_name,
        )
        return Dataset(
            id=uuid.UUID(dataset.id.value),
            asset_path=asset_path,
            description=dataset.description,
            schema=version.schema,
            version=version.name,
            uri=build.location.uri,
            metadata=dict(build.location.metadata.value),
            build=DatasetBuild(
                id=build.id.value,
                status=DatasetBuildStatus(build.build_info.status),
                info=build.build_info.info,
                index=str(build.index),
            ),
        )

    def _get_dataset_by_id(self, id_: str) -> PBDataset:
        return self._service.GetDataset(
            GetDatasetRequest(dataset_id=DatasetId(value=id_)),
        ).dataset

    def _get_version_by_id(self, id_: str) -> PBDatasetVersion:
        return self._service.GetVersion(
            GetVersionRequest(version_id=DatasetVersionId(value=id_)),
        ).version

    def _get_build_by_id(self, id_: str) -> PBDatasetBuild:
        return self._service.GetBuild(
            GetBuildRequest(build_id=DatasetBuildId(value=id_)),
        ).build

    def _get_build_by_name(
        self, project_id: uuid.UUID, name: str, version: str = ""
    ) -> PBDatasetBuild:
        return self._service.GetLatestBuild(
            GetLatestBuildRequest(
                dataset_name=name,
                dataset_version=version,
                project_id=ProjectId(value=str(project_id)),
            ),
        ).build

    def _list_datasets(self, sort_fields: Sequence[SortField] = ()) -> List[PBDataset]:
        return list(
            self._service.GetDatasets(
                GetDatasetsRequest(
                    dataset_list_options=DatasetListOptions(
                        sorting=[
                            DatasetSortField(
                                name=sort_field.name, descending=sort_field.descending
                            )
                            for sort_field in sort_fields
                        ]
                    )
                ),
            ).datasets
        )
