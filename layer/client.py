import math
import os
import tarfile
import tempfile
import uuid
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import pandas
import polling  # type: ignore
import pyarrow
from google.protobuf.timestamp_pb2 import Timestamp
from layerapi.api.entity.dataset_build_pb2 import DatasetBuild as PBDatasetBuild
from layerapi.api.entity.dataset_list_options_pb2 import (
    DatasetListOptions,
    DatasetSortField,
)
from layerapi.api.entity.dataset_pb2 import Dataset as PBDataset
from layerapi.api.entity.dataset_version_pb2 import DatasetVersion as PBDatasetVersion
from layerapi.api.entity.history_event_pb2 import HistoryEvent
from layerapi.api.entity.hyperparameter_tuning_pb2 import (
    HyperparameterTuning as PBHyperparameterTuning,
)
from layerapi.api.entity.model_pb2 import Model as PBModel
from layerapi.api.entity.model_train_pb2 import ModelTrain as PBModelTrain
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.entity.operations_pb2 import ExecutionPlan
from layerapi.api.entity.run_filter_pb2 import RunFilter
from layerapi.api.entity.run_metadata_pb2 import RunMetadata
from layerapi.api.entity.run_pb2 import Run
from layerapi.api.entity.source_code_environment_pb2 import SourceCodeEnvironment
from layerapi.api.entity.user_log_line_pb2 import UserLogLine
from layerapi.api.ids_pb2 import (
    DatasetBuildId,
    DatasetId,
    DatasetVersionId,
    HyperparameterTuningId,
    ModelTrainId,
    ModelVersionId,
    ProjectId,
    RunId,
)
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
    RegisterRawDatasetRequest,
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
from layerapi.api.service.flowmanager.flow_manager_api_pb2 import (
    GetRunByIdRequest,
    GetRunHistoryAndMetadataRequest,
    GetRunsRequest,
    StartRunV2Request,
    TerminateRunRequest,
)
from layerapi.api.service.flowmanager.flow_manager_api_pb2_grpc import (
    FlowManagerAPIStub,
)
from layerapi.api.service.modelcatalog.model_catalog_api_pb2 import (
    CompleteModelTrainRequest,
    CreateModelTrainFromVersionIdRequest,
    CreateModelTrainRequest,
    CreateModelVersionRequest,
    CreateModelVersionResponse,
    GetModelByPathRequest,
    GetModelByPathResponse,
    GetModelTrainParametersRequest,
    GetModelTrainRequest,
    GetModelTrainResponse,
    GetModelTrainStorageConfigurationRequest,
    GetModelVersionRequest,
    GetModelVersionResponse,
    LoadModelTrainDataByPathRequest,
    LogModelTrainParametersRequest,
    LogModelTrainParametersResponse,
    SetHyperparameterTuningIdRequest,
    StartModelTrainRequest,
    StoreTrainingMetadataRequest,
    UpdateModelTrainStatusRequest,
)
from layerapi.api.service.modelcatalog.model_catalog_api_pb2_grpc import (
    ModelCatalogAPIStub,
)
from layerapi.api.service.modeltraining.model_training_api_pb2 import (
    CreateHyperparameterTuningRequest,
    GetHyperparameterTuningRequest,
    GetHyperparameterTuningStatusRequest,
    GetModelTrainStatusRequest,
    GetSourceCodeUploadCredentialsRequest,
    GetSourceCodeUploadCredentialsResponse,
    StartHyperparameterTuningRequest,
    StartModelTrainingRequest,
    StoreHyperparameterTuningMetadataRequest,
    UpdateHyperparameterTuningRequest,
)
from layerapi.api.service.modeltraining.model_training_api_pb2_grpc import (
    ModelTrainingAPIStub,
)
from layerapi.api.service.user_logs.user_logs_api_pb2 import (
    GetPipelineRunLogsRequest,
    GetPipelineRunLogsResponse,
)
from layerapi.api.service.user_logs.user_logs_api_pb2_grpc import UserLogsAPIStub
from layerapi.api.value.dependency_pb2 import DependencyFile
from layerapi.api.value.hyperparameter_tuning_metadata_pb2 import (
    HyperparameterTuningMetadata,
)
from layerapi.api.value.language_version_pb2 import LanguageVersion
from layerapi.api.value.metadata_pb2 import Metadata
from layerapi.api.value.python_dataset_pb2 import PythonDataset as PBPythonDataset
from layerapi.api.value.python_source_pb2 import PythonSource
from layerapi.api.value.s3_path_pb2 import S3Path
from layerapi.api.value.sha256_pb2 import Sha256
from layerapi.api.value.source_code_pb2 import RemoteFileLocation, SourceCode
from layerapi.api.value.storage_location_pb2 import StorageLocation
from layerapi.api.value.ticket_pb2 import DatasetPathTicket, DataTicket
from pyarrow import flight

from layer.clients.dataset_client import DatasetClient, DatasetClientError
from layer.config import ClientConfig
from layer.contracts.asset import AssetPath, AssetType
from layer.exceptions.exceptions import LayerClientException
from layer.grpc_utils import create_grpc_channel, generate_client_error_from_grpc_error
from layer.projects.tracker.project_progress_tracker import ProjectProgressTracker
from layer.projects.tracker.resource_transfer_state import ResourceTransferState

from . import Dataset, Model
from .contracts.datasets import (
    DatasetBuild,
    DatasetBuildStatus,
    PythonDataset,
    RawDataset,
    SortField,
)
from .contracts.models import (
    BayesianSearch,
    GridSearch,
    HyperparameterTuning,
    ManualSearch,
    ParameterType,
    ParameterValue,
    RandomSearch,
    TrainStorageConfiguration,
)
from .mlmodels.flavors import ModelFlavor
from .mlmodels.flavors.model_definition import ModelDefinition
from .mlmodels.service import MLModelService
from .s3 import S3Util


if TYPE_CHECKING:
    from .mlmodels import ModelObject

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
        self, dataset_path: str
    ) -> GetPythonDatasetAccessCredentialsResponse:
        return self._service.GetPythonDatasetAccessCredentials(
            GetPythonDatasetAccessCredentialsRequest(dataset_path=dataset_path),
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
        self, dataset: Dataset, project_id: uuid.UUID
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
                fabric=dataset.fabric,
            )
        )

        return resp

    def complete_build(
        self,
        dataset_build_id: DatasetBuildId,
        dataset: Dataset,
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

    def add_dataset(self, project_id: uuid.UUID, dataset: Dataset) -> Dataset:
        self._logger.debug(
            "Adding or updating a dataset with name %r",
            dataset.name,
        )
        resp = self._service.RegisterDataset(
            RegisterDatasetRequest(
                name=dataset.name,
                description=dataset.description,
                python_dataset=self._get_pb_python_dataset(dataset)
                if isinstance(dataset, PythonDataset)
                else None,
                project_id=ProjectId(value=str(project_id)),
            ),
        )
        return dataset.with_id(uuid.UUID(resp.dataset_id.value))

    def add_raw_dataset(self, project_id: uuid.UUID, dataset: Dataset) -> Dataset:
        self._logger.debug(
            "Adding or updating a raw dataset with name %r",
            dataset.name,
        )
        if isinstance(dataset, RawDataset):
            resp = self._service.RegisterRawDataset(
                RegisterRawDatasetRequest(
                    name=dataset.name,
                    description=dataset.description,
                    success=RegisterRawDatasetRequest.BuildSuccess(
                        location=StorageLocation(
                            uri=dataset.uri, metadata=Metadata(value=dataset.metadata)
                        )
                    ),
                    project_id=ProjectId(value=str(project_id)),
                ),
            )
            return dataset.with_id(uuid.UUID(resp.dataset_id.value))
        else:
            raise LayerClientException("Cannot add a non-raw dataset as raw")

    def _get_pb_python_dataset(self, dataset: PythonDataset) -> PBPythonDataset:
        s3_path = self._upload_dataset_source(dataset)
        return PBPythonDataset(
            s3_path=s3_path,
            python_source=PythonSource(
                content=dataset.entrypoint_content,
                entrypoint=dataset.entrypoint,
                environment=dataset.environment,
                language_version=LanguageVersion(
                    major=dataset.language_version[0],
                    minor=dataset.language_version[1],
                    micro=dataset.language_version[2],
                ),
            ),
            fabric=dataset.fabric,
        )

    def _upload_dataset_source(self, dataset: PythonDataset) -> S3Path:
        response = self._get_python_dataset_access_credentials(dataset.path)
        archive_name = f"{dataset.name}.tgz"
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = f"{tmp_dir}/{archive_name}"
            tar_directory(archive_path, dataset.entrypoint_path.parent)
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
        return RawDataset(
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


class ModelCatalogClient:
    _service: ModelCatalogAPIStub

    def __init__(
        self,
        config: ClientConfig,
        ml_model_service: MLModelService,
        logger: Logger,
    ):
        self._config = config.model_catalog
        self._logger = logger
        self._ml_model_service = ml_model_service
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["ModelCatalogClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = ModelCatalogAPIStub(channel=channel)
            yield self

    def store_training_files_metadata(
        self,
        model: Model,
        s3_path: S3Path,
        version: ModelVersion,
        language_version: Tuple[int, int, int],
    ) -> None:
        train = model.training
        request: StoreTrainingMetadataRequest = StoreTrainingMetadataRequest(
            model_version_id=version.id,
            name=train.name,
            description=train.description,
            source_code_env=SourceCodeEnvironment(
                source_code=SourceCode(
                    remote_file_location=RemoteFileLocation(
                        name=train.entrypoint,
                        location=f"s3://{s3_path.bucket}/{s3_path.key}{train.name}.tgz",
                    ),
                    language=SourceCode.Language.Value("LANGUAGE_PYTHON"),
                    language_version=SourceCode.LanguageVersion(
                        major=int(language_version[0]),
                        minor=int(language_version[1]),
                        micro=int(language_version[2]),
                    ),
                ),
                dependency_file=DependencyFile(
                    name=train.environment,
                    location=train.environment,
                ),
            ),
            entrypoint=train.entrypoint,
            parameters={param.name: param.value for param in train.parameters},
            fabric=train.fabric,
        )
        self._logger.debug(f"StoreTrainingMetadataRequest request: {str(request)}")
        response = self._service.StoreTrainingMetadata(request)
        self._logger.debug(f"StoreTrainingMetadata response: {str(response)}")

    def create_model_version(
        self, project_name: str, model: Model
    ) -> CreateModelVersionResponse:
        """
        Given a model metadata it makes a request to the backend
        and creates a corresponding entity.
        :param project_name: the project name of the model
        :param model: the structured of the parsed entity
        :return: the created model version entity
        """
        model_path = f"{project_name}/models/{model.name}"
        self._logger.debug(
            f"Creating model version for the following model: {model_path}"
        )
        should_create_initial_train = model.training.hyperparameter_tuning is None
        response = self._service.CreateModelVersion(
            CreateModelVersionRequest(
                model_path=model_path,
                description=model.description,
                training_files_hash=Sha256(value=model.training_files_digest),
                should_create_initial_train=should_create_initial_train,
                fabric=model.training.fabric,
            ),
        )
        self._logger.debug(f"CreateModelVersionResponse: {str(response)}")
        return response

    def load_model_definition(self, path: str) -> ModelDefinition:
        load_response = self._service.LoadModelTrainDataByPath(
            LoadModelTrainDataByPathRequest(path=path),
        )
        return ModelDefinition(
            name=path,
            train_id=load_response.id,
            proto_flavor=load_response.flavor,
            s3_path=load_response.s3_path,
            credentials=load_response.credentials,
        )

    def create_model_train_from_version_id(
        self,
        version_id: ModelVersionId,
    ) -> ModelTrainId:
        response = self._service.CreateModelTrainFromVersionId(
            CreateModelTrainFromVersionIdRequest(
                model_version_id=version_id,
            ),
        )
        return response.id

    def load_by_model_definition(
        self,
        model_definition: ModelDefinition,
        no_cache: bool = False,
        state: Optional[ResourceTransferState] = None,
    ) -> Any:
        """
        Loads a model from the model catalog

        :param model_definition: model definition
        :return: a model object
        """
        self._logger.debug(f"Model definition: {model_definition}")
        return self._ml_model_service.retrieve(
            model_definition, no_cache=no_cache, state=state
        )

    def save_model(
        self,
        model_definition: ModelDefinition,
        trained_model_obj: "ModelObject",
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> "ModelObject":
        if not tracker:
            tracker = ProjectProgressTracker()
        flavor = self._ml_model_service.get_model_flavor_from_proto(
            model_definition.proto_flavor
        )
        if not flavor:
            raise LayerClientException("Model flavor not found")
        self._logger.debug(
            f"Storing given model {trained_model_obj} with definition {model_definition}"
        )
        self._ml_model_service.store(
            model_definition=model_definition,
            model_object=trained_model_obj,
            flavor=flavor,
            tracker=tracker,
        )
        return trained_model_obj

    def start_model_train(
        self,
        train_id: ModelTrainId,
    ) -> TrainStorageConfiguration:
        response = self._service.StartModelTrain(
            StartModelTrainRequest(
                model_train_id=train_id,
            ),
        )
        return TrainStorageConfiguration(
            train_id=train_id,
            s3_path=response.s3_path,
            credentials=response.credentials,
        )

    def get_model_train_storage_configuration(
        self,
        train_id: ModelTrainId,
    ) -> TrainStorageConfiguration:
        response = self._service.GetModelTrainStorageConfiguration(
            GetModelTrainStorageConfigurationRequest(
                model_train_id=train_id,
            ),
        )
        return TrainStorageConfiguration(
            train_id=train_id,
            s3_path=response.s3_path,
            credentials=response.credentials,
        )

    def create_model_train(
        self,
        name: str,
        project_name: str,
        version: Optional[str],
    ) -> ModelTrainId:
        response = self._service.CreateModelTrain(
            CreateModelTrainRequest(
                model_name=name,
                model_version="" if version is None else version,
                project_name=project_name,
            ),
        )
        return response.id

    def infer_flavor(self, model_obj: "ModelObject") -> "ModelVersion.ModelFlavor.V":
        tup: Tuple[Any, ModelFlavor] = self._ml_model_service.get_model_flavor(
            model_obj,
            logger=self._logger,
        )
        return tup[0]

    def complete_model_train(
        self, train_id: ModelTrainId, flavor: Optional["ModelVersion.ModelFlavor.V"]
    ) -> None:
        self._service.CompleteModelTrain(
            CompleteModelTrainRequest(id=train_id, flavor=flavor),
        )

    def log_parameter(self, train_id: ModelTrainId, name: str, value: str) -> None:
        """
        Logs given parameter to the model catalog service

        :param train_id: id of the train to associate params with
        :param name: parameter name
        :param value: parameter value
        """
        self.log_parameters(train_id, {name: value})

    def log_parameters(
        self, train_id: ModelTrainId, parameters: Dict[str, str]
    ) -> None:
        """
        Logs given parameters to the model catalog service

        :param train_id: id of the train to associate params with
        :param parameters: map of parameter name to its value
        """
        response: LogModelTrainParametersResponse = (
            self._service.LogModelTrainParameters(
                LogModelTrainParametersRequest(
                    train_id=train_id,
                    parameters=parameters,
                ),
            )
        )
        self._logger.debug(f"LogModelTrainParameters response: {str(response)}")

    def get_model_by_path(self, model_path: str) -> PBModel:
        response: GetModelByPathResponse = self._service.GetModelByPath(
            GetModelByPathRequest(
                path=model_path,
            )
        )
        return response.model

    def get_model_train(self, train_id: ModelTrainId) -> PBModelTrain:
        response: GetModelTrainResponse = self._service.GetModelTrain(
            GetModelTrainRequest(
                model_train_id=train_id,
            ),
        )
        return response.model_train

    def get_model_version(self, version_id: ModelVersionId) -> ModelVersion:
        response: GetModelVersionResponse = self._service.GetModelVersion(
            GetModelVersionRequest(
                model_version_id=version_id,
            ),
        )
        return response.version

    def update_model_train_status(
        self, train_id: ModelTrainId, train_status: "PBModelTrainStatus"
    ) -> None:
        self._service.UpdateModelTrainStatus(
            UpdateModelTrainStatusRequest(
                model_train_id=train_id, train_status=train_status
            )
        )

    def get_model_train_parameters(self, train_id: ModelTrainId) -> Dict[str, str]:
        parameters = self._service.GetModelTrainParameters(
            GetModelTrainParametersRequest(model_train_id=train_id)
        ).parameters
        parameters_dict = {}
        for param in parameters:
            parameters_dict[param.name] = param.value
        return parameters_dict

    def set_hyperparameter_tuning_id(
        self, train_id: ModelTrainId, tuning_id: HyperparameterTuningId
    ) -> None:
        self._service.SetHyperparameterTuningId(
            SetHyperparameterTuningIdRequest(
                model_train_id=train_id, hyperparameter_tuning_id=tuning_id
            )
        )


class ModelTrainingClient:
    _service: ModelTrainingAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.model_training
        self._logger = logger
        self._access_token = config.access_token
        self._s3_endpoint_url = config.s3.endpoint_url
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["ModelTrainingClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = ModelTrainingAPIStub(channel=channel)
            yield self

    def upload_training_files(self, model: Model, source_name: str) -> None:
        response = self.get_source_code_upload_credentials(source_name=source_name)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_directory(
                f"{tmp_dir}/{model.training.name}.tgz", model.local_path.parent
            )
            S3Util.upload_dir(
                Path(tmp_dir),
                response.credentials,
                response.s3_path,
                endpoint_url=self._s3_endpoint_url,
            )

    def store_hyperparameter_tuning_metadata(
        self,
        model: Model,
        s3_path: S3Path,
        version_name: str,
        hyperparameter_tuning_id: HyperparameterTuningId,
        language_version: Tuple[int, int, int],
    ) -> None:
        assert model.training.hyperparameter_tuning is not None
        train = model.training
        hyperparameter_tuning: HyperparameterTuning = train.hyperparameter_tuning  # type: ignore
        hyperparameter_tuning_metadata = HyperparameterTuningMetadata(
            name=train.name,
            description=train.description,
            model_name=model.name,
            model_version=version_name,
            environment=HyperparameterTuningMetadata.Environment(
                source_code_env=SourceCodeEnvironment(
                    source_code=SourceCode(
                        remote_file_location=RemoteFileLocation(
                            name=train.entrypoint,
                            location=f"s3://{s3_path.bucket}/{s3_path.key}{train.name}.tgz",
                        ),
                        language=SourceCode.Language.Value("LANGUAGE_PYTHON"),
                        language_version=SourceCode.LanguageVersion(
                            major=int(language_version[0]),
                            minor=int(language_version[1]),
                            micro=int(language_version[2]),
                        ),
                    ),
                    dependency_file=DependencyFile(
                        name=train.environment,
                        location=train.environment,
                    ),
                ),
            ),
            entrypoint=train.entrypoint,
            objective=HyperparameterTuningMetadata.Objective(
                maximize=hyperparameter_tuning.maximize is not None,
                metric_name=str(hyperparameter_tuning.maximize)
                if hyperparameter_tuning.maximize is not None
                else str(hyperparameter_tuning.minimize),
            ),
            fixed_parameters=hyperparameter_tuning.fixed_parameters,
            strategy=HyperparameterTuningMetadata.Strategy(
                manual_search=self._manual_search_convert(
                    hyperparameter_tuning.manual_search
                ),
                random_search=self._random_search_convert(
                    hyperparameter_tuning.random_search
                ),
                grid_search=self._grid_search_convert(
                    hyperparameter_tuning.grid_search
                ),
                bayesian_search=self._bayesian_search_convert(
                    hyperparameter_tuning.bayesian_search
                ),
            ),
            max_parallel_jobs=hyperparameter_tuning.max_parallel_jobs
            if hyperparameter_tuning.max_parallel_jobs is not None
            else 0,
            early_stop=hyperparameter_tuning.early_stop
            if hyperparameter_tuning.early_stop is not None
            else False,
            fabric=train.fabric,
        )
        self._service.StoreHyperparameterTuningMetadata(
            StoreHyperparameterTuningMetadataRequest(
                hyperparameter_tuning_id=hyperparameter_tuning_id,
                metadata=hyperparameter_tuning_metadata,
            )
        )

    def get_source_code_upload_credentials(
        self, source_name: str
    ) -> GetSourceCodeUploadCredentialsResponse:
        return self._service.GetSourceCodeUploadCredentials(
            GetSourceCodeUploadCredentialsRequest(source_name=source_name)
        )

    def train_model(
        self,
        model: Model,
        version: ModelVersion,
        hyperparameter_tuning_metadata: Dict[str, HyperparameterTuningId],
    ) -> uuid.UUID:
        train = model.training
        response: GetSourceCodeUploadCredentialsResponse = (
            self.get_source_code_upload_credentials(version.id.value)
        )
        self._logger.debug(
            f"GetSourceCodeUploadCredentialsResponse response: {str(response)}"
        )
        if train.hyperparameter_tuning is None:
            return self._execute_regular_train(train_id=version.latest_train_id)
        else:
            hyperparameter_tuning_id = hyperparameter_tuning_metadata[model.name]
            return self._execute_hyperparameter_tuning_train(
                hyperparameter_tuning_id=hyperparameter_tuning_id
            )

    def create_hpt_id(self, version: ModelVersion) -> HyperparameterTuningId:
        return self._service.CreateHyperparameterTuning(
            CreateHyperparameterTuningRequest(model_version_id=version.id)
        ).hyperparameter_tuning_id

    def get_hyperparameter_tuning(self, id: uuid.UUID) -> PBHyperparameterTuning:
        request = GetHyperparameterTuningRequest(
            hyperparameter_tuning_id=HyperparameterTuningId(value=str(id))
        )
        resp = self._service.GetHyperparameterTuning(request)
        return resp.hyperparameter_tuning

    def get_hyperparameter_tuning_status(
        self, id: uuid.UUID
    ) -> "PBHyperparameterTuning.Status.V":
        request = GetHyperparameterTuningStatusRequest(
            hyperparameter_tuning_id=HyperparameterTuningId(value=str(id))
        )
        resp = self._service.GetHyperparameterTuningStatus(request)
        return resp.hyperparameter_tuning_status

    def update_hyperparameter_tuning(
        self,
        hyperparameter_tuning_id: uuid.UUID,
        data: Optional[str],
        output_model_train_id: Optional[uuid.UUID],
        status: PBHyperparameterTuning.Status,
        status_info: Optional[str],
        start_time: Optional[Timestamp],
        finish_time: Optional[Timestamp],
    ) -> None:
        request = UpdateHyperparameterTuningRequest(
            hyperparameter_tuning_id=HyperparameterTuningId(
                value=str(hyperparameter_tuning_id)
            ),
            data=data if data is not None else "",
            output_model_train_id=ModelTrainId(value=str(output_model_train_id))
            if output_model_train_id is not None
            else None,
            status=status,
            start_time=start_time,
            finish_time=finish_time,
            status_info=status_info if status_info is not None else "",
        )
        self._service.UpdateHyperparameterTuning(request)

    def _execute_regular_train(self, train_id: ModelTrainId) -> uuid.UUID:
        request: StartModelTrainingRequest = StartModelTrainingRequest(
            model_train_id=train_id
        )
        self._logger.debug(f"StartExecuteModelTrainRequest request: {str(request)}")
        train_response = self._service.StartModelTraining(request)

        def is_train_completed(status: PBModelTrainStatus) -> bool:
            return (
                status.train_status == PBModelTrainStatus.TRAIN_STATUS_SUCCESSFUL
                or status.train_status == PBModelTrainStatus.TRAIN_STATUS_FAILED
            )

        polling.poll(
            lambda: self._get_model_train_status(train_response.id.value),
            check_success=is_train_completed,
            step=5,
            poll_forever=True,
        )
        status = self._get_model_train_status(train_response.id.value)
        if status.train_status == PBModelTrainStatus.TRAIN_STATUS_FAILED:
            raise LayerClientException(f"regular train failed. Info: {status.info}")
        return uuid.UUID(train_response.id.value)

    def _get_model_train_status(self, id: uuid.UUID) -> PBModelTrainStatus:
        response = self._service.GetModelTrainStatus(
            GetModelTrainStatusRequest(id=ModelTrainId(value=str(id)))
        )
        return response.train_status

    def _parameter_type_convert(
        self, type: ParameterType
    ) -> "HyperparameterTuningMetadata.Strategy.ParameterType.V":
        if type == ParameterType.STRING:
            return HyperparameterTuningMetadata.Strategy.PARAMETER_TYPE_STRING
        elif type == ParameterType.FLOAT:
            return HyperparameterTuningMetadata.Strategy.PARAMETER_TYPE_FLOAT
        elif type == ParameterType.INT:
            return HyperparameterTuningMetadata.Strategy.PARAMETER_TYPE_INT
        else:
            return HyperparameterTuningMetadata.Strategy.PARAMETER_TYPE_INVALID

    def _parameter_value_convert(
        self, type: ParameterType, value: ParameterValue
    ) -> HyperparameterTuningMetadata.Strategy.ParameterValue:
        if type == ParameterType.STRING:
            assert value.string_value is not None
            return HyperparameterTuningMetadata.Strategy.ParameterValue(
                string_value=value.string_value
            )
        elif type == ParameterType.FLOAT:
            assert value.float_value is not None
            return HyperparameterTuningMetadata.Strategy.ParameterValue(
                float_value=value.float_value
            )
        elif type == ParameterType.INT:
            assert value.int_value is not None
            return HyperparameterTuningMetadata.Strategy.ParameterValue(
                int_value=value.int_value
            )
        else:
            raise LayerClientException("Unspecified parameter value")

    def _manual_search_convert(
        self,
        search: Optional[ManualSearch],
    ) -> Optional[HyperparameterTuningMetadata.Strategy.ManualSearch]:
        if search is None:
            return None
        parsed_params = []
        for param_combination in search.parameters:
            mapped_params = {}
            for param in param_combination:
                value = HyperparameterTuningMetadata.Strategy.ParameterInfo(
                    type=self._parameter_type_convert(param.type),
                    value=self._parameter_value_convert(
                        type=param.type, value=param.value
                    ),
                )
                mapped_params[param.name] = value
            param_val = (
                HyperparameterTuningMetadata.Strategy.ManualSearch.ParameterValues(
                    parameter_to_value=mapped_params
                )
            )
            parsed_params.append(param_val)

        return HyperparameterTuningMetadata.Strategy.ManualSearch(
            parameter_values=parsed_params
        )

    def _random_search_convert(
        self,
        search: Optional[RandomSearch],
    ) -> Optional[HyperparameterTuningMetadata.Strategy.RandomSearch]:
        if search is None:
            return None
        parsed_params = {}
        for param in search.parameters:
            parsed_params[
                param.name
            ] = HyperparameterTuningMetadata.Strategy.RandomSearch.ParameterRange(
                type=self._parameter_type_convert(param.type),
                min=self._parameter_value_convert(type=param.type, value=param.min),
                max=self._parameter_value_convert(type=param.type, value=param.max),
            )

        parsed_params_categorical = {}
        for param_categorical in search.parameters_categorical:
            parsed_params_categorical[
                param_categorical.name
            ] = HyperparameterTuningMetadata.Strategy.RandomSearch.ParameterCategoricalRange(
                type=self._parameter_type_convert(param_categorical.type),
                values=[
                    self._parameter_value_convert(
                        param_categorical.type, value=param_val
                    )
                    for param_val in param_categorical.values
                ],
            )

        return HyperparameterTuningMetadata.Strategy.RandomSearch(
            parameters=parsed_params,
            parameters_categorical=parsed_params_categorical,
            max_jobs=search.max_jobs,
        )

    def _grid_search_convert(
        self,
        search: Optional[GridSearch],
    ) -> Optional[HyperparameterTuningMetadata.Strategy.GridSearch]:
        if search is None:
            return None
        parsed_params = {}
        for param in search.parameters:
            parsed_params[
                param.name
            ] = HyperparameterTuningMetadata.Strategy.GridSearch.ParameterRange(
                type=self._parameter_type_convert(param.type),
                min=self._parameter_value_convert(type=param.type, value=param.min),
                max=self._parameter_value_convert(type=param.type, value=param.max),
                step=self._parameter_value_convert(type=param.type, value=param.step),
            )

        return HyperparameterTuningMetadata.Strategy.GridSearch(
            parameters=parsed_params,
        )

    def _bayesian_search_convert(
        self,
        search: Optional[BayesianSearch],
    ) -> Optional[HyperparameterTuningMetadata.Strategy.BayesianSearch]:
        if search is None:
            return None
        parsed_params = {}
        for param in search.parameters:
            parsed_params[
                param.name
            ] = HyperparameterTuningMetadata.Strategy.BayesianSearch.ParameterRange(
                type=self._parameter_type_convert(param.type),
                min=self._parameter_value_convert(type=param.type, value=param.min),
                max=self._parameter_value_convert(type=param.type, value=param.max),
            )

        return HyperparameterTuningMetadata.Strategy.BayesianSearch(
            parameters=parsed_params,
            max_jobs=search.max_jobs,
        )

    def _execute_hyperparameter_tuning_train(
        self,
        hyperparameter_tuning_id: HyperparameterTuningId,
    ) -> uuid.UUID:
        request_tuning = StartHyperparameterTuningRequest(
            hyperparameter_tuning_id=hyperparameter_tuning_id,
        )
        self._logger.debug(
            f"StartHyperparameterTuningRequest request: {str(request_tuning)}"
        )
        tuning_response = self._service.StartHyperparameterTuning(request_tuning)

        def is_tuning_completed(status: "PBHyperparameterTuning.Status.V") -> bool:
            return (
                status == PBHyperparameterTuning.STATUS_FINISHED
                or status == PBHyperparameterTuning.STATUS_FAILED
            )

        polling.poll(
            lambda: self.get_hyperparameter_tuning_status(tuning_response.id.value),
            check_success=is_tuning_completed,
            step=5,
            poll_forever=True,
        )
        hpt = self.get_hyperparameter_tuning(tuning_response.id.value)
        if hpt.status == PBHyperparameterTuning.STATUS_FAILED:
            raise LayerClientException(f"HPT failed. Info: {hpt.status}")
        return uuid.UUID(hpt.output_model_train_id.value)


DEFAULT_READ_TIMEOUT_SEC = 2.0


GetRunsFunction = Callable[[], List[Run]]


class FlowManagerClient:
    _service: FlowManagerAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.flow_manager
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["FlowManagerClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
        ) as channel:
            self._service = FlowManagerAPIStub(channel=channel)
            yield self

    def start_run(
        self,
        name: str,
        execution_plan: ExecutionPlan,
        project_files_hash: str,
        user_command: str,
    ) -> RunId:
        response = self._service.StartRunV2(
            request=StartRunV2Request(
                project_name=name,
                plan=execution_plan,
                project_files_hash=Sha256(value=project_files_hash),
                user_command=user_command,
            )
        )
        return response.run_id

    def get_run(self, run_id: RunId) -> Run:
        response = self._service.GetRunById(GetRunByIdRequest(run_id=run_id))
        return response.run

    def get_run_status_history_and_metadata(
        self, run_id: RunId
    ) -> Tuple[List[HistoryEvent], RunMetadata]:
        response = self._service.GetRunHistoryAndMetadata(
            GetRunHistoryAndMetadataRequest(run_id=run_id)
        )
        return list(response.events), response.run_metadata

    def get_runs(
        self,
        pipeline_run_id: str,
        list_all: bool,
        project_name: Optional[str] = None,
    ) -> Tuple[List[Run], GetRunsFunction]:
        def get_runs_func() -> List[Run]:
            if pipeline_run_id:
                run = self._service.GetRunById(
                    request=GetRunByIdRequest(run_id=RunId(value=pipeline_run_id))
                ).run
                return [run]
            else:
                request = GetRunsRequest(
                    filters=self._get_run_filters(list_all, project_name=project_name)
                )
                _runs = self._service.GetRuns(request=request).runs
                return list(_runs)

        runs = get_runs_func()
        return runs, get_runs_func

    @staticmethod
    def _get_run_filters(
        list_all: bool, project_name: Optional[str] = None
    ) -> List[RunFilter]:
        filters = []
        if not list_all:
            filters.append(
                RunFilter(
                    status_filter=RunFilter.StatusFilter(
                        run_statuses=[
                            Run.Status.STATUS_RUNNING,
                            Run.Status.STATUS_SCHEDULED,
                        ]
                    )
                )
            )
        if project_name:
            filters.append(
                RunFilter(
                    project_name_filter=RunFilter.ProjectNameFilter(
                        project_name=project_name
                    )
                )
            )
        return filters

    def terminate_run(self, run_id: RunId) -> Run:
        response = self._service.TerminateRun(TerminateRunRequest(run_id=run_id))
        return response.run_id


class UserLogsClient:
    _service: UserLogsAPIStub

    def __init__(
        self,
        config: ClientConfig,
        logger: Logger,
    ):
        self._config = config.user_logs
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

    @contextmanager
    def init(self) -> Iterator["UserLogsClient"]:
        with create_grpc_channel(
            self._config.address,
            self._access_token,
            do_verify_ssl=self._do_verify_ssl,
            logs_file_path=self._logs_file_path,
            options=[
                (
                    "grpc.max_receive_message_length",
                    self._config.max_receive_message_length,
                )
            ],
        ) as channel:
            self._service = UserLogsAPIStub(channel=channel)
            yield self

    def get_pipeline_run_logs(
        self, run_id: uuid.UUID, continuation_token: str
    ) -> Tuple[List[UserLogLine], str]:
        response: GetPipelineRunLogsResponse = self._service.GetPipelineRunLogs(
            request=GetPipelineRunLogsRequest(
                run_id=RunId(value=str(run_id)), continuation_token=continuation_token
            )
        )
        return list(response.log_lines), response.continuation_token


def tar_directory(output_filename: str, source_dir: Path) -> None:
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.sep)
