import sys
import tempfile
import warnings
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Tuple

from layerapi.api.entity.model_train_pb2 import ModelTrain as PBModelTrain
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.entity.source_code_environment_pb2 import SourceCodeEnvironment
from layerapi.api.ids_pb2 import ModelTrainId, ModelVersionId
from layerapi.api.service.modelcatalog.model_catalog_api_pb2 import (
    CompleteModelTrainRequest,
    CreateModelTrainFromVersionIdRequest,
    CreateModelTrainRequest,
    CreateModelVersionRequest,
    CreateModelVersionResponse,
    GetModelTrainRequest,
    GetModelTrainResponse,
    GetModelTrainStorageConfigurationRequest,
    GetModelVersionRequest,
    GetModelVersionResponse,
    LoadModelTrainDataByPathRequest,
    StartModelTrainRequest,
    StoreTrainingMetadataRequest,
    UpdateModelTrainStatusRequest,
)
from layerapi.api.service.modelcatalog.model_catalog_api_pb2_grpc import (
    ModelCatalogAPIStub,
)
from layerapi.api.value.dependency_pb2 import DependencyFile
from layerapi.api.value.s3_path_pb2 import S3Path
from layerapi.api.value.sha256_pb2 import Sha256
from layerapi.api.value.source_code_pb2 import RemoteFileLocation, SourceCode

from layer.cache.cache import Cache
from layer.config import ClientConfig
from layer.contracts.asset import AssetPath
from layer.contracts.models import Model, ModelObject, TrainStorageConfiguration
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.tracker import ResourceTransferState
from layer.exceptions.exceptions import LayerClientException
from layer.flavors.base import ModelRuntimeObjects
from layer.flavors.utils import get_flavor_for_proto
from layer.tracker.ui_progress_tracker import UIRunProgressTracker
from layer.utils.grpc import create_grpc_channel
from layer.utils.s3 import S3Util


if TYPE_CHECKING:
    from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PBModelFlavor


class ModelCatalogClient:
    _service: ModelCatalogAPIStub

    def __init__(
        self, config: ClientConfig, logger: Logger, cache_dir: Optional[Path] = None
    ):
        self._config = config.model_catalog
        self._s3_endpoint_url = config.s3.endpoint_url
        self._logger = logger
        self._access_token = config.access_token
        self._do_verify_ssl = config.grpc_do_verify_ssl
        self._logs_file_path = config.logs_file_path

        self._cache = Cache(cache_dir).initialise()

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

    def create_model_version(
        self,
        asset_path: AssetPath,
        description: str,
        source_code_hash: str,
        fabric: str,
    ) -> CreateModelVersionResponse:
        """
        Given a model metadata it makes a request to the backend
        and creates a corresponding entity.
        :return: the created model version entity
        """
        self._logger.debug(f"Creating model version for the model: {asset_path}")
        response = self._service.CreateModelVersion(
            CreateModelVersionRequest(
                model_path=asset_path.path(),
                description=description,
                training_files_hash=Sha256(value=source_code_hash),
                should_create_initial_train=True,
                fabric=fabric,
            ),
        )
        self._logger.debug(f"CreateModelVersionResponse: {str(response)}")
        return response

    def store_training_metadata(
        self,
        asset_name: str,
        description: str,
        entrypoint: str,
        environment: str,
        s3_path: S3Path,
        version: ModelVersion,
        fabric: str,
    ) -> None:
        language_version = _language_version()
        request: StoreTrainingMetadataRequest = StoreTrainingMetadataRequest(
            model_version_id=version.id,
            name=asset_name,
            description=description,
            source_code_env=SourceCodeEnvironment(
                source_code=SourceCode(
                    remote_file_location=RemoteFileLocation(
                        name=entrypoint,
                        location=f"s3://{s3_path.bucket}/{s3_path.key}{asset_name}.tgz",
                    ),
                    language=SourceCode.Language.LANGUAGE_PYTHON,
                    language_version=SourceCode.LanguageVersion(
                        major=language_version[0],
                        minor=language_version[1],
                        micro=language_version[2],
                    ),
                ),
                dependency_file=DependencyFile(
                    name=environment,
                    location=environment,
                ),
            ),
            entrypoint=entrypoint,
            fabric=fabric,
        )
        self._logger.debug(f"StoreTrainingMetadataRequest request: {str(request)}")
        response = self._service.StoreTrainingMetadata(request)
        self._logger.debug(f"StoreTrainingMetadata response: {str(response)}")

    def load_model_by_path(self, path: str) -> Model:
        load_response = self._service.LoadModelTrainDataByPath(
            LoadModelTrainDataByPathRequest(path=path),
        )

        flavor = get_flavor_for_proto(load_response.flavor)
        if flavor is None:
            raise LayerClientException(
                f"Unexpected model flavor {type(load_response.flavor)}"
            )
        return Model(
            asset_path=path,
            id=load_response.id.value,
            flavor=flavor,
            storage_config=TrainStorageConfiguration(
                train_id=load_response.id,
                s3_path=load_response.s3_path,
                credentials=load_response.credentials,
            ),
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

    def load_model_runtime_objects(
        self,
        model: Model,
        state: ResourceTransferState,
        no_cache: bool = False,
    ) -> ModelRuntimeObjects:
        """
        Loads a model artifact from the model catalog

        :param model: the model
        :return: the model artifact
        """
        self._logger.debug(f"Loading model {model.path}")
        try:
            model_cache_dir = self._cache.get_path_entry(str(model.id))
            if no_cache or model_cache_dir is None:
                with tempfile.TemporaryDirectory() as tmp:
                    local_path = Path(tmp) / "model"

                    S3Util.download_dir(
                        local_dir=local_path,
                        credentials=model.storage_config.credentials,
                        s3_path=model.storage_config.s3_path,
                        endpoint_url=self._s3_endpoint_url,
                        state=state,
                    )
                    if no_cache:
                        return self._load_model_runtime_objects(model, local_path)
                    model_cache_dir = self._cache.put_path_entry(
                        str(model.id), local_path
                    )

            assert model_cache_dir is not None
            return self._load_model_runtime_objects(model, model_cache_dir)
        except Exception as ex:
            raise LayerClientException(f"Error while loading model, {ex}")

    def _load_model_runtime_objects(
        self, model: Model, model_dir: Path
    ) -> ModelRuntimeObjects:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return model.flavor.load_model_from_directory(model_dir)

    def save_model_object(
        self,
        model: Model,
        model_object: ModelObject,
        tracker: UIRunProgressTracker,
    ) -> ModelObject:
        self._logger.debug(f"Storing given model {model_object} for {model.path}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_path = Path(tmp) / "model"
                model.flavor.save_model_to_directory(model_object, local_path)
                state = ResourceTransferState()
                tracker.mark_model_saving_result(model.name, state)
                S3Util.upload_dir(
                    local_dir=local_path,
                    credentials=model.storage_config.credentials,
                    s3_path=model.storage_config.s3_path,
                    endpoint_url=self._s3_endpoint_url,
                    state=state,
                )
        except Exception as ex:
            raise LayerClientException(f"Error while storing model, {ex}")
        self._logger.debug(f"User model {model.path} saved successfully")

        return model_object

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

    def create_model_train(
        self,
        name: str,
        project_full_name: ProjectFullName,
        version: Optional[str],
    ) -> ModelTrainId:
        response = self._service.CreateModelTrain(
            CreateModelTrainRequest(
                model_name=name,
                model_version="" if version is None else version,
                project_full_name=project_full_name.path,
            ),
        )
        return response.id

    def complete_model_train(
        self,
        train_id: ModelTrainId,
        flavor: Optional["PBModelFlavor.ValueType"],
    ) -> None:
        self._service.CompleteModelTrain(
            CompleteModelTrainRequest(id=train_id, flavor=flavor),
        )

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
        self, train_id: ModelTrainId, train_status: PBModelTrainStatus
    ) -> None:
        self._service.UpdateModelTrainStatus(
            UpdateModelTrainStatusRequest(
                model_train_id=train_id, train_status=train_status
            )
        )


def _language_version() -> Tuple[int, int, int]:
    return sys.version_info.major, sys.version_info.minor, sys.version_info.micro
