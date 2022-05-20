from contextlib import contextmanager
from logging import Logger
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

from layerapi.api.entity.model_pb2 import Model as PBModel
from layerapi.api.entity.model_train_pb2 import ModelTrain as PBModelTrain
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.entity.source_code_environment_pb2 import SourceCodeEnvironment
from layerapi.api.ids_pb2 import HyperparameterTuningId, ModelTrainId, ModelVersionId
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
from layerapi.api.value.dependency_pb2 import DependencyFile
from layerapi.api.value.s3_path_pb2 import S3Path
from layerapi.api.value.sha256_pb2 import Sha256
from layerapi.api.value.source_code_pb2 import RemoteFileLocation, SourceCode

from layer.clients.model_service import MLModelService
from layer.config import ClientConfig
from layer.contracts.models import Model, TrainStorageConfiguration
from layer.contracts.runs import ResourceTransferState
from layer.exceptions.exceptions import LayerClientException
from layer.flavors import ModelFlavor
from layer.flavors.model_definition import ModelDefinition
from layer.tracker.project_progress_tracker import ProjectProgressTracker
from layer.utils.grpc import create_grpc_channel


if TYPE_CHECKING:
    from layer.contracts.models import TrainedModelObject


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
            PROTO_FLAVOR=load_response.flavor,
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
        trained_model_obj: "TrainedModelObject",
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> "TrainedModelObject":
        if not tracker:
            tracker = ProjectProgressTracker()
        flavor = self._ml_model_service.get_model_flavor_from_proto(
            model_definition.PROTO_FLAVOR
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

    def infer_flavor(
        self, model_obj: "TrainedModelObject"
    ) -> "ModelVersion.ModelFlavor":
        flavor: ModelFlavor = self._ml_model_service.get_model_flavor(
            model_obj,
            logger=self._logger,
        )
        return flavor.PROTO_FLAVOR

    def complete_model_train(
        self, train_id: ModelTrainId, flavor: Optional["ModelVersion.ModelFlavor"]
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
        self, train_id: ModelTrainId, train_status: PBModelTrainStatus
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
