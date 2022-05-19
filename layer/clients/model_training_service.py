import tempfile
import uuid
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple

import polling  # type: ignore
from google.protobuf.timestamp_pb2 import Timestamp
from layerapi.api.entity.hyperparameter_tuning_pb2 import (
    HyperparameterTuning as PBHyperparameterTuning,
)
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.entity.source_code_environment_pb2 import SourceCodeEnvironment
from layerapi.api.ids_pb2 import HyperparameterTuningId, ModelTrainId
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
from layerapi.api.value.dependency_pb2 import DependencyFile
from layerapi.api.value.hyperparameter_tuning_metadata_pb2 import (
    HyperparameterTuningMetadata,
)
from layerapi.api.value.s3_path_pb2 import S3Path
from layerapi.api.value.source_code_pb2 import RemoteFileLocation, SourceCode

from layer.config import ClientConfig
from layer.contracts.models import (
    BayesianSearch,
    GridSearch,
    HyperparameterTuning,
    ManualSearch,
    Model,
    ParameterType,
    ParameterValue,
    RandomSearch,
)
from layer.exceptions.exceptions import LayerClientException
from layer.utils.file_utils import tar_directory
from layer.utils.grpc import create_grpc_channel
from layer.utils.s3 import S3Util


if TYPE_CHECKING:
    pass


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
