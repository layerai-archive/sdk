import tempfile
import uuid
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Iterator

import polling  # type: ignore
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.ids_pb2 import ModelTrainId
from layerapi.api.service.modeltraining.model_training_api_pb2 import (
    GetModelTrainStatusRequest,
    GetSourceCodeUploadCredentialsRequest,
    GetSourceCodeUploadCredentialsResponse,
    StartModelTrainingRequest,
)
from layerapi.api.service.modeltraining.model_training_api_pb2_grpc import (
    ModelTrainingAPIStub,
)
from layerapi.api.value.s3_path_pb2 import S3Path

from layer.config import ClientConfig
from layer.exceptions.exceptions import LayerClientException
from layer.utils.file_utils import tar_directory
from layer.utils.grpc import create_grpc_channel
from layer.utils.s3 import S3Util


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

    def upload_training_files(
        self, asset_name: str, function_home_dir: Path, model_version_id: uuid.UUID
    ) -> S3Path:
        response = self.get_source_code_upload_credentials(model_version_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_directory(f"{tmp_dir}/{asset_name}.tgz", function_home_dir)
            S3Util.upload_dir(
                Path(tmp_dir),
                response.credentials,
                response.s3_path,
                endpoint_url=self._s3_endpoint_url,
            )

        return response.s3_path

    def get_source_code_upload_credentials(
        self, model_version_id: uuid.UUID
    ) -> GetSourceCodeUploadCredentialsResponse:
        return self._service.GetSourceCodeUploadCredentials(
            GetSourceCodeUploadCredentialsRequest(source_name=str(model_version_id))
        )

    def train_model(
        self,
        version: ModelVersion,
    ) -> uuid.UUID:
        response: GetSourceCodeUploadCredentialsResponse = (
            self.get_source_code_upload_credentials(uuid.UUID(version.id.value))
        )
        self._logger.debug(
            f"GetSourceCodeUploadCredentialsResponse response: {str(response)}"
        )
        return self._execute_regular_train(train_id=version.latest_train_id)

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
