import os
import sys
import uuid
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Tuple

import layerapi.api.entity.model_version_pb2
from layerapi.api.entity.model_pb2 import Model as PBModel
from layerapi.api.entity.model_train_pb2 import ModelTrain as PBModelTrain
from layerapi.api.entity.model_train_status_pb2 import (
    ModelTrainStatus as PBModelTrainStatus,
)
from layerapi.api.entity.model_version_pb2 import ModelVersion
from layerapi.api.ids_pb2 import ModelTrainId, ModelVersionId
from layerapi.api.service.modelcatalog.model_catalog_api_pb2 import (
    CreateModelVersionResponse,
)
from layerapi.api.service.modelcatalog.model_catalog_api_pb2_grpc import (
    ModelCatalogAPIStub,
)
from layerapi.api.value.s3_path_pb2 import S3Path

from layer.cache.cache import Cache
from layer.clients.local.asset_db import AssetDB, AssetType
from layer.config import ClientConfig
from layer.contracts.models import Model, ModelObject, TrainStorageConfiguration
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.runs import FunctionDefinition
from layer.contracts.tracker import ResourceTransferState
from layer.exceptions.exceptions import LayerClientException
from layer.flavors import ScikitLearnModelFlavor
from layer.flavors.base import ModelRuntimeObjects
from layer.tracker.progress_tracker import RunProgressTracker


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
    def init(self):
        return self

    def create_model_version(
        self,
        project_full_name: ProjectFullName,
        model: FunctionDefinition,
        is_local: bool,
    ) -> CreateModelVersionResponse:
        version = uuid.uuid4().hex
        db = AssetDB(project_full_name.project_name)
        asset = db.get_asset(model.asset_name, version, AssetType.MODEL)
        asset.set("name", model.asset_name)
        return CreateModelVersionResponse(
            model_version=layerapi.api.entity.model_version_pb2.ModelVersion(
                id=ModelVersionId(value=version), name=model.asset_name
            ),
            should_upload_training_files=False,
        )

    def store_training_metadata(
        self,
        model: FunctionDefinition,
        s3_path: S3Path,
        version: ModelVersion,
    ) -> None:
        raise Exception("Not implemented yet!")

    def _get_asset_by_path(self, path: str):
        _, project_name, _, model_identifier = path.split("/")
        model_params = model_identifier.split(":")

        model_name = model_params[0]
        version = model_params[1] if len(model_params) > 1 else None

        asset = AssetDB.get_asset_by_name_and_version(
            project_name, model_name, version, AssetType.MODEL
        )
        return asset

    def load_model_by_path(self, path: str) -> Model:
        asset = self._get_asset_by_path(path)
        model = Model(id=uuid.UUID(asset.get("id")), asset_path=path)
        return model

    def create_model_train_from_version_id(
        self,
        version_id: ModelVersionId,
    ) -> ModelTrainId:
        self._logger.debug("Creating model train:", version_id.value)
        return ModelTrainId(value=version_id.value)

    def load_model_runtime_objects(
        self,
        model: Model,
        state: ResourceTransferState,
        no_cache: bool = False,
    ) -> ModelRuntimeObjects:
        asset = self._get_asset_by_path(model.path)
        model_path = Path(asset.path) / "model"
        runtimeobject = ScikitLearnModelFlavor().load_model_from_directory(
            directory=model_path.absolute()
        )
        return runtimeobject

    def _load_model_runtime_objects(
        self, model: Model, model_dir: Path
    ) -> ModelRuntimeObjects:
        pass

    def save_model_object(
        self,
        model: Model,
        model_object: ModelObject,
        tracker: RunProgressTracker,
    ) -> ModelObject:
        try:
            from layer.clients.local.layer_local_client import LayerLocalClient

            project_name = LayerLocalClient.get_project_name()
            asset = AssetDB.get_asset_by_id(project_name, model.id.hex, AssetType.MODEL)
            model_path = asset.path / "model"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model.flavor.save_model_to_directory(model_object, model_path)
            tracker.mark_model_saved(model.name)
        except Exception as ex:
            raise LayerClientException(f"Error while storing model, {ex}")
        self._logger.debug(f"User model {model.path} saved successfully")

        return model_object

    def get_model_train_storage_configuration(
        self,
        train_id: ModelTrainId,
    ) -> TrainStorageConfiguration:
        pass

    def start_model_train(
        self,
        train_id: ModelTrainId,
    ) -> TrainStorageConfiguration:
        pass

    def create_model_train(
        self,
        name: str,
        project_full_name: ProjectFullName,
        version: Optional[str],
    ) -> ModelTrainId:
        raise Exception("Not implemented yet!")

    def complete_model_train(
        self,
        train_id: ModelTrainId,
        flavor: Optional["PBModelFlavor.ValueType"],
    ) -> None:
        pass

    def get_model_by_path(self, model_path: str) -> PBModel:
        raise Exception("Not implemented yet!")

    def get_model_train(self, train_id: ModelTrainId) -> PBModelTrain:
        return PBModelTrain(
            index=1, model_version_id=ModelVersionId(value=uuid.uuid4().hex)
        )

    def get_model_version(self, version_id: ModelVersionId) -> ModelVersion:
        return ModelVersionId(value=uuid.uuid4().hex)

    def update_model_train_status(
        self, train_id: ModelTrainId, train_status: PBModelTrainStatus
    ) -> None:
        return None


def _language_version() -> Tuple[int, int, int]:
    return sys.version_info.major, sys.version_info.minor, sys.version_info.micro
