import contextlib
import logging
import os
import tempfile
import uuid
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from layerapi.api.ids_pb2 import ProjectId

import layer
from layer.clients.data_catalog import DataCatalogClient
from layer.clients.model_catalog import ModelCatalogClient
from layer.clients.project_service import ProjectServiceClient
from layer.config.config import ClientConfig
from layer.config.config_manager import ConfigManager
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.fabrics import Fabric
from layer.contracts.models import TrainStorageConfiguration
from layer.executables.packager import FunctionPackageInfo
from layer.executables.runtime import BaseFunctionRuntime
from layer.flavors.utils import get_flavor_for_model
from layer.global_context import current_project_full_name, set_has_shown_update_message


_ProjectId = uuid.UUID

_ENV_LAYER_API_URL = "LAYER_API_URL"
_ENV_LAYER_API_KEY = "LAYER_API_KEY"


class LayerFunctionRuntime(BaseFunctionRuntime):
    def __init__(
        self,
        executable_path: Path,
        project: str,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(executable_path)
        self._project = project
        self._logger = logging.getLogger("layer.runtime")
        self._api_url: Optional[str] = api_url or os.environ.get(_ENV_LAYER_API_URL)
        self._api_key: Optional[str] = api_key or os.environ.get(_ENV_LAYER_API_KEY)
        self._package_info: Optional[FunctionPackageInfo] = None
        self._project_id: Optional[_ProjectId] = None
        self._client_config: Optional[ClientConfig] = None

    def initialise(self, package_info: FunctionPackageInfo) -> None:
        self._package_info = package_info

        # setup client config
        if self._api_url and self._api_key:
            layer.login_with_api_key(self._api_key, url=self._api_url)

        self._client_config = ConfigManager().load().client

        # required to ensure project exists
        self._layer_init()
        self._project_id = self._get_project_id()

    def __call__(self, func: Callable[..., Any]) -> Any:
        function_output = _get_function_output(self._package_info.metadata)  # type: ignore
        if function_output.is_dataset:
            self._create_dataset(function_output.name, func)
        elif function_output.is_model:
            self._create_model(function_output.name, func)
        else:
            raise LayerFunctionRuntimeError(
                f"missing function output type: {function_output.type}"
            )

    def _layer_init(self) -> None:
        set_has_shown_update_message(shown=True)
        layer.init(project_name=self._project)

    def _get_project_id(self) -> _ProjectId:
        projects = ProjectServiceClient.create(self._client_config)  # type: ignore
        project_name = (
            current_project_full_name()
        )  # project is always set in the context by layer.init
        project = projects.get_project(project_name)  # type: ignore
        if project is None:
            raise LayerFunctionRuntimeError(f"project {self._project} does not exist")
        return project.id

    def _create_dataset(self, name: str, func: Callable[..., Any]) -> None:
        data_catalog = DataCatalogClient.create(self._client_config, self._logger)  # type: ignore
        display_fabric = Fabric.F_LOCAL.value  # the fabric to display in the UI
        asset_path = AssetPath(name, AssetType.DATASET)
        data_catalog.add_dataset(
            project_id=self._project_id,  # type: ignore
            asset_path=asset_path,
            description="",
            fabric=display_fabric,
            func_source="",
            entrypoint="",
            environment="",
        )
        build_response = data_catalog.initiate_build(
            ProjectId(value=str(self._project_id)), name, display_fabric
        )
        data_catalog.store_dataset(func(), uuid.UUID(build_response.id.value))

    def _create_model(self, name: str, func: Callable[..., Any]) -> None:
        model = func()
        model_flavor = get_flavor_for_model(model)
        if model_flavor is None:
            raise LayerFunctionRuntimeError("unsupported model flavor")
        with tempfile.TemporaryDirectory() as model_dir:
            model_flavor.save_model_to_directory(model, Path(model_dir))
            model_catalog = ModelCatalogClient.create(self._client_config, self._logger)  # type: ignore
            project = current_project_full_name()
            asset_path = AssetPath(
                name,
                AssetType.MODEL,
                org_name=project.account_name,  # type: ignore
                project_name=project.project_name,  # type: ignore
            )
            source_digest = _get_function_source_digest(self._package_info.metadata)  # type: ignore
            display_fabric = Fabric.F_LOCAL.value  # the fabric to display in the UI
            model_version = model_catalog.create_model_version(
                asset_path=asset_path,
                description="",
                source_code_hash=source_digest,
                fabric=display_fabric,
            )
            model_train_id = model_catalog.create_model_train_from_version_id(
                model_version.model_version.id
            )
            model_catalog.start_model_train(model_train_id)
            model_catalog.complete_model_train(
                model_train_id, model_flavor.PROTO_FLAVOR
            )
            model_storage_config = model_catalog.get_model_train_storage_configuration(
                model_train_id
            )
            _upload_model_artifacts(Path(model_dir), model_storage_config)
            model_catalog.store_training_metadata(
                asset_name=asset_path.asset_name,
                description="",
                entrypoint="main.py",  # not used
                environment="",
                s3_path=model_storage_config.s3_path,
                version=model_version.model_version,
                fabric=display_fabric,
            )


def _add_cli_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        help="project name",
        required=True,
    )
    parser.add_argument(
        "--api-url",
        help=f"api url. Optionally, {_ENV_LAYER_API_URL} environment variable is also supported",
        required=False,
    )
    parser.add_argument(
        "--api-key",
        help=f"api key. Optionally, {_ENV_LAYER_API_KEY} environment variable is also supported",
        required=False,
    )


@dataclass(frozen=True)
class _FunctionOutput:
    type: str
    name: str

    @property
    def is_dataset(self) -> bool:
        return self.type == AssetType.DATASET.value

    @property
    def is_model(self) -> bool:
        return self.type == AssetType.MODEL.value


def _get_function_output(metadata: Dict[str, Any]) -> _FunctionOutput:
    output = metadata.get("function", {}).get("output")
    if output is None:
        raise LayerFunctionRuntimeError("function metadata missing output")
    type = output.get("type")
    if type is None:
        raise LayerFunctionRuntimeError("function metadata missing output type")
    name = output.get("name")
    if name is None:
        raise LayerFunctionRuntimeError("function metadata missing output name")
    return _FunctionOutput(type, name)


def _get_function_source_digest(metadata: Dict[str, Any]) -> str:
    source_digest = metadata.get("function", {}).get("source", {}).get("digest")
    if source_digest is None:
        raise LayerFunctionRuntimeError("function metadata missing source digest")
    return source_digest


def _upload_model_artifacts(model_dir: Path, config: TrainStorageConfiguration) -> None:
    import boto3

    s3_config = {
        "aws_access_key_id": config.credentials.access_key_id,
        "aws_secret_access_key": config.credentials.secret_access_key,
        "aws_session_token": config.credentials.session_token,
    }

    with contextlib.closing(boto3.client("s3", **s3_config)) as s3_client:  # type: ignore
        for (root, _, filenames) in os.walk(model_dir, followlinks=False):
            for file_name in filenames:
                abs_path = Path(root, file_name)
                relative_path = abs_path.relative_to(model_dir)
                file_key = f"{config.s3_path.key}{relative_path}"
                with open(abs_path, "rb") as src:
                    s3_client.upload_fileobj(src, config.s3_path.bucket, file_key)


class LayerFunctionRuntimeError(Exception):
    pass


if __name__ == "__main__":
    LayerFunctionRuntime.main(add_cli_args=_add_cli_args)
