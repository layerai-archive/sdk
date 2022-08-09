import logging
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import layer
from layer.contracts.asset import AssetType
from layer.executables.packager import FunctionPackageInfo
from layer.executables.runtime import BaseFunctionRuntime
from layer.global_context import current_project_full_name, set_has_shown_update_message


_ProjectId = uuid.UUID


class EntrypointLayerFunctionRuntime(BaseFunctionRuntime):
    def __init__(self, executable_path: Path, project: Optional[str] = None) -> None:
        super().__init__(executable_path)
        self._project = project or _get_current_project_name()
        self._project_id = None
        self._function_metadata: Optional[Dict[str, Any]] = None
        self._asset_name: Optional[str] = None
        self._asset_type: Optional[AssetType] = None
        self._logger = logging.getLogger(__name__)

    def install_packages(self, packages: Sequence[str]) -> None:
        pass

    def initialise(self, package_info: FunctionPackageInfo) -> None:
        self._function_metadata = package_info.metadata
        self._asset_name = package_info.metadata["function"]["output"]["name"]
        self._asset_type = AssetType(
            package_info.metadata["function"]["output"]["type"]
        )
        # required to ensure project exists
        self._layer_init()

        if self._project is None:
            raise LayerFunctionRuntimeError(
                "project not specified and could not be resolved"
            )

    def __call__(self, func: Callable[..., Any]) -> Any:
        if self._asset_type == AssetType.MODEL:
            import traceback

            try:
                from layer.executables.model.entrypoint import (
                    _run as model_train_entrypoint,
                )

                model_train_entrypoint(func)
            except Exception as e:
                print("Error during train:", e)
                traceback.print_exc()
        elif self._asset_type == AssetType.DATASET:
            from layer.executables.dataset.entrypoint import (
                _run as dataset_build_entrypoint,
            )

            dataset_build_entrypoint(func)
        else:
            raise ValueError(f"Unknown asset type: {self._asset_type}")

    def _layer_init(self) -> None:
        set_has_shown_update_message(True)
        import os

        client_auth_url = os.environ["LAYER_CLIENT_AUTH_URL"]
        client_auth_token = os.environ["LAYER_CLIENT_AUTH_TOKEN"]
        project_name = os.environ["LAYER_PROJECT_NAME"]
        layer.login_with_access_token(
            access_token=client_auth_token, url=client_auth_url
        )
        layer.init(project_name=project_name)
        self._project = project_name


def _add_cli_args(parser: ArgumentParser) -> None:
    parser.add_argument("--project", help="project name", required=False)


def _get_current_project_name() -> Optional[str]:
    # try get the project from the global context
    project = current_project_full_name()
    if project is not None:
        return project.project_name

    return None


class LayerFunctionRuntimeError(Exception):
    pass


if __name__ == "__main__":
    EntrypointLayerFunctionRuntime.main(add_cli_args=_add_cli_args)
