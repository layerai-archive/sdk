import os
from logging import Logger
from pathlib import Path
from typing import Any, List, Optional

from layer import global_context
from layer.async_utils import asyncio_run_in_thread
from layer.common import LayerClient
from layer.config import ConfigManager
from layer.contracts.fabrics import Fabric
from layer.global_context import (
    set_default_fabric,
    set_pip_packages,
    set_pip_requirements_file,
)
from layer.projects.project import Project, ProjectLoader, get_or_create_remote_project


class InitProjectRunner:
    """Class responsible for setting up a user's project locally
    and creating a project in the Layer Backend if it doesn't exist already
    """

    def __init__(
        self,
        project_name: str,
        project_root_path: Optional[str] = None,
        logger: Optional[Logger] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        self._project_name = project_name
        self._project_root_path = project_root_path
        self._config_manager = (
            config_manager if config_manager is not None else ConfigManager()
        )
        self._logger = (
            logger if logger is not None else Logger(__name__)  # not sure about this
        )

    def _ensure_user_logged_in(self) -> Any:
        login_config = asyncio_run_in_thread(self._config_manager.refresh())
        return login_config

    def _update_readme(self, project: Project, layer_client: LayerClient) -> None:
        readme_discover_path = self._project_root_path
        if not readme_discover_path:
            # We expect README file in the running directory unless told otherwise
            readme_discover_path = "./"
        project_root_path = Path(os.path.relpath(readme_discover_path))

        readme_contents = ProjectLoader.load_project_readme(project_root_path)
        if readme_contents:
            layer_client.project_service_client.update_project_readme(
                project_name=project.name, readme=readme_contents
            )

    def setup_project(
        self,
        layer_client: Optional[LayerClient] = None,
        fabric: Optional[Fabric] = None,
        pip_packages: Optional[List[str]] = None,
        pip_requirements_file: Optional[str] = None,
    ) -> Project:
        if not layer_client:
            login_config = self._ensure_user_logged_in()
            layer_client = LayerClient(login_config.client, self._logger)

        with layer_client.init() as initialized_client:
            project = get_or_create_remote_project(
                initialized_client, Project(name=self._project_name)
            )

            self._update_readme(project, layer_client)

        global_context.reset_to(self._project_name)
        if fabric:
            set_default_fabric(fabric)
        if pip_packages:
            set_pip_packages(pip_packages)
        if pip_requirements_file:
            set_pip_requirements_file(pip_requirements_file)

        return project
