import atexit
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Sequence, Set

from layer.clients.layer import LayerClient
from layer.contracts.fabrics import Fabric
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.projects import Project, ProjectLoader
from layer.contracts.runs import Run, RunStatus
from layer.runs import context


LAYER_RUN_ID_ENV_VARIABLE = "LAYER_RUN_ID"

logger = logging.Logger(__name__)


class RunInitializer:
    """
    Class responsible for setting up a run
    and creating a project in the Layer Backend if it doesn't exist already
    """

    def __init__(self, layer_client: LayerClient):
        self._layer_client = layer_client

    def setup_project(
        self,
        project_full_name: ProjectFullName,
        run_name: Optional[str] = None,
        run_index: Optional[int] = None,
        labels: Optional[Set[str]] = None,
        project_root_path: Optional[Path] = None,
        fabric: Optional[Fabric] = None,
        pip_packages: Optional[Sequence[str]] = None,
        pip_requirements_file: Optional[str] = None,
    ) -> Project:
        project = get_or_create_project(self._layer_client, project_full_name)
        run = get_or_create_run(
            self._layer_client,
            project_id=project.id,
            run_name=run_name,
            run_index=run_index,
        )

        context.reset_to(
            project_full_name,
            project_id=project.id,
            run_id=run.id,
            labels=labels or set(),
        )
        atexit.register(
            finalize_run, self._layer_client, context.get_run_context(), run
        )
        if fabric:
            context.set_default_fabric(fabric)
        if pip_packages:
            context.set_pip_packages(pip_packages)
        if pip_requirements_file:
            context.set_pip_requirements_file(pip_requirements_file)

        self._update_readme(project_full_name, project_root_path)

        return project

    def _update_readme(
        self, project_full_name: ProjectFullName, project_root_path: Optional[Path]
    ) -> None:
        if not project_root_path:
            # We expect README file in the running directory unless told otherwise
            project_root_path = Path(".")

        project_root_path = Path(os.path.relpath(project_root_path))

        readme_contents = ProjectLoader.load_project_readme(project_root_path)
        if readme_contents:
            self._layer_client.project_service_client.update_project_readme(
                project_full_name=project_full_name,
                readme=readme_contents,
            )


def finalize_run(
    layer_client: LayerClient, run_context: context.RunContext, run: Run
) -> None:
    """
    This is the atexit handler that sets the status of the run as completed
    """
    logger.info("finalizing run: %s", run.id)
    if run_context.error is None:
        layer_client.run_service_client.finish_run(run.id, status=RunStatus.SUCCEEDED)
    else:
        # TODO(volkan) save status info here
        layer_client.run_service_client.finish_run(run.id, status=RunStatus.FAILED)


def get_or_create_project(
    client: LayerClient, project_full_name: ProjectFullName
) -> Project:
    project = client.project_service_client.get_project(project_full_name)
    if project is not None:
        return project

    return client.project_service_client.create_project(
        project_full_name,
    )


def get_or_create_run(
    client: LayerClient,
    project_id: uuid.UUID,
    run_name: Optional[str],
    run_index: Optional[int],
    labels: Optional[Set[str]] = None,
) -> Run:
    run_id_env = os.getenv(LAYER_RUN_ID_ENV_VARIABLE)
    if run_id_env is not None:
        run = client.run_service_client.get_run_by_id(uuid.UUID(run_id_env))
    elif run_index is not None:
        run = client.run_service_client.get_run_by_index(project_id, run_index)
    else:
        run = client.run_service_client.create_run(project_id, run_name=run_name)

    if labels is not None and len(labels) > 0:
        client.label_service_client.add_labels_to_run(run.id, label_names=labels)

    return run
