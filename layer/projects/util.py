from uuid import UUID

from layer import current_project_name
from layer.common import LayerClient
from layer.exceptions.exceptions import ProjectInitializationException


def verify_project_exists_and_retrieve_project_id(
    client: LayerClient, project_name: str
) -> UUID:
    project_uuid = client.project_service_client.get_project_id_and_org_id(
        project_name
    ).project_id
    if not project_uuid:
        raise ProjectInitializationException(
            f"Project with the name {project_name} does not exist."
        )
    return project_uuid


def get_current_project_name() -> str:
    current_project_name_ = current_project_name()
    if not current_project_name_:
        raise ProjectInitializationException(
            "Please specify the current project name globally with `layer.init('project-name')`"
        )
    return current_project_name_
