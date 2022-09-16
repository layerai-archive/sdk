import uuid

from layerapi.api.ids_pb2 import ProjectId


def to_project_id(id: uuid.UUID) -> ProjectId:
    return ProjectId(value=str(id))


def from_project_id(id: ProjectId) -> uuid.UUID:
    return uuid.UUID(id.value)
