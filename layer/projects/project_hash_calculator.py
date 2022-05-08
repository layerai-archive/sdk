import hashlib
from typing import List

from layer.definitions import Definition


def calculate_project_hash_by_definitions(
    entity_definitions: List[Definition],
) -> str:
    project_files_hash = hashlib.sha256()

    for entity_definition in entity_definitions:
        project_files_hash.update(entity_definition.get_pickled_function())

    return project_files_hash.hexdigest()
