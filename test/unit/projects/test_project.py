import uuid

import pytest

from layer.contracts.projects import Project
from layer.exceptions.exceptions import ProjectException


class TestProject:
    def test_project_id_property_throws_when_absent(self) -> None:
        with pytest.raises(ProjectException, match="project has no id defined"):
            project = Project(name="test")

            _ = project.id

    def test_project_id_property_returns_when_present(self) -> None:
        project = Project(name="test")

        new_id = uuid.uuid4()
        project = project.with_id(new_id)

        assert new_id == project.id
