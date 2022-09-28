import uuid
from typing import Iterator

import pytest

from test.conftest import _pseudo_random_project_name


@pytest.fixture(autouse=True)
def project_name(request: pytest.FixtureRequest) -> Iterator[str]:
    from layer.contracts.project_full_name import ProjectFullName
    from layer.runs import context

    project_full_name = ProjectFullName(
        account_name="test-acc-from-conftest",
        project_name=_pseudo_random_project_name(request),
    )
    context.reset_to(
        project_full_name=project_full_name,
        project_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        labels=set(),
    )
    yield project_full_name.project_name
    context.reset()
