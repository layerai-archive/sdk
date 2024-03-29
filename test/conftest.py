import uuid
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def tmp_dir(tmpdir: Any) -> Path:
    return Path(tmpdir)


def _pseudo_random_project_name(fixture_request: pytest.FixtureRequest) -> str:
    test_name_parametrized: str
    if fixture_request.cls is not None:
        test_name_parametrized = f"{fixture_request.cls.__module__}-{fixture_request.cls.__name__}-{fixture_request.node.name}"
    else:
        test_name_parametrized = (
            f"{fixture_request.module.__name__}-{fixture_request.node.name}"
        )
    test_name_parametrized = test_name_parametrized.replace("[", "-").replace("]", "")
    test_name_parametrized = _cut_off_prefixing_dirs(test_name_parametrized)

    return f"sdk-{test_name_parametrized}-{uuid.uuid4().hex[:8]}"


def _cut_off_prefixing_dirs(module_name: str) -> str:
    # in some environments __module__.__name__ is prefixed with directories tests.e2e. this is useless
    # for debugging and our project names can't have dots, so we cut it off
    return module_name.split(".")[-1]
