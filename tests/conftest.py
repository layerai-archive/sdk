from pathlib import Path
from typing import Any

import pytest

from layer import global_context


@pytest.fixture()
def tmp_dir(tmpdir: Any) -> Path:
    return Path(tmpdir)


@pytest.fixture(autouse=True)
def test_project_name() -> str:
    test_project_name = "the-test-project"
    global_context.reset_to(test_project_name)
    yield test_project_name
    global_context.reset_to(None)
