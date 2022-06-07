import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import ddtrace
import pytest

import layer
from layer.clients.layer import LayerClient
from layer.config import ClientConfig, Config, ConfigManager
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from layer.projects.project_runner import ProjectRunner
from layer.tracker.remote_execution_project_progress_tracker import (
    RemoteExecutionRunProgressTracker,
)
from test.e2e.assertion_utils import E2ETestAsserter


logger = logging.getLogger(__name__)

TEST_ASSETS_PATH = Path(__file__).parent / "assets"
TEST_TEMPLATES_PATH = TEST_ASSETS_PATH / "templates"

REFERENCE_E2E_PROJECT_NAME = "e2e-reference-project-titanic"
REFERENCE_E2E_PROJECT_PYTEST_CACHE_KEY = "e2e_reference_project_id"


def pytest_sessionstart(session):
    if os.getenv("CI"):
        ddtrace.patch_all()


@pytest.fixture(autouse=True)
def _default_function_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(layer.config, "DEFAULT_FUNC_PATH", tmp_path)


def pseudo_random_project_name(fixture_request: Any) -> str:
    test_name_parametrized: str
    if fixture_request.cls is not None:
        test_name_parametrized = f"{fixture_request.cls.__module__}-{fixture_request.cls.__name__}-{fixture_request.node.name}"
    else:
        test_name_parametrized = (
            f"{fixture_request.module.__name__}-{fixture_request.node.name}"
        )
    test_name_parametrized = test_name_parametrized.replace("[", "-").replace("]", "")
    test_name_parametrized = _cut_off_prefixing_dirs(test_name_parametrized)

    return f"sdk-e2e-{test_name_parametrized}-{str(uuid.uuid4())[:8]}"


def _cut_off_prefixing_dirs(module_name: str) -> str:
    # in some environments __module__.__name__ is prefixed with directories tests.e2e. this is useless
    # for debugging and our project names can't have dots, so we cut it off
    return module_name.split(".")[-1]


@pytest.fixture()
def _write_stdout_stderr_to_file(capsys, request) -> Iterator[Any]:
    """
    Since we parallelize pytest tests via xdist, stdout and stderr are swallowed.
    With this fixture we make stdout and stderr available with the test results.

    Args:
        capsys: Core pytest fixture to capture stdout and stderr
        request: Core pytest fixture to access the test request

    Returns:
        None
    """
    yield

    # Run after the test has finished
    if "SDK_E2E_TESTS_LOGS_DIR" in os.environ:
        captured = capsys.readouterr()

        test_name_parametrized: str
        if request.cls is not None:
            test_name_parametrized = (
                f"{request.cls.__module__}.{request.cls.__name__}.{request.node.name}"
            )
        else:
            test_name_parametrized = f"{request.module.__name__}.{request.node.name}"

        test_results_directory = os.environ.get("SDK_E2E_TESTS_LOGS_DIR")

        if len(captured.out) > 0:
            f_path = Path(test_results_directory).joinpath(
                f"{test_name_parametrized}.stdout.log"
            )
            f_path.parent.mkdir(parents=True, exist_ok=True)
            with open(f_path, "w") as f:
                f.write(captured.out)

        if len(captured.err) > 0:
            f_path = Path(test_results_directory).joinpath(
                f"{test_name_parametrized}.stderr.log"
            )
            f_path.parent.mkdir(parents=False, exist_ok=True)
            with open(f_path, "w") as f:
                f.write(captured.err)


@pytest.fixture()
async def config(_write_stdout_stderr_to_file: Any) -> Config:
    return await ConfigManager().refresh()


@pytest.fixture()
async def client_config(config: Config) -> ClientConfig:
    return config.client


@pytest.fixture()
def client(client_config: ClientConfig) -> Iterator[LayerClient]:
    with LayerClient(client_config, logger).init() as client:
        yield client


@pytest.fixture()
def initialized_project(client: LayerClient, request: Any) -> Iterator[Project]:
    project_name = pseudo_random_project_name(request)
    project = layer.init(project_name, fabric=Fabric.F_XSMALL.value)

    yield project
    _cleanup_project(client, project)


def _cleanup_project(client: LayerClient, project: Project):
    project_id = client.project_service_client.get_project_id_and_org_id(
        project.full_name
    ).project_id
    if project_id:
        print(f"project {project.name} exists, will remove")
        client.project_service_client.remove_project(project_id)


@pytest.fixture()
def project_runner(config: Config) -> ProjectRunner:
    return ProjectRunner(
        config, project_progress_tracker_factory=RemoteExecutionRunProgressTracker
    )


@pytest.fixture()
def asserter(client: LayerClient) -> E2ETestAsserter:
    return E2ETestAsserter(client=client)


@pytest.fixture()
async def guest_context(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., Any]:
    guest_config_path = tmp_path / "guest_config.json"
    await ConfigManager(guest_config_path).login_as_guest(config.url)

    @contextmanager
    def manager() -> Iterator[None]:
        with monkeypatch.context() as m:
            m.setattr(
                layer.config.ConfigManager,
                "_get_default_path",
                lambda x: guest_config_path,
            )
            yield

    return manager
