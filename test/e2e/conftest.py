import asyncio
import configparser
import logging
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import ddtrace
import pytest
import xdist

import layer
from layer.clients.layer import LayerClient
from layer.config import DEFAULT_LAYER_PATH, ClientConfig, Config, ConfigManager
from layer.contracts.accounts import Account
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from test.e2e.assertion_utils import E2ETestAsserter


logger = logging.getLogger(__name__)

TEST_ASSETS_PATH = Path(__file__).parent / "assets"
TEST_TEMPLATES_PATH = TEST_ASSETS_PATH / "templates"

REFERENCE_E2E_PROJECT_NAME = "e2e-reference-project-titanic"
REFERENCE_E2E_PROJECT_PYTEST_CACHE_KEY = "e2e_reference_project_id"


def pytest_sessionstart(session):
    if os.getenv("CI"):
        ddtrace.patch_all()

    if xdist.is_xdist_worker(session):
        return

    loop = asyncio.get_event_loop()
    org_account: Account = loop.run_until_complete(create_organization_account())

    _write_organization_account_to_test_session_config(org_account)


def pytest_sessionfinish(session):
    """
    WARNING This hook runs BEFORE fixture teardown.

    We use it to set a flag across processes that the global session has finished,
    so we can safely cleanup shared setup. Otherwise this shared setup can be deleted
    while some tests have not started yet due to parallelism.
    """
    if xdist.is_xdist_worker(session):
        return

    with open(_test_session_config_file_path(), "a") as f:
        f.write(
            """[STATE]
is_finished=True
"""
        )


def _cleanup_test_session_config() -> None:
    os.remove(_test_session_config_file_path())


def _is_test_session_finished() -> bool:
    """
    Since we parallelize tests via xdist and fixture tear downs run after
    pytest_sessionfinish hook, we need this inter process communication via
    file
    """
    config = configparser.ConfigParser()
    if not config.read(_test_session_config_file_path()):
        return False

    return config.get("STATE", "is_finished") == "True"


def _write_organization_account_to_test_session_config(org_account):
    with open(_test_session_config_file_path(), "w") as f:
        f.write(
            f"""
[ORGANIZATION_ACCOUNT]
id={str(org_account.id)}
name={org_account.name}
"""
        )

    account_to_verify = _read_organization_account_from_test_session_config()
    assert account_to_verify
    assert account_to_verify.id == org_account.id
    assert account_to_verify.name == org_account.name


def _read_organization_account_from_test_session_config() -> Optional[Account]:
    """
    Returns None if there is no organization account for this session.
    One reason being that it got deleted already by a fixture cleanup.
    """
    config = configparser.ConfigParser()
    if not config.read(_test_session_config_file_path()):
        return None

    return Account(
        id=uuid.UUID(config.get("ORGANIZATION_ACCOUNT", "id")),
        name=config.get("ORGANIZATION_ACCOUNT", "name"),
    )


def _test_session_config_file_path() -> str:
    return DEFAULT_LAYER_PATH / "e2e_test_session.ini"


async def create_organization_account() -> Account:
    config = await ConfigManager().refresh()
    client = LayerClient(config.client, logger)
    org_account_name, display_name = pseudo_random_account_name()
    account = client.account.create_organization_account(org_account_name, display_name)

    # We need a new token with permissions for the new org account
    # TODO LAY-3583 LAY-3652
    await ConfigManager().refresh(force=True)

    return account


def _cleanup_organization_account(client: LayerClient) -> None:
    account = _read_organization_account_from_test_session_config()
    session_finished = _is_test_session_finished()
    if not session_finished or not account:
        # we assume there is no more account to cleanup
        return
    try:
        client.account.delete_account(account_id=account.id)
    except Exception as e:
        print(f"could not delete account: {e}")

    _cleanup_test_session_config()


@pytest.fixture()
def initialized_organization_account(client: LayerClient) -> Iterator[Account]:
    account = _read_organization_account_from_test_session_config()
    assert account

    yield account
    _cleanup_organization_account(client)


@pytest.fixture()
def initialized_organization_project(
    client: LayerClient,
    initialized_organization_account: Account,
    request: Any,
) -> Iterator[Project]:
    account_name = initialized_organization_account.name
    project_name = pseudo_random_project_name(request)
    project = layer.init(f"{account_name}/{project_name}", fabric=Fabric.F_XSMALL.value)

    yield project
    _cleanup_project(client, project)


@pytest.fixture(autouse=True)
def _default_function_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(layer.config, "DEFAULT_FUNC_PATH", tmp_path)


def pseudo_random_project_name(fixture_request: Any) -> str:
    module, test_name = _extract_module_and_test_name(fixture_request)
    module, test_name = _slugify_name(module), _slugify_name(test_name)
    test_name_parametrized = f"{module}-{test_name}"

    random_suffix = str(uuid.uuid4())[:8]

    return f"sdk-e2e-{test_name_parametrized}-{random_suffix}"


def pseudo_random_account_name() -> Tuple[str, str]:
    name_max_length = 50
    # We remove all useless characters to have a valid account name helpful in debugging

    name_prefix = f"sdk-e2e-org-"
    random_suffix = str(uuid.uuid4()).replace("-", "")[
        : name_max_length - len(name_prefix)
    ]
    name = f"{name_prefix}{random_suffix}"
    display_name = f"SDK E2E Test Organization Account - {random_suffix}"

    return name, display_name


def truncate(src: str, max_length: int) -> str:
    if max_length > len(src):
        return src

    return src[:max_length]


def _extract_module_and_test_name(fixture_request: Any) -> Tuple[str, str]:
    if fixture_request.cls is not None:
        return (
            f"{fixture_request.cls.__module__}-{fixture_request.cls.__name__}",
            fixture_request.node.name,
        )
    else:
        return fixture_request.module.__name__, fixture_request.node.name


def _slugify_name(src: str) -> str:
    return _cut_off_prefixing_dirs(src).replace("[", "-").replace("]", "").lower()


def _cut_off_prefixing_dirs(module_name: str) -> str:
    # in some environments __module__.__name__ is prefixed with directories tests.e2e. this is useless
    # for debugging and our project names can't have dots, so we cut it off
    return module_name.split(".")[-1]


@pytest.fixture(autouse=True)
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
async def config() -> Config:
    return await ConfigManager().refresh()


@pytest.fixture()
async def client_config(config: Config) -> ClientConfig:
    return config.client


@pytest.fixture()
def client(client_config: ClientConfig) -> Iterator[LayerClient]:
    yield LayerClient(client_config, logger)


@pytest.fixture()
def initialized_project(client: LayerClient, request: Any) -> Iterator[Project]:
    project_name = pseudo_random_project_name(request)
    project = layer.init(project_name, fabric=Fabric.F_XSMALL.value)

    yield project
    # _cleanup_project(client, project)


def _cleanup_project(client: LayerClient, project: Project):
    project = client.project_service_client.get_project(project.full_name)
    if project:
        print(f"project {project.name} exists, will remove")
        client.project_service_client.remove_project(project.id)


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
