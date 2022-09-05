import asyncio
import configparser
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import ddtrace
import pytest
import xdist
from filelock import FileLock

import layer
from layer.clients.layer import LayerClient
from layer.config import DEFAULT_LAYER_PATH, ClientConfig, Config, ConfigManager
from layer.contracts.accounts import Account
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from layer.exceptions.exceptions import LayerClientResourceNotFoundException
from test.e2e.assertion_utils import E2ETestAsserter


logger = logging.getLogger(__name__)

TEST_SESSION_CONFIG_TMP_PATH = DEFAULT_LAYER_PATH / "e2e_test_session.ini"

TEST_ORG_ACCOUNT_NAME_PREFIX = "sdk-e2e-org-"


def pytest_sessionstart(session):
    """
    We use this hook to create a single new organization account to be used in all
    organization e2e tests.

    We clean it up (teardown) using `_track_session_counts_and_cleanup` fixture.
    """
    if os.getenv("CI"):
        ddtrace.patch_all()

    if xdist.is_xdist_worker(session):
        return

    org_account = _read_organization_account_from_test_session_config()
    if org_account:
        # This is unexpected, so we try to cleanup or fail silently
        _cleanup_organization_account()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(delete_old_org_accounts())
    org_account: Account = loop.run_until_complete(create_organization_account())

    _write_organization_account_to_test_session_config(org_account)


@pytest.fixture(scope="session", autouse=True)
def _track_session_counts_and_cleanup() -> Iterator:
    """
    This will run for every pytest session.
    Since we optionally use xdist to parallelize test runs,
    we need extra logic to track parallel pytest sessions and do teardown once there are no more.
    """
    _increment_session_count()
    yield
    open_sessions = _decrement_session_count()
    if open_sessions < 1:
        _cleanup_organization_account()
        _cleanup_test_session_config()
        if open_sessions < 0:
            raise Exception(
                f"unexpected negative number of open sessions {open_sessions}"
            )


def _cleanup_test_session_config() -> None:
    config_path = TEST_SESSION_CONFIG_TMP_PATH
    if os.path.exists(config_path):
        os.remove(config_path)


def _increment_session_count() -> int:
    with FileLock(str(TEST_SESSION_CONFIG_TMP_PATH) + ".lock"):
        config = configparser.ConfigParser()
        config.read(TEST_SESSION_CONFIG_TMP_PATH)
        current_counter = 0
        if "PYTEST_SESSIONS" in config.sections():
            current_counter = int(config.get("PYTEST_SESSIONS", "count", fallback=0))
        else:
            config["PYTEST_SESSIONS"] = {}
        current_counter += 1
        config["PYTEST_SESSIONS"]["count"] = str(current_counter)

        with open(TEST_SESSION_CONFIG_TMP_PATH, "w") as configfile:
            config.write(configfile)

        return current_counter


def _decrement_session_count() -> int:
    """
    Expects to be called after at least one increment
    """
    with FileLock(str(TEST_SESSION_CONFIG_TMP_PATH) + ".lock"):
        config = configparser.ConfigParser()
        config.read(TEST_SESSION_CONFIG_TMP_PATH)
        current_counter = int(config.get("PYTEST_SESSIONS", "count"))
        current_counter -= 1
        config["PYTEST_SESSIONS"]["count"] = str(current_counter)

        with open(TEST_SESSION_CONFIG_TMP_PATH, "w") as configfile:
            config.write(configfile)

    return current_counter


def _write_organization_account_to_test_session_config(org_account):
    with FileLock(str(TEST_SESSION_CONFIG_TMP_PATH) + ".lock"):
        with open(TEST_SESSION_CONFIG_TMP_PATH, "a") as f:
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
    One reason being that it got deleted already by a fixture teardown.
    """
    with FileLock(str(TEST_SESSION_CONFIG_TMP_PATH) + ".lock"):
        config = configparser.ConfigParser()
        if (
            not config.read(TEST_SESSION_CONFIG_TMP_PATH)
            or "ORGANIZATION_ACCOUNT" not in config.sections()
        ):
            return None

        return Account(
            id=uuid.UUID(config.get("ORGANIZATION_ACCOUNT", "id")),
            name=config.get("ORGANIZATION_ACCOUNT", "name"),
        )


async def delete_old_org_accounts() -> None:
    config = await ConfigManager().refresh()
    client = LayerClient(config.client, logger)

    now_utc = datetime.now(tz=timezone.utc)
    for acc_id in config.client.organization_account_ids():
        try:
            acc_name = client.account.get_account_name_by_id(acc_id)
            if not acc_name.startswith(TEST_ORG_ACCOUNT_NAME_PREFIX):
                continue
            acc_creation_date = client.account.get_account_creation_date(acc_id)
            if now_utc - timedelta(days=1) > acc_creation_date:
                print(f"will delete org account {acc_id} because it is too old")
                try:
                    client.account.delete_account(account_id=acc_id)
                    print(f"deleted old org account {acc_id}")
                except LayerClientResourceNotFoundException:
                    continue
        except LayerClientResourceNotFoundException:
            continue


async def create_organization_account() -> Account:
    config = await ConfigManager().refresh()
    client = LayerClient(config.client, logger)
    org_account_name, display_name = pseudo_random_account_name()
    account = client.account.create_organization_account(
        org_account_name, display_name, deletion_allowed=True
    )

    # We need a new token with permissions for the new org account
    # TODO LAY-3583 LAY-3652
    await ConfigManager().refresh(force=True)

    return account


def _cleanup_organization_account() -> None:
    async def refresh() -> Config:
        return await ConfigManager().refresh()

    account = _read_organization_account_from_test_session_config()
    if not account:
        # we assume there is no more account to cleanup
        return

    loop = asyncio.get_event_loop()
    config = loop.run_until_complete(refresh())
    client = LayerClient(config.client, logger)
    try:
        client.account.delete_account(account_id=account.id)
    except LayerClientResourceNotFoundException as e:
        print(f"account already deleted: {e}")


@pytest.fixture()
def initialized_organization_account(client: LayerClient) -> Account:
    """
    This fixture injects the account created by the test session set up.
    Cleanup is handled by the session teardown.
    """
    account = _read_organization_account_from_test_session_config()
    assert account

    return account


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
    name_prefix = TEST_ORG_ACCOUNT_NAME_PREFIX
    gh_run_id = os.getenv("GITHUB_RUN_ID")
    gh_run_number = os.getenv("GITHUB_RUN_NUMBER")
    gh_job_id = os.getenv("GITHUB_JOB_ID")
    random_suffix = (
        f"{gh_run_id}-{gh_run_number}-{gh_job_id}"
        if gh_run_id
        else str(uuid.uuid4()).replace("-", "")
    )
    random_suffix_that_fits = random_suffix[: name_max_length - len(name_prefix)]
    name = f"{name_prefix}{random_suffix_that_fits}"
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
async def guest_client(guest_context) -> Iterator[LayerClient]:
    with guest_context():
        config = await ConfigManager().refresh()
        yield LayerClient(config.client, logger)


@pytest.fixture()
def initialized_project(client: LayerClient, request: Any) -> Iterator[Project]:
    project_name = pseudo_random_project_name(request)
    project = layer.init(project_name, fabric=Fabric.F_XSMALL.value)

    yield project
    _cleanup_project(client, project)


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
