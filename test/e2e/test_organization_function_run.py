from pathlib import Path
from typing import Any, Iterator

import pytest

import layer
from layer import refresh_login
from layer.clients import LayerClient
from layer.contracts.accounts import Account
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from test.e2e.assertion_utils import E2ETestAsserter
from test.e2e.conftest import (
    _cleanup_project,
    pseudo_random_account_name,
    pseudo_random_project_name,
)
from test.e2e.test_function_dataset_run import (
    test_remote_run_with_dependent_datasets_succeeds_and_registers_metadata as test_remote_dependent_datasets,
)
from test.e2e.test_function_model_run import (
    test_remote_run_succeeds_and_registers_metadata as test_remote_model_train,
)
from test.e2e.test_layer_runtime import (
    test_dataset_build as test_remote_layer_runtime_dataset_build,
)


# We use a wrapping Test class so we create only one organization
# until we handle newly acquired permissions (after org creation)
# better
class TestOrganization:
    @pytest.fixture(scope="class")
    def initialized_organization_account(
        self, client: LayerClient, request: Any
    ) -> Iterator[Account]:
        org_account_name = pseudo_random_account_name(request)
        account = client.account.create_organization_account(org_account_name)

        # We need a new token with permissions for the new org account
        # TODO LAY-3583 LAY-3652
        refresh_login(force=True)

        yield account
        self._cleanup_account(client, account)

    @staticmethod
    def _cleanup_account(client: LayerClient, account: Account):
        client.account.delete_account(account_id=account.id)

    @pytest.fixture()
    def initialized_organization_project(
        self,
        client: LayerClient,
        initialized_organization_account: Account,
        request: Any,
    ) -> Iterator[Project]:
        account_name = initialized_organization_account.name
        project_name = pseudo_random_project_name(request)
        project = layer.init(
            f"{account_name}/{project_name}", fabric=Fabric.F_XSMALL.value
        )

        yield project
        _cleanup_project(client, project)

    def test_remote_run_dependent_datasets(
        self, initialized_organization_project: Project, asserter: E2ETestAsserter
    ):
        test_remote_dependent_datasets(initialized_organization_project, asserter)

    def test_remote_run_model_train(
        self, initialized_organization_project: Project, asserter: E2ETestAsserter
    ):
        test_remote_model_train(initialized_organization_project, asserter)

    def test_layer_runtime_dataset_build(
        self, initialized_organization_project: Project, tmpdir: Path
    ):
        test_remote_layer_runtime_dataset_build(
            initialized_organization_project, tmpdir
        )
