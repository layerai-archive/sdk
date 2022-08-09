from layer.contracts.projects import Project
from test.e2e.assertion_utils import E2ETestAsserter
from test.e2e.common_scenarios import (
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata,
    remote_run_with_model_train_succeeds_and_registers_metadata,
)


def test_remote_run_dependent_dataset(
    initialized_organization_project: Project, asserter: E2ETestAsserter
):
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata(asserter)


def test_remote_run_model_train(
    initialized_organization_project: Project, asserter: E2ETestAsserter
):
    remote_run_with_model_train_succeeds_and_registers_metadata(asserter)
