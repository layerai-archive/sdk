from layer.contracts.projects import Project
from test.e2e.assertion_utils import E2ETestAsserter
from test.e2e.common_scenarios import (
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata,
    remote_run_with_model_train_succeeds_and_registers_metadata,
)


def test_remote_runs(
    initialized_organization_project: Project, asserter: E2ETestAsserter
):
    """
    Due to max_active_runs=1 limitation on a new organization account, we cannot
    have parallel tests.
    """
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata(asserter)

    remote_run_with_model_train_succeeds_and_registers_metadata(asserter)
