import pandas as pd
import pytest

import layer
from layer import context, global_context
from layer.clients.layer import LayerClient
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from layer.decorators import dataset
from layer.exceptions.exceptions import LayerClientException
from test.e2e.assertion_utils import E2ETestAsserter
from test.e2e.common_scenarios import (
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata,
)
from test.e2e.conftest import _cleanup_project


def test_remote_run_with_dependent_datasets_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata(asserter)


def test_multiple_inits_switch_context(
    initialized_project: Project, client: LayerClient
):
    # given
    second_project_name = initialized_project.name + "-double"
    dataset_name = "sample"

    @dataset(dataset_name)
    def prepare_data():
        data = [["id1", 10], ["id2", 15], ["id3", 14]]
        pandas_df = pd.DataFrame(data, columns=["id", "value"])
        return pandas_df

    # when
    project = layer.init(
        second_project_name,
        fabric=Fabric.F_XSMALL.value,
        pip_packages=["tensorflow==2.3.2"],
    )
    prepare_data()

    # then
    assert (
        global_context.current_project_full_name().project_name == second_project_name
    )
    assert global_context.default_fabric() == Fabric.F_XSMALL
    assert global_context.get_pip_packages() == ["tensorflow==2.3.2"]
    assert context.get_active_context() is None

    # and when
    layer.init(initialized_project.name)

    # then
    with pytest.raises(LayerClientException, match=r"Build by Path not found.*"):
        layer.get_dataset(dataset_name).to_pandas()
    assert context.get_active_context() is None

    assert (
        len(
            layer.get_dataset(
                "{}/datasets/{}".format(second_project_name, dataset_name)
            ).to_pandas()
        )
        == 3
    )
    assert (
        global_context.current_project_full_name().project_name
        == initialized_project.name
    )
    assert global_context.default_fabric() is None
    assert global_context.get_pip_packages() is None
    assert context.get_active_context() is None

    _cleanup_project(client, project)
