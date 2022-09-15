from typing import Callable, Optional
from uuid import UUID

import pandas as pd
import pytest

import layer
from layer import Context, context
from layer.clients.layer import LayerClient
from layer.contracts.asset import AssetType
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from layer.decorators import dataset
from layer.exceptions.exceptions import LayerClientException
from layer.runs import context as run_context
from test.e2e.assertion_utils import E2ETestAsserter
from test.e2e.common_scenarios import (
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata,
)
from test.e2e.conftest import _cleanup_project


def test_remote_run_with_dependent_datasets_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    remote_run_with_dependent_datasets_succeeds_and_registers_metadata(asserter)


def test_local_run_without_decorator_succeeds_and_registers_metadata(
    initialized_project: Project, client: LayerClient
):
    # given
    dataset_name = "zoo-animals"

    def run_experiment(test_arg, cb: Callable[[Context], None]):
        assert test_arg == "test_arg"

        ctx = context.get_active_context()
        assert ctx is not None
        assert ctx.asset_type() == AssetType.DATASET
        assert ctx.asset_name() == dataset_name
        cb(ctx)

        layer.log({"test_key": 1.5})

    # and
    class ContextHolder:
        """
        We need to have access to the context after it is gone so we can run assertions
        """

        context: Optional[Context]

        def save(self, ctx: Context):
            self.context = ctx

        def dataset_build_id(self) -> UUID:
            assert self.context is not None
            assert self.context.dataset_build() is not None
            return self.context.dataset_build().id

    ctx_holder = ContextHolder()

    # when
    layer.dataset(dataset_name)(run_experiment)("test_arg", ctx_holder.save)

    # then
    assert context.get_active_context() is None
    # and
    logged_data = client.logged_data_service_client.get_logged_data(
        tag="test_key", dataset_build_id=ctx_holder.dataset_build_id()
    )
    assert logged_data.value == "1.5"


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
    assert run_context.current_project_full_name().project_name == second_project_name
    assert run_context.default_fabric() == Fabric.F_XSMALL
    assert run_context.get_pip_packages() == ["tensorflow==2.3.2"]
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
        run_context.current_project_full_name().project_name == initialized_project.name
    )
    assert run_context.default_fabric() is None
    assert run_context.get_pip_packages() is None
    assert context.get_active_context() is None

    _cleanup_project(client, project)
