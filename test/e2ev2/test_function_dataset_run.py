from test.e2ev2.assertion_utils import E2ETestAsserter
from test.e2ev2.conftest import _cleanup_project

import pandas as pd
import pytest

import layer
from layer import Dataset, global_context
from layer.common import LayerClient
from layer.data_classes import Fabric
from layer.decorators import dataset, pip_requirements
from layer.exceptions.exceptions import LayerClientException
from layer.projects.project import Project


def test_remote_run_with_dependent_datasets_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    # given
    dataset_name = "users"
    transformed_dataset_name = "tusers"

    @dataset(dataset_name)
    @pip_requirements(packages=["Faker==13.2.0"])
    def prepare_data():
        from faker import Faker

        fake = Faker()
        pandas_df = pd.DataFrame(
            [
                {
                    "name": fake.name(),
                    "address": fake.address(),
                    "email": fake.email(),
                    "city": fake.city(),
                    "state": fake.state(),
                }
                for _ in range(10)
            ]
        )
        return pandas_df

    @dataset(transformed_dataset_name, dependencies=[Dataset(dataset_name)])
    def transform_data():
        df = layer.get_dataset(dataset_name).to_pandas()
        df = df.drop(["address"], axis=1)
        return df

    # when
    run = layer.run([prepare_data, transform_data])

    # then
    asserter.assert_run_succeeded(run.run_id)

    first_ds = layer.get_dataset(dataset_name)
    first_pandas = first_ds.to_pandas()
    assert len(first_pandas.index) == 10
    assert len(first_pandas.values[0]) == 5

    ds = layer.get_dataset(transformed_dataset_name)
    pandas = ds.to_pandas()
    assert len(pandas.index) == 10
    assert len(pandas.values[0]) == 4  # only 4 columns in modified dataset


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
    assert global_context.current_project_name() == second_project_name
    assert global_context.default_fabric() == Fabric.F_XSMALL
    assert global_context.get_pip_packages() == ["tensorflow==2.3.2"]
    assert global_context.get_active_context() is None

    # and when
    layer.init(initialized_project.name)

    # then
    with pytest.raises(LayerClientException, match=r"Dataset not found.*"):
        layer.get_dataset(dataset_name).to_pandas()
    assert global_context.get_active_context() is None

    assert (
        len(
            layer.get_dataset(
                "{}/datasets/{}".format(second_project_name, dataset_name)
            ).to_pandas()
        )
        == 3
    )
    assert global_context.current_project_name() == initialized_project.name
    assert global_context.default_fabric() is None
    assert global_context.get_pip_packages() is None
    assert global_context.get_active_context() is None

    _cleanup_project(client, project)
