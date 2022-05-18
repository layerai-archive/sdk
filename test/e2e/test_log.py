from test.e2e.assertion_utils import E2ETestAsserter

import pandas as pd

import layer
from layer.clients.layer import LayerClient
from layer.contracts.logged_data import LoggedDataType
from layer.contracts.projects import Project
from layer.decorators import dataset


def test_remote_run_with_log(
    initialized_project: Project, asserter: E2ETestAsserter, client: LayerClient
):
    # given
    dataset_name = "users"

    @dataset(dataset_name)
    def prepare_data():
        data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
        dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
        layer.log(
            {
                "str-log": "bar",
                "int-log": 123,
            }
        )
        return dataframe

    # when
    run = layer.run([prepare_data])

    # then
    asserter.assert_run_succeeded(run.run_id)

    first_ds = client.data_catalog.get_dataset_by_name(
        initialized_project.id, dataset_name
    )

    logged_data = client.logged_data_service_client.get_logged_data(
        tag="str-log", dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "bar"
    assert logged_data.logged_data_type == LoggedDataType.TEXT
    assert logged_data.tag == "str-log"

    logged_data = client.logged_data_service_client.get_logged_data(
        tag="int-log", dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "123"
    assert logged_data.logged_data_type == LoggedDataType.NUMBER
    assert logged_data.tag == "int-log"
