import pandas as pd

import layer
from layer.clients.layer import LayerClient
from layer.contracts.logged_data import LoggedDataType
from layer.contracts.projects import Project
from layer.decorators import dataset
from test.e2e.assertion_utils import E2ETestAsserter


def test_scalar_values_logged(
    initialized_project: Project, asserter: E2ETestAsserter, client: LayerClient
):
    # when
    layer.init(initialized_project.name)

    dataset_name = "scalar_ds"

    # given
    @dataset(dataset_name)
    def scalar():
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
    scalar()

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


def test_pandas_dataframe_logged(initialized_project: Project):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("pandas_dataframe_log")
    def pandas_dataframe():
        d = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data=d)
        layer.log({"dataframe": df})
        return pd.DataFrame()

    # then
    pandas_dataframe()


def test_path_logged(initialized_project: Project):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("path_log")
    def path():
        import os
        from pathlib import Path

        path = Path(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_logo.jpeg")
        layer.log({"path": path})
        return pd.DataFrame()

    # then
    path()


def test_pil_image_logged(initialized_project: Project):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("pil_image")
    def pil_image():
        import os

        from PIL import Image

        image = Image.open(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_logo.jpeg")
        layer.log({"pil_image": image})
        return pd.DataFrame()

    # then
    pil_image()


def test_matplotlib_figure_logged(initialized_project: Project):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("matplotlib_figure")
    def matplotlib_figure():
        import matplotlib.pyplot as plt

        figure = plt.figure()
        plt = figure.add_subplot(111)

        layer.log({"matplotlib_figure": figure})
        return pd.DataFrame()

    # then
    matplotlib_figure()


def test_matplotlib_plot_logged(initialized_project: Project):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("matplotlib_pyplot")
    def matplotlib_pyplot():
        import seaborn

        data = pd.DataFrame({"col": [1, 2, 42]})
        plot = seaborn.histplot(data=data, x="col", color="green")
        layer.log({"matplotlib_pyplot": plot})
        return pd.DataFrame()

    # then
    matplotlib_pyplot()


def test_metrics_logged(initialized_project: Project):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("metrics")
    def metrics():
        for step in range(1, 5):
            layer.log({"metric1": "value1", "metric2": "value2"}, step)
        return pd.DataFrame()

    # then
    metrics()
