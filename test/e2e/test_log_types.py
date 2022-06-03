import pandas as pd

import layer
from layer.contracts.projects import Project
from layer.decorators import dataset
from test.e2e.assertion_utils import E2ETestAsserter


# Metrics (i.e. layer.log with number and epoch)

def test_plot_logged():
    pass
    # from: https://colab.research.google.com/github/layerai/examples/blob/main/recommendation-system/Ecommerce_Recommendation_System.ipynb#scrollTo=kqVnGUjtmicl
    # plt.hist(kmeans_model.labels_, rwidth=0.7)
    # plt.ylabel("Number of Products")
    # plt.xlabel("Cluster No")

    # Layer logs the plot (figure)
    # fig = plt.gcf()
    # layer.log({"Product Distribution over Clusters": fig})


def test_scalar_values_logged(initialized_project: Project, asserter: E2ETestAsserter):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("scalar_log")
    def scalar():
        layer.log({"string": "string_value"})
        layer.log({"boolean": True})
        layer.log({"numeric_int": 123})
        layer.log({"numeric_float": 1.1})
        layer.log({"string": "string_value", "boolean": True, "numeric_int": 123,
                   "numeric_float": 1.1})
        return pd.DataFrame()

    # then
    scalar()


def test_pandas_dataframe_logged(initialized_project: Project, asserter: E2ETestAsserter):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("pandas_dataframe_log")
    def pandas_dataframe():
        d = {'col1': [1, 2], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        layer.log({"dataframe": df})
        return pd.DataFrame()

    # then
    pandas_dataframe()


def test_path_logged(initialized_project: Project, asserter: E2ETestAsserter):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("path_log")
    def path():
        from pathlib import Path
        import os
        path = Path(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_logo.jpeg")
        layer.log({"path": path})
        return pd.DataFrame()

    # then
    path()


def test_pil_image_logged(initialized_project: Project, asserter: E2ETestAsserter):
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


def test_matplotlib_figure_logged(initialized_project: Project, asserter: E2ETestAsserter):
    # when
    layer.init(initialized_project.name)

    # given
    @dataset("matplotlib_figure")
    def matplotlib_figure():
        import matplotlib.pyplot as plt
        figure = plt.figure()
        plt = figure.add_subplot(111)

        plt.title = 'Figure'
        layer.log({"matplotlib_figure": figure})
        return pd.DataFrame()

    # then
    matplotlib_figure()


def test_matplotlib_plot_logged(initialized_project: Project, asserter: E2ETestAsserter):
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


def test_metrics_logged(initialized_project: Project, asserter: E2ETestAsserter):
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
