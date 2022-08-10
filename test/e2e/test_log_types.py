from uuid import UUID

import pandas as pd
from sklearn.svm import SVC

import layer
from layer.clients.layer import LayerClient
from layer.contracts.logged_data import LoggedDataType, Video
from layer.contracts.projects import Project
from layer.decorators import dataset, model, pip_requirements
from test.e2e.assertion_utils import E2ETestAsserter


def test_logging_in_remote_execution(
    initialized_project: Project, asserter: E2ETestAsserter, client: LayerClient
):
    # given
    dataset_name = "scalar_ds"

    str_tag = "str_tag"

    @dataset(dataset_name)
    def scalar():
        data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
        dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
        layer.log(
            {
                str_tag: "bar",
            }
        )
        return dataframe

    # when
    run = layer.run([scalar])

    # then
    asserter.assert_run_succeeded(run.id)

    first_ds = client.data_catalog.get_dataset_by_name(
        initialized_project.id, dataset_name
    )

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=str_tag, dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "bar"
    assert logged_data.logged_data_type == LoggedDataType.TEXT
    assert logged_data.tag == str_tag


def test_scalar_values_logged(
    initialized_project: Project, asserter: E2ETestAsserter, client: LayerClient
):
    # given
    dataset_name = "scalar_ds"

    str_tag = "str_tag"
    int_tag = "int_tag"
    bool_tag = "bool_tag"
    float_tag = "float_tag"
    list_tag = "list_tag"

    @dataset(dataset_name)
    def scalar():
        data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
        dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
        layer.log(
            {
                str_tag: "bar",
                int_tag: 123,
                bool_tag: True,
                float_tag: 1.11,
                list_tag: ["a", "b", "c"],
            }
        )
        return dataframe

    # when
    scalar()

    # then
    first_ds = client.data_catalog.get_dataset_by_name(
        initialized_project.id, dataset_name
    )

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=str_tag, dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "bar"
    assert logged_data.logged_data_type == LoggedDataType.TEXT
    assert logged_data.tag == str_tag

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=int_tag, dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "123"
    assert logged_data.logged_data_type == LoggedDataType.NUMBER
    assert logged_data.tag == int_tag

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=bool_tag, dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "True"
    assert logged_data.logged_data_type == LoggedDataType.BOOLEAN
    assert logged_data.tag == bool_tag

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=float_tag, dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == "1.11"
    assert logged_data.logged_data_type == LoggedDataType.NUMBER
    assert logged_data.tag == float_tag

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=list_tag, dataset_build_id=first_ds.build.id
    )
    assert logged_data.data == str(["a", "b", "c"])
    assert logged_data.logged_data_type == LoggedDataType.TEXT
    assert logged_data.tag == list_tag


def test_pandas_dataframe_logged(initialized_project: Project, client: LayerClient):
    # given
    ds_tag = "dataframe_tag"
    ds_name = "pandas_dataframe_log"

    @dataset(ds_name)
    def dataset_func():
        d = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data=d)
        layer.log({ds_tag: df})
        return df

    # then
    dataset_func()

    ds = client.data_catalog.get_dataset_by_name(initialized_project.id, ds_name)

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=ds_tag, dataset_build_id=ds.build.id
    )

    assert logged_data.logged_data_type == LoggedDataType.TABLE


def test_markdown_logged(initialized_project: Project, client: LayerClient):
    # given
    ds_tag = "dataframe_tag"
    ds_name = "markdown_dataframe_log"

    markdown = """
        # Markdown header
        Some code with [link](http://my link)
        """

    @dataset(ds_name)
    def dataset_func():
        layer.log({ds_tag: layer.Markdown(markdown)})
        return pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    # then
    dataset_func()

    ds = client.data_catalog.get_dataset_by_name(initialized_project.id, ds_name)

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=ds_tag, dataset_build_id=ds.build.id
    )

    assert logged_data.logged_data_type == LoggedDataType.MARKDOWN
    assert logged_data.data == markdown


def test_image_and_video_logged(initialized_project: Project, client: LayerClient):
    # given
    ds_name = "multimedia"
    model_name = "model_with_stepped_log"
    pil_image_tag = "pil_image_tag"
    image_path_tag = "image_path_tag"
    video_path_tag = "video_path_tag"
    stepped_pil_image_tab = "stepped_pil_image_tag"
    pytorch_tensor_video_tag = "pytorch_tensor_video_tag"

    @dataset(ds_name)
    @pip_requirements(packages=["moviepy==0.2.3.5"])
    def multimedia():
        import os
        from pathlib import Path

        from PIL import Image

        image = Image.open(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_logo.jpeg")
        layer.log({pil_image_tag: image})

        image_path = Path(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_logo.jpeg")
        layer.log({image_path_tag: image_path})

        video_path = Path(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_video.mp4")
        layer.log({video_path_tag: video_path})

        import torchvision

        tensor_video = torchvision.io.read_video(str(video_path))
        tensor_video = tensor_video[0].permute(0, 3, 1, 2)

        layer.log({pytorch_tensor_video_tag: Video(tensor_video)})

        return pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    layer.run([multimedia])

    ds = client.data_catalog.get_dataset_by_name(initialized_project.id, ds_name)

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=pil_image_tag, dataset_build_id=ds.build.id
    )
    assert logged_data.data.startswith("https://logged-data--layer")
    assert logged_data.data.endswith(pil_image_tag)
    assert logged_data.logged_data_type == LoggedDataType.IMAGE

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=image_path_tag, dataset_build_id=ds.build.id
    )
    assert logged_data.data.startswith("https://logged-data--layer")
    assert logged_data.data.endswith(image_path_tag)
    assert logged_data.logged_data_type == LoggedDataType.IMAGE

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=video_path_tag, dataset_build_id=ds.build.id
    )
    assert logged_data.data.startswith("https://logged-data--layer")
    assert logged_data.data.endswith(video_path_tag)
    assert logged_data.logged_data_type == LoggedDataType.VIDEO

    @pip_requirements(packages=["scikit-learn==0.23.2"])
    @model(model_name)
    def train_model():
        import os

        from PIL import Image
        from sklearn import datasets

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)

        image = Image.open(f"{os.getcwd()}/test/e2e/assets/log_assets/layer_logo.jpeg")
        for step in range(4, 6):
            layer.log({stepped_pil_image_tab: image}, step=step)

        print("model1 computed fully")
        return result

    train_model()

    mdl = layer.get_model(model_name)
    logged_data = client.logged_data_service_client.get_logged_data(
        tag=stepped_pil_image_tab, train_id=UUID(mdl.storage_config.train_id.value)
    )
    assert logged_data.logged_data_type == LoggedDataType.IMAGE
    assert len(logged_data.epoched_data) == 2
    assert logged_data.epoched_data[4].startswith("https://logged-data--layer")
    assert logged_data.epoched_data[4].endswith(f"{stepped_pil_image_tab}/epoch/4")
    assert logged_data.epoched_data[5].startswith("https://logged-data--layer")
    assert logged_data.epoched_data[5].endswith(f"{stepped_pil_image_tab}/epoch/5")


def test_matplotlib_objects_logged(initialized_project: Project, client: LayerClient):
    # given
    figure_tag = "matplotlib_figure_tag"
    plot_tag = "matplotlib_pyplot_tag"

    ds_name = "ds_with_plots"

    @dataset(ds_name)
    def dataset_func():
        import matplotlib.pyplot as plt
        import seaborn

        data = pd.DataFrame({"col": [1, 2, 42]})
        plot = seaborn.histplot(data=data, x="col", color="green")
        layer.log({plot_tag: plot})

        figure = plt.figure()
        figure.add_subplot(111)

        layer.log({figure_tag: figure})
        return pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    # then
    dataset_func()

    ds = client.data_catalog.get_dataset_by_name(initialized_project.id, ds_name)

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=figure_tag, dataset_build_id=ds.build.id
    )

    assert logged_data.data.startswith("https://logged-data--layer")
    assert logged_data.data.endswith(figure_tag)
    assert logged_data.logged_data_type == LoggedDataType.IMAGE

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=plot_tag, dataset_build_id=ds.build.id
    )

    assert logged_data.data.startswith("https://logged-data--layer")
    assert logged_data.data.endswith(plot_tag)
    assert logged_data.logged_data_type == LoggedDataType.IMAGE


def test_metrics_logged(initialized_project: Project, client: LayerClient):
    # given
    metric_tag_1 = "metric_tag_1"
    metric_tag_2 = "metric_tag_2"

    ds_name = "metrics_ds"

    @dataset(ds_name)
    def metrics():
        for step in range(1, 5):
            layer.log(
                {metric_tag_1: f"value {step}", metric_tag_2: f"value {step}"}, step
            )
        return pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})

    # then
    metrics()

    ds = client.data_catalog.get_dataset_by_name(initialized_project.id, ds_name)

    logged_data = client.logged_data_service_client.get_logged_data(
        tag=metric_tag_1, dataset_build_id=ds.build.id
    )

    assert logged_data.logged_data_type == LoggedDataType.TEXT
    # value from the last step
    assert logged_data.data == "value 4"
