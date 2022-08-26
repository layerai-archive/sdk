from uuid import UUID

import pandas as pd
import pytest

import layer
from layer import context
from layer.clients.layer import LayerClient
from layer.contracts.asset import AssetPath, AssetType
from layer.contracts.logged_data import LoggedDataType
from layer.contracts.projects import Project
from layer.decorators import dataset, model
from layer.exceptions.exceptions import (
    LayerClientException,
    LayerClientResourceNotFoundException,
)


dataset_log_tag = "dataset-log-tag"
dataset_log_value = 123
model_log_tag = "model-log-tag"
model_log_value = 345


@pytest.fixture()
def populated_project(initialized_project: Project) -> Project:
    """
    A project that is populated with a dataset and a model
    """

    @dataset("dataset1")
    def prepare_data():
        data = [["id1", 10], ["id2", 15], ["id3", 14]]
        pandas_df = pd.DataFrame(data, columns=["id", "value"])

        layer.log({dataset_log_tag: dataset_log_value})

        return pandas_df

    @model("model1")
    def train_model():
        from sklearn import datasets
        from sklearn.svm import SVC

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")

        layer.log({model_log_tag: model_log_value})

        return result

    # when
    prepare_data()
    train_model()

    assert context.get_active_context() is None

    return initialized_project


@pytest.fixture()
def populated_public_project(
    client: LayerClient, populated_project: Project
) -> Project:
    # make project public
    client.project_service_client.set_project_visibility(
        populated_project.full_name, is_public=True
    )
    return populated_project


def test_guest_user_private_dataset_read(
    client: LayerClient,
    populated_project: Project,
    guest_context,
    guest_client: LayerClient,
):
    project_path = f"{populated_project.account.name}/{populated_project.name}"
    name = f"{project_path}/models/model1"
    asset_path = AssetPath.parse(name, expected_asset_type=AssetType.MODEL)
    # We use the non-guest client because guests cannot access this model, but we do need it to find the model train ID, which is needed for get_logged_data below.
    mdl = client.model_catalog.load_model_by_path(path=asset_path.path())

    with guest_context():
        # dataset
        project_path = f"{populated_project.account.name}/{populated_project.name}"
        with pytest.raises(LayerClientException) as error:
            layer.get_dataset(f"{project_path}/datasets/dataset1").to_pandas()

        assert "not found" in str(error)

        dataset = client.data_catalog.get_dataset_by_name(
            populated_project.id, "dataset1"
        )

        with pytest.raises(LayerClientResourceNotFoundException):
            guest_client.logged_data_service_client.get_logged_data(
                tag=dataset_log_tag, dataset_build_id=dataset.build.id
            )

        # model
        with pytest.raises(LayerClientResourceNotFoundException):
            layer.get_model(f"{project_path}/models/model1")

        with pytest.raises(LayerClientResourceNotFoundException):
            guest_client.logged_data_service_client.get_logged_data(
                tag=model_log_tag, train_id=UUID(mdl.storage_config.train_id.value)
            )


def test_guest_user_public_dataset_read(
    populated_public_project: Project,
    guest_context,
    guest_client: LayerClient,
):
    with guest_context():
        # dataset
        project_path = (
            f"{populated_public_project.account.name}/{populated_public_project.name}"
        )
        df = layer.get_dataset(f"{project_path}/datasets/dataset1").to_pandas()

        assert len(df) == 3
        assert df.values[0][0] == "id1"
        assert df.values[0][1] == 10

        dataset = guest_client.data_catalog.get_dataset_by_name(
            populated_public_project.id, "dataset1"
        )

        logged_data = guest_client.logged_data_service_client.get_logged_data(
            tag=dataset_log_tag, dataset_build_id=dataset.build.id
        )

        assert logged_data.value == str(dataset_log_value)
        assert logged_data.logged_data_type == LoggedDataType.NUMBER
        assert logged_data.tag == dataset_log_tag

        # model
        project_path = (
            f"{populated_public_project.account.name}/{populated_public_project.name}"
        )
        mdl = layer.get_model(f"{project_path}/models/model1")
        train = mdl.get_train()

        from sklearn.svm import SVC

        assert isinstance(train, SVC)

        logged_data = guest_client.logged_data_service_client.get_logged_data(
            tag=model_log_tag, train_id=UUID(mdl.storage_config.train_id.value)
        )

        assert logged_data.value == str(model_log_value)
        assert logged_data.logged_data_type == LoggedDataType.NUMBER
        assert logged_data.tag == model_log_tag


def test_guest_user_cannot_init_project(
    guest_context,
):
    with guest_context():
        with pytest.raises(LayerClientException) as error:
            layer.init("foo-bar")

        assert "UNAUTHENTICATED" in str(
            error
        ) and "does not have a Layer account" in str(error)
