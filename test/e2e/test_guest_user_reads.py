import pandas as pd
import pytest

import layer
from layer import global_context
from layer.clients.layer import LayerClient
from layer.contracts.projects import Project
from layer.decorators import dataset, model
from layer.exceptions.exceptions import LayerClientException


@pytest.fixture()
def populated_project(initialized_project: Project) -> Project:
    """
    A project that is populated with a dataset and a model
    """

    @dataset("dataset1")
    def prepare_data():
        data = [["id1", 10], ["id2", 15], ["id3", 14]]
        pandas_df = pd.DataFrame(data, columns=["id", "value"])
        return pandas_df

    @model("model1")
    def train_model():
        from sklearn import datasets
        from sklearn.svm import SVC

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")
        return result

    # when
    prepare_data()
    train_model()

    assert global_context.get_active_context() is None

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
    populated_project: Project,
    guest_context,
):
    with guest_context():
        project_path = f"{populated_project.account.name}/{populated_project.name}"
        with pytest.raises(LayerClientException) as error:
            layer.get_dataset(f"{project_path}/datasets/dataset1").to_pandas()

        assert "not found" in str(error)


def test_guest_user_public_dataset_read(
    populated_public_project: Project,
    guest_context,
):
    with guest_context():
        project_path = (
            f"{populated_public_project.account.name}/{populated_public_project.name}"
        )
        df = layer.get_dataset(f"{project_path}/datasets/dataset1").to_pandas()

        assert len(df) == 3
        assert df.values[0][0] == "id1"
        assert df.values[0][1] == 10


def test_guest_user_private_model_read(
    populated_project: Project,
    guest_context,
):
    with guest_context():
        project_path = f"{populated_project.account.name}/{populated_project.name}"
        with pytest.raises(LayerClientException) as error:
            layer.get_model(f"{project_path}/models/model1")

        assert "not found" in str(error)


def test_guest_user_public_model_read(
    populated_public_project: Project,
    guest_context,
):
    with guest_context():
        project_path = (
            f"{populated_public_project.account.name}/{populated_public_project.name}"
        )
        train = layer.get_model(f"{project_path}/models/model1").get_train()

        from sklearn.svm import SVC

        assert isinstance(train, SVC)
