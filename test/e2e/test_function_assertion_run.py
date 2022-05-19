from test.e2e.assertion_utils import E2ETestAsserter

import pandas as pd
import pytest
from sklearn.svm import SVC

import layer
from layer.contracts.projects import Project
from layer.decorators import dataset, model, pip_requirements
from layer.decorators.assertions import (
    assert_not_null,
    assert_true,
    assert_valid_values,
)
from layer.exceptions.exceptions import (
    ProjectDatasetBuildExecutionException,
    ProjectModelExecutionException,
)


def test_remote_run_succeeds_and_registers_metadata_when_assertion_succeeds(
    initialized_project: Project, asserter: E2ETestAsserter
):
    # given
    dataset_name = "fake_users_1"

    @dataset(dataset_name)
    @assert_not_null(["name", "address"])
    @pip_requirements(packages=["Faker==13.2.0"])
    @assert_valid_values("name", ["test_name"])
    def prepare_data():
        from faker import Faker

        fake = Faker()
        pandas_df = pd.DataFrame(
            [
                {
                    "name": "test_name",
                    "address": fake.address(),
                    "email": fake.email(),
                    "city": fake.city(),
                    "state": fake.state(),
                }
                for _ in range(10)
            ]
        )
        return pandas_df

    # when
    run = layer.run([prepare_data])

    # then
    asserter.assert_run_succeeded(run.run_id)
    ds = layer.get_dataset(dataset_name)
    assert len(ds.to_pandas().index) == 10


def test_remote_run_with_model_fails_when_assertion_fails(initialized_project: Project):
    def assert_function(model: SVC) -> bool:
        return model.max_iter == 10

    @model("iris-model")
    @assert_true(assert_function)
    @pip_requirements(packages=["scikit-learn==0.23.2"])
    def train_model():
        from sklearn import datasets

        iris = datasets.load_iris()
        clf = SVC(max_iter=-1)
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")
        return result

    with pytest.raises(ProjectModelExecutionException, match=".*failed.*assert.*"):
        layer.run([train_model])


def test_remote_run_with_dataset_fails_when_assertion_fails(
    initialized_project: Project,
):
    @dataset("fake_users_2")
    @assert_valid_values("name", ["valid_name"])
    @pip_requirements(packages=["Faker==13.2.0"])
    def prepare_data():
        from faker import Faker

        fake = Faker()
        pandas_df = pd.DataFrame(
            [
                {
                    "name": "invalid_name",
                    "address": fake.address(),
                    "email": fake.email(),
                    "city": fake.city(),
                    "state": fake.state(),
                }
                for _ in range(10)
            ]
        )
        return pandas_df

    with pytest.raises(
        ProjectDatasetBuildExecutionException, match=".*failed.*assert.*"
    ):
        layer.run([prepare_data])
