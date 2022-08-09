import pandas as pd
from sklearn.svm import SVC

import layer
from layer import Dataset, model
from layer.decorators import dataset, pip_requirements
from test.e2e.assertion_utils import E2ETestAsserter


"""
IMPORTANT:
These scenarios are defined OUTSIDE test modules (files) for a reason:
importing test methods from a test file has many unwanted behaviours like
running tests with different contexts, injecting wrong fixtures and creating
unwanted projects, etc...
"""


# project set from execution context
def remote_run_with_dependent_datasets_succeeds_and_registers_metadata(
    asserter: E2ETestAsserter,
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
    asserter.assert_run_succeeded(run.id)

    first_ds = layer.get_dataset(dataset_name)
    first_pandas = first_ds.to_pandas()
    assert len(first_pandas.index) == 10
    assert len(first_pandas.values[0]) == 5

    ds = layer.get_dataset(transformed_dataset_name)
    pandas = ds.to_pandas()
    assert len(pandas.index) == 10
    assert len(pandas.values[0]) == 4  # only 4 columns in modified dataset


# project set from execution context
def remote_run_with_model_train_succeeds_and_registers_metadata(
    asserter: E2ETestAsserter,
):
    # given
    model_name = "foo-model"

    @model(model_name)
    @pip_requirements(packages=["scikit-learn==0.23.2"])
    def train_model():
        from sklearn import datasets

        iris = datasets.load_iris()
        clf = SVC()
        result = clf.fit(iris.data, iris.target)
        print("model1 computed")
        return result

    # when
    run = layer.run([train_model])

    # then
    asserter.assert_run_succeeded(run.id)
    mdl = layer.get_model(model_name)
    assert isinstance(mdl.get_train(), SVC)
