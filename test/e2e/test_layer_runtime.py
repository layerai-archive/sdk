import pandas as pd
from pandas.testing import assert_frame_equal

import layer
from layer import dataset, model
from layer.contracts.projects import Project
from layer.executables.runtime import BaseFunctionRuntime


def test_dataset_build(initialized_project: Project):
    @dataset("test_dataset")
    def build_dataset(x):
        return pd.DataFrame({"a": [x, 2, 3]})

    function = build_dataset.bind(1).get_definition_with_bound_arguments()
    executable_path = function.package(executables_feature_active=True)
    BaseFunctionRuntime.execute(executable_path)

    actual_dataset = layer.get_dataset("test_dataset").to_pandas()

    assert_frame_equal(actual_dataset, pd.DataFrame({"a": [1, 2, 3]}))


def test_model_trains(initialized_project: Project):
    @model("test_model")
    def train_model(n_features=0):
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(
            n_samples=1000,
            n_features=n_features,
            n_informative=2,
            n_redundant=0,
            random_state=0,
            shuffle=False,
        )

        classifier = RandomForestClassifier(max_depth=2, random_state=0)

        return classifier.fit(X, y)

    function = train_model.bind(n_features=4).get_definition_with_bound_arguments()
    executable_path = function.package(executables_feature_active=True)
    BaseFunctionRuntime.execute(executable_path)

    actual_model = layer.get_model("test_model")

    assert actual_model
    assert actual_model.predict([[0, 0, 0, 0]]).to_dict() == {0: {0: 1}}


def test_dataset_build_local_run(initialized_project: Project):
    @dataset("test_dataset_local")
    def build_dataset(x):
        return pd.DataFrame({"a": [x, 2, 3]})

    expected_dataset = pd.DataFrame({"a": [1, 2, 3]})

    assert_frame_equal(build_dataset(1), expected_dataset)

    fetched_dataset = layer.get_dataset("test_dataset_local").to_pandas()

    assert_frame_equal(fetched_dataset, expected_dataset)


def test_model_trains_local_run(initialized_project: Project):
    @model("test_model_local")
    def train_model(n_features=0):
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(
            n_samples=1000,
            n_features=n_features,
            n_informative=2,
            n_redundant=0,
            random_state=0,
            shuffle=False,
        )

        classifier = RandomForestClassifier(max_depth=2, random_state=0)

        return classifier.fit(X, y)

    trained_model = train_model(n_features=4)
    assert trained_model.predict([[0, 0, 0, 0]]) == [1]

    fetched_model = layer.get_model("test_model_local")

    assert fetched_model
    fetched_model.predict([[0, 0, 0, 0]]).to_dict() == {0: {0: 1}}
