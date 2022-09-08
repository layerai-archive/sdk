import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import layer
from layer import dataset, model
from layer.contracts.fabrics import Fabric
from layer.contracts.projects import Project
from layer.executables.ray_runtime import RayClientFunctionRuntime


@pytest.mark.ray
def test_dataset_build_ray(initialized_project: Project):
    @dataset("test_dataset_ray")
    def build_dataset(x):
        return pd.DataFrame({"a": [x, 2, 3]})

    function = build_dataset.bind(1).get_definition_with_bound_arguments()
    executable_path = function.package()
    RayClientFunctionRuntime.execute(
        executable_path,
        address="ray://127.0.0.1:20001",
        fabric=Fabric.F_SMALL,
    )

    actual_dataset = layer.get_dataset("test_dataset_ray").to_pandas()

    assert_frame_equal(actual_dataset, pd.DataFrame({"a": [1, 2, 3]}))


@pytest.mark.ray
def test_model_train_ray(initialized_project: Project):
    @model("test_model_ray")
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
    executable_path = function.package()
    RayClientFunctionRuntime.execute(
        executable_path,
        address="ray://127.0.0.1:20001",
        fabric=Fabric.F_SMALL,
    )

    actual_model = layer.get_model("test_model_ray")

    assert actual_model
    assert actual_model.predict([[0, 0, 0, 0]]).to_dict() == {0: {0: 1}}
