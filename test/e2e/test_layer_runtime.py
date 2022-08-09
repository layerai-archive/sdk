from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import layer
from layer import dataset, model
from layer.contracts.projects import Project
from layer.executables.function import Function
from layer.executables.layer_runtime import LayerFunctionRuntime


def test_dataset_build(initialized_project: Project, tmpdir: Path):
    @dataset("test_dataset")
    def build_dataset():
        return pd.DataFrame({"a": [1, 2, 3]})

    function = Function.from_decorated(build_dataset)
    executable_path = function.package(output_dir=tmpdir)
    LayerFunctionRuntime.execute(executable_path, project=initialized_project.name)

    actual_dataset = layer.get_dataset("test_dataset").to_pandas()

    assert_frame_equal(actual_dataset, pd.DataFrame({"a": [1, 2, 3]}))


def test_model_trains(initialized_project: Project, tmpdir: Path):
    @model("test_model")
    def train_model():
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(
            n_samples=1000,
            n_features=4,
            n_informative=2,
            n_redundant=0,
            random_state=0,
            shuffle=False,
        )

        classifier = RandomForestClassifier(max_depth=2, random_state=0)

        return classifier.fit(X, y)

    function = Function.from_decorated(train_model)
    executable_path = function.package(output_dir=tmpdir)
    LayerFunctionRuntime.execute(executable_path, project=initialized_project.name)

    actual_model = layer.get_model("test_model")

    assert actual_model
    assert actual_model.predict([[0, 0, 0, 0]]).to_dict() == {0: {0: 1}}
