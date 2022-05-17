import pickle
from test.unit.decorators.util import project_client_mock
from unittest.mock import ANY, patch

import pytest
from sklearn.ensemble import RandomForestClassifier

from layer import Dataset, Model
from layer.decorators.model_decorator import model
from layer.decorators.pip_requirements_decorator import pip_requirements
from layer.exceptions.exceptions import ProjectInitializationException
from layer.global_context import set_current_project_name
from layer.projects.asset import AssetType


@model("model1", dependencies=[Dataset("bar"), Model("zoo")])
@pip_requirements(packages=["scikit-learn==0.23.2"])
def model1():
    from sklearn import datasets
    from sklearn.svm import SVC

    iris = datasets.load_iris()
    clf = SVC()
    result = clf.fit(iris.data, iris.target)
    print("model1 computed")
    return result


class TestModelDecorator:
    def test_model_decorator_assigns_attributes_to_function_before_calling_function(
        self,
    ):
        @model("forest")
        def train_my_model() -> RandomForestClassifier:
            return RandomForestClassifier()

        assert train_my_model.layer.get_entity_name() == "forest"
        assert train_my_model.layer.get_asset_type() == AssetType.MODEL

    def test_model_decorator_given_no_current_project_set_raise_exception(self):
        set_current_project_name(None)

        @model("forest2")
        def train_my_model() -> RandomForestClassifier:
            return RandomForestClassifier()

        with pytest.raises(
            ProjectInitializationException,
            match="Please specify the current project name globally with",
        ):
            train_my_model()

    @pytest.mark.parametrize(
        ("name", "function"),
        [
            ("model1", model1),
        ],
    )
    def test_model_definition_created_correctly(self, name, function):
        set_current_project_name("foo-test")
        with project_client_mock(), patch.object(
            function, "_train_model_locally_and_store_remotely"
        ) as mock_create_model:
            function()

            mock_create_model.assert_called_with(ANY, ANY, ANY)
            args = mock_create_model.call_args
            model_definition = args[0][0]

            assert model_definition.name == name
            assert model_definition.project_name == "foo-test"

            assert len(model_definition.dependencies) == 2
            assert model_definition.dependencies[0].name == "bar"
            assert isinstance(model_definition.dependencies[0], Dataset)
            assert model_definition.dependencies[1].name == "zoo"
            assert isinstance(model_definition.dependencies[1], Model)

            # Check if the function is decorated correctly
            assert function.layer.get_entity_name() == name
            assert function.layer.get_asset_type() == AssetType.MODEL
            assert function.layer.get_pip_packages() == ["scikit-learn==0.23.2"]

            # Check if the unpickled file contains the correct function
            model_definition.get_local_entity()
            # needed to create the pickled file on disk.
            loaded = pickle.load(open(model_definition._get_pickle_path(), "rb"))
            assert loaded.layer.get_entity_name() == name
            assert loaded.layer.get_asset_type() == AssetType.MODEL
            assert loaded.layer.get_pip_packages() == ["scikit-learn==0.23.2"]
