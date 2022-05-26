import pickle
from typing import Any, Callable
from unittest.mock import ANY, patch

import pytest

from layer import Dataset, Model
from layer.contracts.asset import AssetType
from layer.decorators.model_decorator import model
from layer.decorators.pip_requirements_decorator import pip_requirements
from layer.exceptions.exceptions import ProjectInitializationException
from layer.global_context import set_current_project_name
from test.unit.decorators.util import project_client_mock


def _make_test_model_function(name: str) -> Callable[..., Any]:
    from sklearn.ensemble import RandomForestClassifier

    @model(
        name, dependencies=["datasets/bar", "models/foo", Dataset("baz"), Model("zoo")]
    )
    @pip_requirements(packages=["scikit-learn==0.23.2"])
    def func() -> RandomForestClassifier:
        return RandomForestClassifier()

    return func


class TestModelDecorator:
    def test_model_decorator_assigns_attributes_to_function_before_calling_function(
        self,
    ) -> None:
        func = _make_test_model_function("forest")

        assert func.layer.get_entity_name() == "forest"
        assert func.layer.get_asset_type() == AssetType.MODEL

    def test_model_decorator_given_no_current_project_set_raise_exception(self) -> None:
        set_current_project_name(None)

        func = _make_test_model_function("forest2")

        with pytest.raises(
            ProjectInitializationException,
            match="Please specify the current project name globally with",
        ):
            func()

    @pytest.mark.parametrize(("name",), [("model1",)])
    def test_model_definition_created_correctly(self, name: str) -> None:
        func = _make_test_model_function(name)
        set_current_project_name("foo-test")
        with project_client_mock(), patch.object(
            func, "_train_model_locally_and_store_remotely"
        ) as mock_create_model:
            func()

            mock_create_model.assert_called_with(ANY, ANY, ANY)
            args = mock_create_model.call_args
            model_definition = args[0][0]

            assert model_definition.name == name
            assert model_definition.project_name == "foo-test"

            assert len(model_definition.dependencies) == 4
            assert model_definition.dependencies[0].entity_name == "bar"
            assert model_definition.dependencies[0].asset_type == AssetType.DATASET
            assert model_definition.dependencies[1].entity_name == "foo"
            assert model_definition.dependencies[1].asset_type == AssetType.MODEL
            assert model_definition.dependencies[2].entity_name == "baz"
            assert model_definition.dependencies[2].asset_type == AssetType.DATASET
            assert model_definition.dependencies[3].entity_name == "zoo"
            assert model_definition.dependencies[3].asset_type == AssetType.MODEL

            # Check if the function is decorated correctly
            assert func.layer.get_entity_name() == name
            assert func.layer.get_asset_type() == AssetType.MODEL
            assert func.layer.get_pip_packages() == ["scikit-learn==0.23.2"]

            # Check if the unpickled file contains the correct function
            assert model_definition.pickle_path.exists()
            loaded = pickle.load(open(model_definition.pickle_path, "rb"))
            assert loaded.layer.get_entity_name() == name
            assert loaded.layer.get_asset_type() == AssetType.MODEL
            assert loaded.layer.get_pip_packages() == ["scikit-learn==0.23.2"]
