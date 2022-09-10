import re
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from layer import Dataset, Model
from layer.clients.model_catalog import ModelCatalogClient
from layer.contracts.asset import AssetType
from layer.decorators.model_decorator import model
from layer.decorators.pip_requirements_decorator import pip_requirements
from layer.exceptions.exceptions import (
    ProjectInitializationException,
    RuntimeMemoryException,
)
from layer.global_context import reset_to
from layer.settings import LayerSettings
from test.unit.decorators.util import project_client_mock


def _make_test_model_function(name: str) -> Callable[..., Any]:
    from sklearn.ensemble import RandomForestClassifier

    @model(
        name,
        dependencies=["datasets/bar", "models/foo", Dataset("baz"), Model("zoo")],
        description="my description",
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

        assert func.layer.get_asset_name() == "forest"
        assert func.layer.get_asset_type() == AssetType.MODEL

    def test_model_decorator_given_no_current_project_set_raise_exception(self) -> None:
        reset_to(None)

        func = _make_test_model_function("forest2")

        with pytest.raises(
            ProjectInitializationException,
            match="Please specify the current project name globally with",
        ):
            func()

    @pytest.mark.parametrize(("name",), [("model1",)])
    def test_model_definition_created_correctly(self, name: str) -> None:
        func = _make_test_model_function(name)
        reset_to("acc-name/foo-test")

        model_definition = func.get_definition_with_bound_arguments()

        assert model_definition.asset_name == name
        assert model_definition.project_name == "foo-test"
        assert model_definition.description == "my description"

        assert len(model_definition.asset_dependencies) == 4
        assert model_definition.asset_dependencies[0].asset_name == "bar"
        assert model_definition.asset_dependencies[0].asset_type == AssetType.DATASET
        assert model_definition.asset_dependencies[1].asset_name == "foo"
        assert model_definition.asset_dependencies[1].asset_type == AssetType.MODEL
        assert model_definition.asset_dependencies[2].asset_name == "baz"
        assert model_definition.asset_dependencies[2].asset_type == AssetType.DATASET
        assert model_definition.asset_dependencies[3].asset_name == "zoo"
        assert model_definition.asset_dependencies[3].asset_type == AssetType.MODEL

        # Check if the function is decorated correctly
        settings: LayerSettings = func.layer
        assert settings.get_asset_name() == name
        assert settings.get_asset_type() == AssetType.MODEL
        assert settings.get_pip_packages() == ["scikit-learn==0.23.2"]

    def test_should_complete_remote_train_when_failed_with_runtime_error(self) -> None:
        model_catalog_client = MagicMock(spec=ModelCatalogClient)
        with project_client_mock(model_catalog_client=model_catalog_client):

            @model("foo")
            def create_my_model() -> None:
                raise RuntimeError()

            with pytest.raises(RuntimeError):
                create_my_model()

            model_catalog_client.create_model_train_from_version_id.assert_called_once()
            model_catalog_client.update_model_train_status.assert_called()

    def test_should_complete_remote_train_when_failed_with_memory_error(self) -> None:
        exc_message = "Out of available memory"

        model_catalog_client = MagicMock(spec=ModelCatalogClient)
        with project_client_mock(model_catalog_client=model_catalog_client):

            @model("foo")
            def create_my_model() -> None:
                raise MemoryError(exc_message)

            with pytest.raises(RuntimeMemoryException):
                create_my_model()

            model_catalog_client.create_model_train_from_version_id.assert_called_once()
            model_catalog_client.update_model_train_status.assert_called()

    def test_should_complete_remote_train_when_failed_with_import_error(self) -> None:
        exc_message_pattern = re.compile(".*No module named 'missing_dep'.*")

        model_catalog_client = MagicMock(spec=ModelCatalogClient)
        with project_client_mock(model_catalog_client=model_catalog_client):

            @model("foo")
            def create_my_model() -> None:
                import importlib

                importlib.import_module("missing_dep")

            with pytest.raises(Exception) as exc:
                create_my_model()

            assert exc_message_pattern.match(str(exc))

            model_catalog_client.create_model_train_from_version_id.assert_called_once()
            model_catalog_client.update_model_train_status.assert_called()
