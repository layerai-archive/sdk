import pickle
import uuid
from pathlib import Path
from test.unit.decorators.util import project_client_mock
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID

import pandas as pd
import pytest
from layerapi.api.ids_pb2 import DatasetBuildId
from layerapi.api.service.datacatalog.data_catalog_api_pb2 import InitiateBuildResponse

from layer.client import DataCatalogClient
from layer.data_classes import Dataset, Fabric, Model
from layer.decorators import dataset, fabric, pip_requirements
from layer.definitions import DatasetDefinition
from layer.exceptions.exceptions import (
    ConfigError,
    LayerClientResourceNotFoundException,
    ProjectInitializationException,
)
from layer.global_context import set_current_project_name, set_default_fabric
from layer.projects.asset import AssetType


@dataset("foo1", dependencies=[Dataset("bar"), Model("zoo")])
@pip_requirements(packages=["sklearn==0.0"])
def f1():
    return pd.DataFrame()


@dataset("foo2", dependencies=[Dataset("bar"), Model("zoo")])
@pip_requirements(packages=["sklearn==0.0"])
def f2():
    return pd.DataFrame()


class TestDatasetDecorator:
    def test_dataset_decorator_assigns_attributes_to_function_before_calling_function(
        self,
    ):
        @dataset("foo")
        def create_my_dataset():
            return pd.DataFrame()

        assert create_my_dataset.layer.get_entity_name() == "foo"
        assert create_my_dataset.layer.get_asset_type() == AssetType.DATASET

    def test_dataset_decorator_assigns_attributes_to_function_before_calling_bound_function(
        self,
    ):
        class MyClass:
            @dataset("foo")
            def create_my_dataset(self):
                return pd.DataFrame()

        assert MyClass.create_my_dataset.layer.get_entity_name() == "foo"
        assert MyClass.create_my_dataset.layer.get_asset_type() == AssetType.DATASET

        my_class = MyClass()
        assert my_class.create_my_dataset.layer.get_entity_name() == "foo"
        assert my_class.create_my_dataset.layer.get_asset_type() == AssetType.DATASET

    def test_dataset_decorator_given_no_current_project_set_raise_exception(self):
        set_current_project_name(None)

        @dataset("foo1")
        def create_my_dataset():
            return pd.DataFrame()

        with pytest.raises(
            ProjectInitializationException,
            match="Please specify the current project name globally with",
        ):
            create_my_dataset()

    def test_dataset_decorator_given_set_project_does_not_exist_raise_exception(self):
        mock_project_api = MagicMock()
        mock_project_api.GetProjectByName.side_effect = (
            LayerClientResourceNotFoundException()
        )

        with project_client_mock(project_api_stub=mock_project_api):

            @dataset("foo2")
            def create_my_dataset():
                return pd.DataFrame()

            with pytest.raises(
                ProjectInitializationException,
                match="Project with the name foo-test does not exist.",
            ):
                set_current_project_name("foo-test")
                create_my_dataset()

    @pytest.mark.parametrize(
        ("name", "function"),
        [
            ("foo1", f1),
            ("foo2", f2),
        ],
    )
    def test_dataset_definition_created_correctly(self, name, function):
        set_current_project_name("foo-test")

        with project_client_mock(), patch(
            "layer.decorators.dataset_decorator._build_locally_update_remotely",
            return_value=("", UUID(int=0x12345678123456781234567812345678)),
        ), patch(
            "layer.decorators.dataset_decorator.register_derived_datasets"
        ) as mock_register_datasets:
            func = function
            func()

            mock_register_datasets.assert_called_with(ANY, ANY, ANY, ANY)
            (
                client,
                current_project_uuid,
                python_definition,
                tracker,
            ) = mock_register_datasets.call_args[0]

            assert python_definition.name == name
            assert python_definition.project_name == "foo-test"
            assert python_definition.entrypoint_path.exists()
            assert python_definition.environment_path.exists()
            assert (
                Path(python_definition.environment_path).read_text() == "sklearn==0.0\n"
            )
            assert len(python_definition.dependencies) == 2
            assert python_definition.dependencies[0].name == "bar"
            assert isinstance(python_definition.dependencies[0], Dataset)
            assert python_definition.dependencies[1].name == "zoo"
            assert isinstance(python_definition.dependencies[1], Model)

            # Check if the unpickled file contains the correct function
            loaded = pickle.load(open(python_definition.entrypoint_path, "rb"))
            assert loaded.layer.get_entity_name() == name
            assert loaded.layer.get_asset_type() == AssetType.DATASET
            assert loaded.layer.get_pip_packages() == ["sklearn==0.0"]

    def test_should_complete_remote_build_when_failed(self):
        data_catalog_client = MagicMock(spec=DataCatalogClient)
        data_catalog_client.initiate_build.return_value = InitiateBuildResponse(
            id=DatasetBuildId(value=str(uuid.uuid4()))
        )

        with patch(
            "layer.decorators.dataset_decorator.register_derived_datasets",
            return_value=Dataset(asset_path="test"),
        ), project_client_mock(data_catalog_client=data_catalog_client):

            @dataset("foo")
            def create_my_dataset():
                raise RuntimeError()

            with pytest.raises(RuntimeError):
                create_my_dataset()

            data_catalog_client.initiate_build.assert_called_once()
            data_catalog_client.complete_build.assert_called_once()

    def test_given_fabric_override_uses_it_over_default(self):
        set_default_fabric(Fabric.F_SMALL)
        set_current_project_name("test-project")

        dataset_def = MagicMock(spec=DatasetDefinition)

        with patch(
            "layer.definitions.DatasetDefinition.__new__", return_value=dataset_def
        ), project_client_mock(), patch(
            "layer.decorators.dataset_decorator._build_dataset_locally_and_store_remotely"
        ) as mock_build_locally:

            @dataset("test")
            @fabric(Fabric.F_MEDIUM.value)
            def create_my_dataset():
                return pd.DataFrame()

            @dataset("test-2")
            def create_another_dataset():
                return pd.DataFrame()

            create_my_dataset()
            create_another_dataset()

            (
                func,
                settings,
                ds,
                tracker,
                client,
                assertions,
            ) = mock_build_locally.call_args_list[0][0]
            assert settings.get_fabric() == Fabric.F_MEDIUM

            (
                func,
                settings,
                ds,
                tracker,
                client,
                assertions,
            ) = mock_build_locally.call_args_list[1][0]
            assert settings.get_fabric() == Fabric.F_SMALL

    def test_not_named_dataset_cannot_be_run_even_locally(self):
        @dataset("")
        def func():
            pass

        with pytest.raises(
            ConfigError,
            match="^Your @dataset and @model must be named. Pass an entity name as a first argument to your decorators.$",
        ):
            func()
