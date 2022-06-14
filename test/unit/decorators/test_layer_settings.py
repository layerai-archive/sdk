import pytest

from layer.contracts.assets import AssetType
from layer.contracts.fabrics import Fabric
from layer.decorators import dataset, fabric, pip_requirements
from layer.exceptions.exceptions import ConfigError
from layer.decorators.settings import LayerSettings


settings = LayerSettings()


def test_raises_error_on_attempt_to_set_fabric_setting_to_invalid():
    with pytest.raises(
        ValueError,
        match='Fabric setting "this-is-invalid" is not valid. You can check valid values in Fabric enum definition.',
    ):
        settings.set_fabric("this-is-invalid")


def test_fabric_can_be_saved_and_retrieved_as_string():
    settings.set_fabric("f-small")
    assert settings.get_fabric() == Fabric.F_SMALL


def test_settings_from_multiple_decorators_are_cumulated():
    @dataset("test")
    @fabric(Fabric.F_MEDIUM.value)
    @pip_requirements(file="test.txt")
    def test():
        pass

    assert test.layer.get_fabric() == Fabric.F_MEDIUM
    assert test.layer.get_asset_name() == "test"
    assert test.layer.get_pip_requirements_file() == "test.txt"
    assert test.layer.get_asset_type() == AssetType.DATASET


def test_doesnt_validate_dataset_with_gpu_fabric():
    with pytest.raises(
        ConfigError,
        match="GPU fabrics can only be used for model training. Use a different fabric for your dataset build.",
    ):
        settings.set_fabric(Fabric.F_GPU_SMALL)
        settings.set_asset_name("my_dataset")
        settings.set_asset_type(AssetType.DATASET)
        settings.validate()
