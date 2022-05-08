import pandas as pd
import pytest

from layer.decorators import dataset, pip_requirements


def test_pip_decorator_assigns_packages_to_function_before_calling_function():
    @pip_requirements(packages=["sklearn==0.0"])
    @dataset("foo")
    def create_my_dataset():
        return pd.DataFrame()

    assert create_my_dataset.layer.get_pip_packages() == ["sklearn==0.0"]


def test_pip_decorator_assigns_files_to_function_before_calling_function():
    @pip_requirements(file="/path/to/requirements.txt")
    @dataset("foo")
    def create_my_dataset():
        return pd.DataFrame()

    assert (
        create_my_dataset.layer.get_pip_requirements_file()
        == "/path/to/requirements.txt"
    )


def test_pip_decorator_raises_exception_if_file_and_package_provided():
    with pytest.raises(
        ValueError,
        match="either file or packages should be provided, not both",
    ):

        @pip_requirements(file="/path/to/requirements.txt", packages=["sklearn==0.0"])
        @dataset("foo")
        def create_my_dataset():
            return pd.DataFrame()
