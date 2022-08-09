from pathlib import Path

import pandas as pd
import pytest

from layer.decorators import conda, dataset


def test_conda_decorator_assigns_environment_to_function_before_calling_function():
    @conda(environment={"name": "test", "dependencies": ["tensorflow"]})
    @dataset("foo")
    def create_my_dataset():
        return pd.DataFrame()

    conda_env = create_my_dataset.layer.conda_environment
    assert conda_env
    assert conda_env.environment == {"name": "test", "dependencies": ["tensorflow"]}


def test_conda_decorator_assigns_files_to_function_before_calling_function():
    assets = Path("test") / "unit" / "decorators" / "assets"

    @conda(environment_file=assets / "environment.yml")
    @dataset("foo")
    def create_my_dataset():
        return pd.DataFrame()

    conda_env = create_my_dataset.layer.conda_environment
    assert conda_env
    assert conda_env.environment == {
        "name": "stats2",
        "channels": ["javascript"],
        "dependencies": [
            "python=3.9",
            "bokeh=2.4.2",
            "numpy=1.21.*",
            "nodejs=16.13.*",
            "flask",
            "pip",
            {"pip": ["Flask-Testing"]},
        ],
    }


def test_conda_decorator_raises_exception_if_file_and_environment_dict_provided():
    with pytest.raises(
        ValueError,
        match="either environment_file or environment dictionary should be provided, not both.",
    ):

        @conda(
            environment_file="/path/to/environment.yml", environment={"name": "test"}
        )
        @dataset("foo")
        def create_my_dataset():
            return pd.DataFrame()
