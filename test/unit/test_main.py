import re
from unittest.mock import patch

import pytest

import layer
from layer.contracts.fabrics import Fabric
from layer.decorators import fabric
from layer.exceptions.exceptions import ConfigError
from layer.main import init


def test_when_both_packages_and_requirements_file_provided_on_init_then_throws_error() -> None:
    # given
    pip_packages = ["sklearn==0.0"]
    pip_requirements_file_name = "req.txt"
    # when
    with pytest.raises(
        ValueError,
        match="either pip_requirements_file or pip_packages should be provided, not both",
    ):
        init(
            "test",
            pip_packages=pip_packages,
            pip_requirements_file=pip_requirements_file_name,
        )


class TestRunValidation:
    def test_when_no_top_level_decorator_config_error_is_raised(self) -> None:
        @fabric(Fabric.F_MEDIUM.value)
        def sample_function() -> None:
            pass

        with pytest.raises(
            ConfigError,
            match=re.escape(
                'Either @dataset(name="...") or @model(name="...") top level decorator '
                "is required for each function. Add @dataset or @model decorator on top "
                "of existing decorators to run functions in Layer."
            ),
        ):
            layer.run([sample_function])

    def test_when_not_decorated_function_config_error_is_raised(self) -> None:
        def test() -> None:
            pass

        with pytest.raises(
            ConfigError,
            match=re.escape(
                'Either @dataset(name="...") or @model(name="...") is required for each function. '
                "Add @dataset or @model decorator to your functions to run them in Layer."
            ),
        ):
            layer.run([test])


def test_clear_cache() -> None:
    with patch("layer.cache.cache.Cache.clear") as cache:
        layer.clear_cache()
        cache.assert_called_once_with()
