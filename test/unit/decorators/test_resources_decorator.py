import pathlib

import pytest

from layer.contracts.runs import ResourcePath
from layer.decorators import resources


def test_resources_decorator_combines_all_paths():
    @resources("/", "./abc", "../xyz")
    def func():
        pass

    assert func.layer.get_resource_paths() == [
        ResourcePath(path=pathlib.Path("/")),
        ResourcePath(path=pathlib.Path("./abc")),
        ResourcePath(path=pathlib.Path("../xyz")),
    ]


def test_resources_decorator_requires_at_least_one_path():
    with pytest.raises(
        ValueError, match="resource paths must be a string or a pathlib.Path"
    ):

        @resources
        def func():
            pass
