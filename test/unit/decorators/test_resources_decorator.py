import pytest

from layer.decorators import resources


def test_resources_decorator_combines_all_paths():
    @resources("/", "./abc", "../xyz")
    def func():
        pass

    assert func.layer.get_resource_paths() == ["/", "./abc", "../xyz"]


def test_resources_decorator_requires_at_least_one_path():
    with pytest.raises(ValueError, match="resource paths cannot be empty"):

        @resources
        def func():
            pass
