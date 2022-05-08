from functools import wraps

import pytest

from layer.decorators import fabric
from layer.fabric import Fabric


def test_decorated_function_can_be_called():
    @fabric(Fabric.F_SMALL.value)
    def sample_function():
        return 56.78

    assert sample_function() == 56.78


def test_fabric_setting_is_set_on_function_even_if_its_not_called():
    @fabric(Fabric.F_XSMALL.value)
    def another_sample():
        pass

    assert another_sample.layer.get_fabric() == Fabric.F_XSMALL


def test_fabric_setting_can_be_retrieved_by_top_level_decorator():
    recorded_fabric_value = None

    def top_level_decorator_for_func(func):
        nonlocal recorded_fabric_value
        recorded_fabric_value = func.layer.get_fabric()

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)

        return wrapper

    @top_level_decorator_for_func
    @fabric(Fabric.F_SMALL.value)
    def sample_func():
        pass

    assert recorded_fabric_value == Fabric.F_SMALL


def test_raises_error_on_attempt_to_set_fabric_to_invalid_value():
    with pytest.raises(
        ValueError,
        match='Fabric setting "f-invalid" is not valid. You can check valid values in Fabric enum definition.',
    ):

        @fabric("f-invalid")
        def invalid_decorator_arg():
            return "test"


def test_decorated_method_can_be_called():
    recorded_fabric_value = None

    def top_level_decorator_for_obj_method(method):
        nonlocal recorded_fabric_value
        recorded_fabric_value = Fabric.F_GPU_SMALL

        @wraps(method)
        def inner(self, *args, **kwargs):
            return method(self, *args, **kwargs)

        return inner

    class SampleClass:
        content = "test"

        @top_level_decorator_for_obj_method
        @fabric(Fabric.F_GPU_SMALL.value)
        def bound_function(self):
            return self.content

    instance = SampleClass()
    assert recorded_fabric_value == Fabric.F_GPU_SMALL
    assert instance.bound_function.layer.get_fabric() == Fabric.F_GPU_SMALL
    assert instance.bound_function() == "test"
    assert instance.bound_function.layer.get_fabric() == Fabric.F_GPU_SMALL
