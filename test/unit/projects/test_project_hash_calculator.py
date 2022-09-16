from typing import Any, Callable

import pandas

from layer.contracts.asset import AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.decorators import model
from layer.projects.utils import calculate_hash_by_definitions


def test_given_same_function_when_hash_calculated_then_hash_equal():
    # given
    @model("foo")
    def func1():
        return func2

    def func2():
        return pandas.DataFrame({})

    def_1 = _make_test_function_definition(func=func1)
    def_2 = _make_test_function_definition(func=func1)

    # when
    first_hash = calculate_hash_by_definitions([def_1, def_1])
    second_hash = calculate_hash_by_definitions([def_2, def_2])

    # then
    assert first_hash == second_hash


def test_given_different_function_when_hash_calculated_then_hash_different():
    # given
    @model("foo")
    def func1():
        return func3()

    @model("bar")
    def func2():
        return func3()

    def func3():
        return pandas.DataFrame({})

    def_1 = _make_test_function_definition(func=func1)
    def_2 = _make_test_function_definition(func=func2)

    # when
    first_hash = calculate_hash_by_definitions([def_1, def_1])
    second_hash = calculate_hash_by_definitions([def_1, def_2])

    # then
    assert first_hash != second_hash


def _make_test_function_definition(
    func: Callable[..., Any],
) -> FunctionDefinition:
    return FunctionDefinition(
        func=func,
        args=tuple(),
        kwargs={},
        asset_type=AssetType.MODEL,
        asset_name=func.layer.get_asset_name(),
        fabric=Fabric.F_LOCAL,
        asset_dependencies=[],
        pip_dependencies=[],
        conda_env=None,
        resource_paths=[],
        assertions=[],
    )
