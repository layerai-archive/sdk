import pandas

from layer.decorators import model
from layer.definitions import ModelDefinition
from layer.projects.project_hash_calculator import calculate_project_hash_by_definitions


def test_given_same_function_when_hash_calculated_then_hash_equal():
    # given
    @model("foo")
    def func1():
        return func2

    def func2():
        return pandas.DataFrame({})

    def_1 = ModelDefinition(func=func1, project_name="project-name")
    def_2 = ModelDefinition(func=func1, project_name="project-name")

    # when
    first_hash = calculate_project_hash_by_definitions([def_1, def_1])
    second_hash = calculate_project_hash_by_definitions([def_2, def_2])

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

    def_1 = ModelDefinition(func=func1, project_name="project-name")
    def_2 = ModelDefinition(func=func2, project_name="project-name")

    # when
    first_hash = calculate_project_hash_by_definitions([def_1, def_1])
    second_hash = calculate_project_hash_by_definitions([def_1, def_2])

    # then
    assert first_hash != second_hash
