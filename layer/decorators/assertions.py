from typing import Any, Callable, Dict, List

import numpy
import wrapt  # type: ignore

from layer.contracts.assertions import Assertion
from layer.decorators.layer_wrapper import LayerFunctionWrapper
from layer.decorators.utils import ensure_has_layer_settings


def assert_true(assert_function: Callable[..., bool]) -> Callable[..., Any]:
    """
    Asserts that a condition is true.
    This assertion can be used with Layer dataset and model entities.

    :param assert_function: Test function with a boolean return type.
    :return: Function object.

    .. code-block:: python

        from layer.decorators.assertions import assert_true
        from layer.decorators import dataset

        def test_function(data_frame):
            return len(data_frame.index) == 0

        @dataset("empty-data")
        @assert_true(test_function)
        def create_my_dataset():
            return pandas.DataFrame()
    """

    @wrapt.decorator(proxy=_assert_true_wrapper(assert_function))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _assert_true_wrapper(assert_function: Callable[..., bool]) -> Any:
    class FunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any = None) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            self.__wrapped__.layer.append_assertions(
                [assert_true.__name__, assert_function]
            )

    return FunctionWrapper


def _assert_true(assert_function: Callable[..., Any]) -> Callable[[Any], Any]:
    def assert_func(to_check: Any) -> Any:
        assertion_result = assert_function(to_check)
        if not isinstance(assertion_result, (bool, numpy.bool_)):
            raise AssertionError(
                "Test FAILED: assert_true only accepts functions with boolean return type."
            )
        if assertion_result:
            print(f"Test SUCCESS: assert_true({assert_function.__name__}).")
        else:
            raise AssertionError(
                f"Test FAILED: assert_true({assert_function.__name__})."
            )
        return to_check

    return assert_func


def assert_valid_values(
    column_name: str, valid_values: List[Any]
) -> Callable[..., Any]:
    """
    Asserts that only the given valid values present in the given column.
    This assertion can be used with Layer dataset entities.

    :param column_name: Column name to be checked from the given dataframe.
    :param valid_values: Valid values.
    :return: Function object.

    .. code-block:: python

        from layer.decorators.assertions import assert_valid_values
        from layer.decorators import dataset

        @dataset("user-os-data")
        @assert_valid_values("OperatingSystem", ["linux", "other"])
        def create_my_dataset():
            data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
            dataframe = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
            return dataframe
    """

    @wrapt.decorator(proxy=_assert_valid_values_wrapper(column_name, valid_values))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _assert_valid_values_wrapper(column_name: str, valid_values: List[Any]) -> Any:
    class FunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any = None) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            if not isinstance(valid_values, list):
                raise AssertionError(
                    "Test FAILED: assert_valid_values only accepts list type valid_values."
                )
            self.__wrapped__.layer.append_assertions(
                [assert_valid_values.__name__, column_name, valid_values]
            )

    return FunctionWrapper


def _assert_valid_values(
    column_name: str, valid_values: List[Any]
) -> Callable[[Any], Any]:
    def assert_func(result: Any) -> Any:
        existing_unique_values = result[column_name].unique()

        invalid_values = list(set(existing_unique_values) - set(valid_values))
        if not invalid_values:
            print(
                f"Test SUCCESS: assert_valid_values('{column_name}', {valid_values})."
            )
        else:
            raise AssertionError(
                f"Test FAILED: assert_valid_values('{column_name}', {valid_values}). "
                f"Values {invalid_values} not in valid values."
            )

        return result

    return assert_func


def assert_not_null(column_names: List[str]) -> Callable[..., Any]:
    """
    Asserts that given columns do not have any null values.
    This assertion can be used with Layer dataset entities.

    :param column_names: List of columns to be checked.
    :return: Function object.

    .. code-block:: python

        from layer.decorators.assertions import assert_not_null
        from layer.decorators import dataset

        @dataset("user-os-data")
        @assert_not_null(["UserName", "OperatingSystem"])
        def create_my_dataset():
            # No null value in the dataframe. Prints a success message.
            data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
            dataframe = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
            return dataframe
    """

    @wrapt.decorator(proxy=_assert_not_null_wrapper(column_names))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _assert_not_null_wrapper(column_names: List[str]) -> Any:
    class FunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any = None) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            self.__wrapped__.layer.append_assertions(
                [assert_not_null.__name__, column_names]
            )
            if not isinstance(column_names, list) or not all(
                isinstance(x, str) for x in column_names
            ):
                raise AssertionError(
                    "Test FAILED: assert_not_null only accepts string list type column_names."
                )

    return FunctionWrapper


def _assert_not_null(column_names: List[str]) -> Callable[[Any], Any]:
    def assert_func(result: Any) -> Any:
        selected_test_data = result[column_names]
        columns_with_null = selected_test_data.columns[
            selected_test_data.isnull().any()
        ].tolist()

        if not columns_with_null:
            print(f"Test SUCCESS: assert_not_null({column_names}).")
        else:
            raise AssertionError(
                f"Test FAILED: assert_not_null({column_names}). "
                f"Columns {columns_with_null} have null values."
            )
        return result

    return assert_func


def assert_unique(column_subset: List[str]) -> Callable[..., Any]:
    """
    Asserts that given subset of columns together have no duplicate rows.
    This assertion can be used with Layer dataset entities.

    :param column_subset: Subset of the columns to be checked.
    :return: Function object.

    .. code-block:: python

        from layer.decorators.assertions import assert_unique
        from layer.decorators import dataset

        @dataset("user-data")
        @assert_unique(["Id", "UserName", "Company"])
        def create_my_dataset():
            # Row 1 and Row 2 are duplicates. It throws an error.
            data = [[1, "user1", "company1"], [1, "user1", "company1"], [1, "user1", "company2"]]
            dataframe = pd.DataFrame(data, columns=["Id", "UserName", "Company"])
            return dataframe
    """

    @wrapt.decorator(proxy=_assert_unique_wrapper(column_subset))
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _assert_unique_wrapper(column_subset: List[str]) -> Any:
    class FunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any = None) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            self.__wrapped__.layer.append_assertions(
                [assert_unique.__name__, column_subset]
            )
            if not isinstance(column_subset, list) or not all(
                isinstance(x, str) for x in column_subset
            ):
                raise AssertionError(
                    "Test FAILED: assert_unique only accepts string list type column_subset."
                )

    return FunctionWrapper


def _assert_unique(column_subset: List[str]) -> Callable[[Any], Any]:
    def assert_func(result: Any) -> Any:
        if not result.duplicated(subset=column_subset).any():
            print(f"Test SUCCESS: assert_unique({column_subset}).")
        else:
            raise AssertionError(
                f"Test FAILED: assert_unique({column_subset}). "
                f"Columns {column_subset} have duplicates."
            )
        return result

    return assert_func


def assert_skewness(
    column_name: str, min_skewness: float, max_skewness: float
) -> Callable[..., Any]:
    """
    Asserts that skewness value of the given column is in between the given minimum and maximum values.
    This assertion can be used with Layer dataset entities.

    :param column_name: Column to be checked.
    :param min_skewness: Accepted minimum skewness value.
    :param max_skewness: Accepted maximum skewness value.
    :return: Function object.

    .. code-block:: python

        from layer.decorators.assertions import assert_skewness
        from layer.decorators import dataset

        @dataset("product-data")
        @assert_skewness("Price", -0.3, 0.3)
        def create_my_dataset():
            # Assert success as skewness is lower than the threshold.
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            dataframe = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return dataframe
    """

    @wrapt.decorator(
        proxy=_assert_skewness_wrapper(column_name, min_skewness, max_skewness)
    )
    def wrapper(
        wrapped: Any, instance: Any, args: List[Any], kwargs: Dict[str, Any]
    ) -> None:
        return wrapped(*args, **kwargs)

    return wrapper


def _assert_skewness_wrapper(
    column_name: str, min_skewness: float, max_skewness: float
) -> Any:
    class FunctionWrapper(LayerFunctionWrapper):
        def __init__(self, wrapped: Any, wrapper: Any, enabled: Any = None) -> None:
            super().__init__(wrapped, wrapper, enabled)
            ensure_has_layer_settings(self.__wrapped__)
            self.__wrapped__.layer.append_assertions(
                [assert_skewness.__name__, column_name, min_skewness, max_skewness]
            )
            if (
                not isinstance(column_name, str)
                or not isinstance(min_skewness, (int, float))
                or not isinstance(max_skewness, (int, float))
            ):
                raise AssertionError(
                    "assert_skewness only accepts string type column_name and "
                    "numeric type min_skewness and max_skewness values."
                )

    return FunctionWrapper


def _assert_skewness(
    column_name: str, min_skewness: float, max_skewness: float
) -> Callable[[Any], Any]:
    def assert_func(result: Any) -> Any:
        skewness_value = result[column_name].skew(axis=0)
        in_range = max_skewness > skewness_value > min_skewness

        if in_range:
            print(
                f"Test SUCCESS: assert_skewness('{column_name}', {min_skewness}, {max_skewness})."
            )
        else:
            raise AssertionError(
                f"Test FAILED: assert_skewness('{column_name}', {min_skewness}, {max_skewness}). "
                f"Skewness value [{skewness_value}] of {column_name} column is not in the given range."
            )
        return result

    return assert_func


def get_asserted_function(function: Callable[..., Any]) -> Callable[..., Any]:
    asserted_function = function
    assertion_list = _get_assertion_list(function)
    for assertion in assertion_list:
        name = assertion[0]
        values = assertion[1:]
        assertion_function = _assertion_name_function_mapper(name, values)
        asserted_function = _call_function_with_assertion(
            asserted_function, assertion_function
        )

    return asserted_function


def get_assertion_functions_data(function: Callable[..., Any]) -> List[Assertion]:
    assertion_list = _get_assertion_list(function)
    result = []
    for assertion in assertion_list:
        name = assertion[0]
        values = assertion[1:]
        assertion_function = _assertion_name_function_mapper(name, values)
        result.append(Assertion(name, values, assertion_function))

    return result


def _assertion_name_function_mapper(name: str, values: List[Any]) -> Callable[..., Any]:
    if name == assert_true.__name__ and len(values) == 1:
        return _assert_true(values[0])
    elif name == assert_valid_values.__name__ and len(values) == 2:
        return _assert_valid_values(values[0], values[1])
    elif name == assert_not_null.__name__ and len(values) == 1:
        return _assert_not_null(values[0])
    elif name == assert_unique.__name__ and len(values) == 1:
        return _assert_unique(values[0])
    elif name == assert_skewness.__name__ and len(values) == 3:
        return _assert_skewness(values[0], values[1], values[2])
    else:
        raise ValueError(f"Invalid assertion function: {name}")


def _call_function_with_assertion(
    function: Callable[..., Any], assertion: Callable[..., Any]
) -> Callable[..., Any]:
    def with_assertion() -> Callable[..., Any]:
        return assertion(function())

    return with_assertion


def _get_assertion_list(function: Callable[..., Any]) -> List[Any]:
    layer_settings = function.__dict__.get("layer")
    if not layer_settings:
        return []
    return layer_settings.get_assertions()
