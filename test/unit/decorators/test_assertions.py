from typing import Any, Callable, List
from unittest.mock import ANY, MagicMock, patch

import numpy
import pandas as pd
import pytest

from layer.contracts.assertions import Assertion
from layer.decorators.assertions_decorator import (
    assert_not_null,
    assert_skewness,
    assert_true,
    assert_unique,
    assert_valid_values,
)
from layer.decorators.settings import LayerSettings


def get_asserted_function(function: Callable[..., Any]) -> Callable[..., Any]:
    asserted_function = function
    settings: LayerSettings = function.layer  # type:ignore
    for assertion in settings.get_assertions():
        asserted_function = _call_function_with_assertion(
            asserted_function, assertion.function
        )

    return asserted_function


def _call_function_with_assertion(
    function: Callable[..., Any], assertion: Callable[..., Any]
) -> Callable[..., Any]:
    def with_assertion() -> Callable[..., Any]:
        return assertion(function())

    return with_assertion


class TestAssertions:
    def test_given_condition_true_assert_true_returns_success(self):
        method = MagicMock()
        method.__name__ = "test_method"
        method.return_value = True

        @assert_true(method)
        def f1():
            return pd.DataFrame()

        get_asserted_function(f1)()
        method.assert_called_with(ANY)

    def test_given_condition_true_with_models_assert_true_returns_success(self):
        method = MagicMock()
        method.__name__ = "test_method"
        method.return_value = True
        method_2 = MagicMock()
        method_2.__name__ = "test_method_2"
        method_2.return_value = numpy.bool_(True)

        @assert_true(method_2)
        @assert_true(method)
        def f1():
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier()

        get_asserted_function(f1)()
        method.assert_called_with(ANY)
        method_2.assert_called_with(ANY)

    def test_given_test_with_invalid_return_type_assert_true_returns_error_message(
        self,
    ):
        method = MagicMock()
        method.__name__ = "test_method"
        method.return_value = "string_type"

        @assert_true(method)
        def f1():
            return pd.DataFrame()

        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_true only accepts functions with boolean return type.",
        ):
            get_asserted_function(f1)()

    def test_given_test_fails_assert_true_returns_error_message(self):
        method = MagicMock()
        method.__name__ = "test_method"
        method.return_value = False

        @assert_true(method)
        def f1():
            return pd.DataFrame()

        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_true\(test_method\).",
        ):
            get_asserted_function(f1)()

    def test_given_condition_true_assert_valid_values_returns_success(self):
        @assert_valid_values("OperatingSystem", ["linux", "other"])
        def f1():
            data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
            df = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
            return df

        get_asserted_function(f1)()

    def test_given_test_with_non_list_type_assert_valid_values_returns_error_message(
        self,
    ):
        with pytest.raises(
            AssertionError,
            match=r".*Test FAILED: assert_valid_values only accepts list type valid_values.*",
        ):

            @assert_valid_values("OperatingSystem", "linux")
            def f1():
                data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
                df = pd.DataFrame(data, columns=["Name", "Age"])
                return df

    def test_given_test_fails_assert_valid_values_returns_error_message(self):
        @assert_valid_values("OperatingSystem", ["linux"])
        def f1():
            data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
            df = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
            return df

        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_valid_values\(\'OperatingSystem\', \[\'linux\'\]\). "
            r"Values \[\'other\'\] not in valid values.",
        ):
            get_asserted_function(f1)()

    def test_given_condition_true_assert_not_null_returns_success(self):
        @assert_not_null(["UserName", "OperatingSystem"])
        def f1():
            data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
            df = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
            return df

        get_asserted_function(f1)()

    def test_given_test_with_non_string_list_type_assert_not_null_returns_error_message(
        self,
    ):
        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_not_null only accepts string list type column_names.",
        ):

            @assert_not_null("OperatingSystem")
            def f1():
                data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
                df = pd.DataFrame(data, columns=["Name", "Age"])
                return df

            f1()

    def test_given_test_fails_assert_not_null_returns_error_message(self):
        @assert_not_null(["UserName", "OperatingSystem"])
        def f1():
            data = [["user1", "linux"], ["user2"], ["user3", "other"]]
            df = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
            return df

        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_not_null\(\[\'UserName\', \'OperatingSystem\'\]\). "
            r"Columns \[\'OperatingSystem\'\] have null values.",
        ):
            get_asserted_function(f1)()

    def test_given_condition_true_assert_unique_returns_success(self):
        @assert_unique(["UserName", "OperatingSystem"])
        def f1():
            data = [[1, "user1", "linux"], [1, "user1", "other"]]
            df = pd.DataFrame(data, columns=["Id", "UserName", "OperatingSystem"])
            return df

        get_asserted_function(f1)()

    def test_given_test_with_non_string_list_type_assert_unique_returns_error_message(
        self,
    ):
        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_unique only accepts string list type column_subset.",
        ):

            @assert_unique("OperatingSystem")
            def f1():
                data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
                df = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
                return df

            f1()

    def test_given_test_fails_assert_unique_returns_error_message(self):
        @assert_unique(["UserName", "OperatingSystem"])
        def f1():
            data = [[1, "user1", "linux"], [2, "user1", "linux"], [3, "user2", "other"]]
            df = pd.DataFrame(data, columns=["Id", "UserName", "OperatingSystem"])
            return df

        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_unique\(\[\'UserName\', \'OperatingSystem\'\]\). "
            r"Columns \[\'UserName\', \'OperatingSystem\'\] have duplicates.",
        ):
            get_asserted_function(f1)()

    def test_given_test_fails_with_nonexisting_column_assert_unique_returns_error_message(
        self,
    ):
        @assert_unique(["UserName", "NonExisting"])
        def f1():
            data = [[1, "user1", "linux"], [2, "user1", "linux"], [3, "user2", "other"]]
            df = pd.DataFrame(data, columns=["Id", "UserName", "OperatingSystem"])
            return df

        with pytest.raises(
            KeyError,
            match=r"Index\(\[\'NonExisting\'\], dtype=\'object\'\)",
        ):
            get_asserted_function(f1)()

    def test_given_condition_true_assert_skewness_returns_success(self):
        @assert_skewness("Price", -0.5, 0.5)
        def f1():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
            df = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return df

        get_asserted_function(f1)()

    def test_given_test_with_invalid_type_assert_skewness_returns_error_message(
        self,
    ):

        with pytest.raises(
            AssertionError,
            match=".*assert_skewness only accepts string type column_name and "
            "numeric type min_skewness and max_skewness values.*",
        ):

            @assert_skewness("Price", "1", 1.5)
            def f1():
                data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 10]]
                df = pd.DataFrame(data, columns=["Id", "Product", "Price"])
                return df

            get_asserted_function(f1)()

    def test_given_test_fails_assert_skewness_returns_error_message(self):
        @assert_skewness("Price", 0.5, 1)
        def f1():
            data = [[1, "product1", 15], [2, "product2", 20], [3, "product3", 50]]
            df = pd.DataFrame(data, columns=["Id", "Product", "Price"])
            return df

        with pytest.raises(
            AssertionError,
            match=r"Test FAILED: assert_skewness\(\'Price\', 0.5, 1\). "
            r"Skewness value \[1.5970969928697372\] of Price column is not in the given range.",
        ):
            get_asserted_function(f1)()

    def test_get_assertion_list(self):
        @assert_unique(["UserName", "NonExisting"])
        @assert_skewness("Price", -0.5, 0.5)
        def f1():
            return pd.DataFrame()

        assertions: List[Assertion] = f1.__dict__.get("layer").get_assertions()
        assert len(assertions) == 2

    def test_call_function_with_assertion(self):
        @assert_unique(["UserName", "NonExisting"])
        def f1():
            return pd.DataFrame()

        assertions: List[Assertion] = f1.__dict__.get("layer").get_assertions()

        assert assertions[0].name == "assert_unique"
        assert assertions[0].values == [["UserName", "NonExisting"]]

    def test_get_asserted_function(self):

        with patch(
            "layer.decorators.assertions_decorator._assert_unique"
        ) as mocked_unique, patch(
            "layer.decorators.assertions_decorator._assert_not_null"
        ) as mocked_not_null:
            mocked_unique.__name__ = "_assert_unique"
            mocked_not_null.__name__ = "_assert_not_null"

            @assert_unique(["UserName"])
            @assert_not_null(["UserName", "OperatingSystem"])
            def f1():
                data = [["user1", "linux"], ["user2", "linux"], ["user3", "other"]]
                df = pd.DataFrame(data, columns=["UserName", "OperatingSystem"])
                return df

            asserted_function = get_asserted_function(f1)
            asserted_function()
        mocked_unique.assert_called_with(["UserName"])
        mocked_not_null.assert_called_with(["UserName", "OperatingSystem"])
