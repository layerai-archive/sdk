import pandas as pd
import pytest

from layer.utils.runtime_utils import check_and_convert_to_df


def test_check_and_convert_to_df_when_passed_df_return_successfully() -> None:
    data = [
        {"a": 0, "b": 1, "c": 2},
        {"d": 3, "c": 4, "b": 5},
        {"a": 5, "d": 6, "e": 1},
    ]
    input = pd.DataFrame.from_records(data)
    assert input is check_and_convert_to_df(input)


def test_check_and_convert_to_df_when_passed_dicts_return_successfully() -> None:
    data = [
        {"a": 0, "b": 1, "c": 2},
        {"d": 3, "c": 4, "b": 5},
        {"a": 5, "d": 6, "e": 1},
    ]
    output = check_and_convert_to_df(data)
    assert isinstance(output, pd.DataFrame)


def test_check_and_convert_to_df_when_passed_dicts_with_wrong_element_then_throw() -> None:
    data = [{"a": 0, "b": 1, "c": 2}, {"d": 3, "c": 4, "b": 5}, ["Wrong element type"]]
    with pytest.raises(Exception, match=".*Expected a list of dict elements.*"):
        check_and_convert_to_df(data)


def test_check_and_convert_to_df_when_passed_empty_df_then_throw() -> None:
    with pytest.raises(Exception, match=".*pandas.DataFrame object has 0 columns.*"):
        check_and_convert_to_df(pd.DataFrame())


def test_check_and_convert_to_df_when_passed_illegal_object_type_then_throw() -> None:
    with pytest.raises(Exception, match=".*Unsupported return type*"):
        check_and_convert_to_df("Unsupported object type")
