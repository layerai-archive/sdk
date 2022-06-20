from typing import Any

import pandas as pd


def check_and_convert_to_df(result: Any) -> pd.DataFrame:
    allowed_types = [pd.DataFrame, list]
    if not any([isinstance(result, x_type) for x_type in allowed_types]):
        raise Exception(
            f"Unsupported return type. Expected one of {allowed_types}, got: {type(result)}"
        )
    if isinstance(result, pd.DataFrame):
        if len(result.columns) == 0:
            raise Exception("pandas.DataFrame object has 0 columns")
        return result
    if isinstance(result, list):
        dict_elements = all([isinstance(list_element, dict) for list_element in result])
        if not dict_elements:
            raise Exception("Expected a list of dict elements")
        return pd.DataFrame.from_records(result)  # type: ignore
    raise Exception(
        f"Unsupported return type. Expected one of {allowed_types}, got: {type(result)}"
    )
