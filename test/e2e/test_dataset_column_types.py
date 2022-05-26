import pandas as pd
from pandas.testing import assert_frame_equal

import layer
from layer.contracts.projects import Project
from layer.decorators import dataset
from test.e2e.assertion_utils import E2ETestAsserter


def test_remote_run_with_supported_column_types_succeeds_and_registers_metadata(
    initialized_project: Project, asserter: E2ETestAsserter
):
    # given
    dataset_name = "types_dataset"

    @dataset(dataset_name)
    def prepare_data():
        import datetime

        import numpy as np

        pandas_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "bool": [True, False, True],
                "string": ["チキンカレー", "string2", "string3"],
                "int8": np.array([10, 11, 12], dtype=np.dtype(np.int8)),
                "int16": np.array([13, 14, 15], dtype=np.dtype(np.int16)),
                "int32": np.array([16, 17, 18], dtype=np.dtype(np.int32)),
                "int64": np.array([19, 20, 21], dtype=np.dtype(np.int64)),
                "float32": np.array([22, 23, 24], dtype=np.dtype(np.float32)),
                "float64": np.array([25, 26, 27], dtype=np.dtype(np.float64)),
                "date": [
                    datetime.date(2022, 4, 5),
                    datetime.date(2022, 4, 6),
                    datetime.date(2022, 4, 7),
                ],
                "time": [
                    datetime.time(3, 45, 12, 2),
                    datetime.time(3, 45, 12, 1),
                    datetime.time(3, 45, 12, 3),
                ],
                "int64_arr": [
                    np.array([-1, -2, 0, 1, 2, 42], dtype=np.dtype(np.int64)),
                    np.array(
                        [9_223_372_036_854_775_807, 43, -1, 7, 7, 42],
                        dtype=np.dtype(np.int64),
                    ),
                    np.array([9_223_372_036_854_775_807], dtype=np.dtype(np.int64)),
                ],
                "int32_arr": [
                    np.array([-1, 56, 0, 1, 2, 42], dtype=np.dtype(np.int32)),
                    np.array([-1, -2, 0, 8_900_32, 2, 42], dtype=np.dtype(np.int32)),
                    np.array([], dtype=np.dtype(np.int32)),
                ],
                "int16_arr": [
                    np.array([-1, 0, 1, 41], dtype=np.dtype(np.int16)),
                    np.array([-1, 0, 1, 42], dtype=np.dtype(np.int16)),
                    np.array([-1, 0, 1, 43], dtype=np.dtype(np.int16)),
                ],
                "int8_arr": [
                    np.array([-1, 0, 1, 120], dtype=np.dtype(np.int8)),
                    np.array([-1, 0, 1, 121], dtype=np.dtype(np.int8)),
                    np.array([-1, 0, 1, 122], dtype=np.dtype(np.int8)),
                ],
                "float64_arr": [
                    np.array(
                        [-0.1, -0.2, 0.0, 1.98, 1.7976931348623157e308, 0.841],
                        dtype=np.dtype(np.float64),
                    ),
                    np.array(
                        [-0.1, -0.2, 0.0, 1.98, 100.87, 0.842],
                        dtype=np.dtype(np.float64),
                    ),
                    np.array(
                        [-0.1, -0.2, 0.0, 1.98, 100.87, 0.843],
                        dtype=np.dtype(np.float64),
                    ),
                ],
                "float32_arr": [
                    np.array(
                        [-0.1, -0.2, 0.0, 1.98, 1.797, 0.41, 1],
                        dtype=np.dtype(np.float32),
                    ),
                    np.array(
                        [-0.1, -0.2, 0.0, 1.98, 100.87, 0.42, 2],
                        dtype=np.dtype(np.float32),
                    ),
                    np.array(
                        [-0.1, -0.2, 0.0, 1.98, 100.87, 0.43, 3],
                        dtype=np.dtype(np.float32),
                    ),
                ],
                "str_arr": [
                    np.array(["1", "a", "b", "チキンカレー"], dtype=np.dtype(np.str_)),
                    np.array(["2", "a", "b", "チキンカレー"], dtype=np.dtype(np.str_)),
                    np.array(["3", "a", "b", "チキンカレー"], dtype=np.dtype(np.str_)),
                ],
                "empty_and_none_arr": [
                    np.array(["x"], dtype=np.dtype(np.str_)),
                    np.array([], dtype=np.dtype(np.str_)),
                    None,
                ],
            }
        )
        return pandas_df

    # when
    run = layer.run([prepare_data])

    # then
    asserter.assert_run_succeeded(run.run_id)

    types_ds = layer.get_dataset(dataset_name)
    types_pandas = types_ds.to_pandas()

    assert_frame_equal(prepare_data(), types_pandas)
