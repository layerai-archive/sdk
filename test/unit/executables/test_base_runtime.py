from pathlib import Path

import pandas as pd
from pandas import DataFrame as pdDataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from layer.executables.packager import package_function
from layer.executables.runtime import BaseFunctionRuntime


def test_import_function_works_with_pickled_pandas_df_function(tmp_path: Path) -> None:
    executable_path = package_function(
        _pickle_function_pandas_dataframe, output_dir=tmp_path
    )
    runtime = BaseFunctionRuntime(executable_path)

    result = runtime.run_executable()

    assert len(result) == 3


def test_import_function_works_with_pickled_pandas_ml_model_function(
    tmp_path: Path,
) -> None:
    executable_path = package_function(
        _pickle_function_pandas_ml_model, output_dir=tmp_path
    )
    runtime = BaseFunctionRuntime(executable_path)

    result = runtime.run_executable()
    test_df = pd.DataFrame([[35, 1]], columns=["AgeBand", "EmbarkStatus"])

    assert len(result.predict(test_df)) == 1


def _pickle_function_pandas_dataframe() -> pdDataFrame:
    return pd.DataFrame(
        [[35, 1, 0], [36, 2, 1], [37, 3, 2]],
        columns=["AgeBand", "EmbarkStatus", "IsAlone"],
    )


def _pickle_function_pandas_ml_model() -> RandomForestClassifier:
    random_forest = RandomForestClassifier(n_estimators=1)
    df = pd.DataFrame(
        [[35, 1, 0], [36, 2, 1], [37, 3, 2]],
        columns=["AgeBand", "EmbarkStatus", "IsAlone"],
    )
    x = df.drop(["IsAlone"], axis=1)
    y = df["IsAlone"]
    x_train, unused_x_test, y_train, unused_y_test = train_test_split(
        x, y, test_size=0.5
    )
    random_forest.fit(x_train, y_train)
    return random_forest
