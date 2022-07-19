import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame as pdDataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import layer.cloudpickle as cloudpickle
from layer.training.runtime.common import import_function


logger = logging.getLogger(__name__)


class TestCommon:
    def test_import_function_works_with_pickled_pandas_df_function(
        self, tmp_path: Path
    ) -> None:
        with (tmp_path / "pandas_df.pickle").open("wb") as f:
            self._pickle_object(self._pickle_function_pandas_dataframe, f)
        imported_func = import_function(tmp_path, "pandas_df.pickle")
        assert len(imported_func()) == 3

    def test_import_function_works_with_pickled_pandas_ml_model_function(
        self, tmp_path: Path
    ) -> None:
        with (tmp_path / "pandas_ml_model.pickle").open("wb") as f:
            self._pickle_object(self._pickle_function_pandas_ml_model, f)
        imported_func = import_function(tmp_path, "pandas_ml_model.pickle")
        test_df = pd.DataFrame([[35, 1]], columns=["AgeBand", "EmbarkStatus"])
        imported_func().predict(test_df)  # Should not throw errors

    def _pickle_function_pandas_dataframe(self) -> pdDataFrame:
        return pd.DataFrame(
            [[35, 1, 0], [36, 2, 1], [37, 3, 2]],
            columns=["AgeBand", "EmbarkStatus", "IsAlone"],
        )

    def _pickle_function_pandas_ml_model(self) -> RandomForestClassifier:
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

    @staticmethod
    def _pickle_object(python_object: Any, file_object: Any) -> None:
        """
        Covers cases where cloudpickle can use a higher protocol thand the serializing runtime.
        E.g Python 3.7 supporting protocol <= 4 and cloudpickle supporting 5.
        """
        cloudpickle.dump(python_object, file_object, protocol=pickle.HIGHEST_PROTOCOL)
