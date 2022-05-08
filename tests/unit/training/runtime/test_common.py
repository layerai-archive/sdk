import logging
import pickle
from pathlib import Path
from typing import Any

import cloudpickle
import pandas as pd
import pytest
from pandas import DataFrame as pdDataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from layer.exceptions.exceptions import MissingColumnsInDataframeException
from layer.training.runtime.common import check_if_object_is_dataframe, import_function


logger = logging.getLogger(__name__)

TEST_TEMPLATES_PATH = Path(__file__).parent.parent.parent / "assets" / "templates"


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
        X = df.drop(["IsAlone"], axis=1)
        y = df["IsAlone"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        random_forest.fit(X_train, y_train)
        return random_forest

    @staticmethod
    def _pickle_object(python_object: Any, file_object: Any) -> None:
        """
        Covers cases where cloudpickle can use a higher protocol thand the serializing runtime.
        E.g Python 3.7 supporting protocol <= 4 and cloudpickle supporting 5.
        """
        cloudpickle.dump(python_object, file_object, protocol=pickle.HIGHEST_PROTOCOL)

    def test_check_if_object_is_dataframe_checks_columns(self) -> None:
        # Pandas
        pandas_df = pd.DataFrame(
            [["id1", 10], ["id2", 15], ["id3", 14]], columns=["id", "value"]
        )
        check_if_object_is_dataframe(pandas_df)
        with pytest.raises(MissingColumnsInDataframeException):
            pandas_df = pd.DataFrame([["id1"], ["id2"], ["id3"]], columns=["id"])
            check_if_object_is_dataframe(pandas_df)
        with pytest.raises(MissingColumnsInDataframeException):
            pandas_df = pd.DataFrame([[], [], []], columns=[])
            check_if_object_is_dataframe(pandas_df)
