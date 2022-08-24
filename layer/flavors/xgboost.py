from pathlib import Path

import numpy as np
import pandas as pd
from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor
from scipy.sparse import spmatrix  # type: ignore

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


class XGBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of XGBoost Models.

    Uses XGBoost model (an instance of xgboost.Booster).

    """

    MODULE_KEYWORD = "xgboost"
    PROTO_FLAVOR = PbModelFlavor.MODEL_FLAVOR_XGBOOST

    def save_model_to_directory(
        self, model_object: ModelObject, directory: Path
    ) -> None:
        import mlflow.xgboost

        mlflow.xgboost.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        import mlflow.xgboost

        model = mlflow.xgboost.load_model(directory.as_uri())
        return ModelRuntimeObjects(
            model, lambda input_df: self.__predict(model, input_df)
        )

    @staticmethod
    def __predict(model: ModelObject, input_df: pd.DataFrame) -> pd.DataFrame:
        from mlflow.xgboost import _XGBModelWrapper

        model_prediction_obj = _XGBModelWrapper(model)
        predictions = model_prediction_obj.predict(input_df)
        if isinstance(predictions, np.ndarray):
            return pd.DataFrame(predictions)
        elif isinstance(predictions, pd.DataFrame):
            return predictions
        elif isinstance(predictions, spmatrix):
            return pd.DataFrame.sparse.from_spmatrix(predictions)  # type: ignore
        else:
            raise Exception(f"Unsupported return type: {type(predictions)}")
