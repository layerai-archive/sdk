from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix  # type: ignore
from layerapi.api.entity.model_train_pb2 import ModelTrain

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


class LightGBMModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of LightGBM Models.
    Uses LightGBM model (an instance of lightgbm.Booster).

    """

    MODULE_KEYWORD = "lightgbm"
    PROTO_FLAVOR = ModelTrain.ModelFlavor.MODEL_FLAVOR_LIGHTGBM

    def save_model_to_directory(
        self,
        model_object: ModelObject,
        directory: Path,
    ) -> None:
        import mlflow.lightgbm

        return mlflow.lightgbm.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        import mlflow.lightgbm

        model = mlflow.lightgbm.load_model(directory.as_uri())
        return ModelRuntimeObjects(
            model, lambda input_df: self.__predict(model, input_df)
        )

    @staticmethod
    def __predict(model: ModelObject, input_df: pd.DataFrame) -> pd.DataFrame:
        prediction = model.predict(input_df)  # type: ignore
        if isinstance(prediction, np.ndarray):
            return pd.DataFrame(prediction)
        elif isinstance(prediction, pd.DataFrame):
            return prediction
        elif isinstance(prediction, spmatrix):
            return pd.DataFrame.sparse.from_spmatrix(prediction)  # type: ignore
        else:
            raise Exception(f"Unsupported return type: {type(prediction)}")
