from pathlib import Path
from typing import Any, Callable, Tuple

import pandas as pd
from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.types import ModelArtifact

from .base import ModelFlavor


class CatBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of CatBoost Models."""

    MODULE_KEYWORD = "catboost"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.MODEL_FLAVOR_CATBOOST

    def save_model_to_directory(
        self, model_object: ModelArtifact, directory: Path
    ) -> None:
        import mlflow.catboost

        mlflow.catboost.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(
        self, directory: Path
    ) -> Tuple[ModelArtifact, Callable[[pd.DataFrame], pd.DataFrame]]:
        import mlflow.catboost

        model = mlflow.catboost.load_model(directory.as_uri())
        return model, lambda input_df: self.__predict(model, input_df)

    @staticmethod
    def __predict(model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
        prediction_np_array = model.predict(input_df)
        return pd.DataFrame(prediction_np_array, columns=["prediction"])
