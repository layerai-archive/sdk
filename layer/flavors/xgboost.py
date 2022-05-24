from pathlib import Path
from typing import Any, Callable, Tuple

import pandas as pd
from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.contracts.models import TrainedModelObject

from .base import ModelFlavor


class XGBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of XGBoost Models.

    Uses XGBoost model (an instance of xgboost.Booster).

    """

    MODULE_KEYWORD = "xgboost"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_XGBOOST")

    def save_model_to_directory(
        self, model_object: TrainedModelObject, directory: Path
    ) -> None:
        import mlflow.xgboost

        mlflow.xgboost.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(
        self, directory: Path
    ) -> Tuple[TrainedModelObject, Callable[[pd.DataFrame], pd.DataFrame]]:
        import mlflow.xgboost

        model = mlflow.xgboost.load_model(directory.as_uri())
        return model, lambda input_df: self.__predict(model, input_df)

    @staticmethod
    def __predict(model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
        raise Exception("Not implemented")
