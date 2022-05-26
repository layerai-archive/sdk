from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.types import ModelArtifact

from .base import ModelFlavor


class XGBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of XGBoost Models.

    Uses XGBoost model (an instance of xgboost.Booster).

    """

    MODULE_KEYWORD = "xgboost"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.MODEL_FLAVOR_XGBOOST

    def save_model_to_directory(
        self, model_object: ModelArtifact, directory: Path
    ) -> None:
        import mlflow.xgboost

        mlflow.xgboost.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> ModelArtifact:
        import mlflow.xgboost

        return mlflow.xgboost.load_model(directory.as_uri())
