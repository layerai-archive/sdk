from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.contracts.models import TrainedModelObject

from .base import ModelFlavor


class CatBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of CatBoost Models."""

    MODULE_KEYWORD = "catboost"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_CATBOOST")

    def save_model_to_directory(
        self, model_object: TrainedModelObject, directory: Path
    ) -> None:
        import mlflow.catboost

        mlflow.catboost.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> TrainedModelObject:
        import mlflow.catboost

        return mlflow.catboost.load_model(directory.as_uri())
