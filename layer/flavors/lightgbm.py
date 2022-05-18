from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.contracts.models import TrainedModelObject

from .base import ModelFlavor


class LightGBMModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of LightGBM Models.
    Uses LightGBM model (an instance of lightgbm.Booster).

    """

    MODULE_KEYWORD = "lightgbm"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_LIGHTGBM")

    def save_model_to_directory(
        self,
        model_object: TrainedModelObject,
        directory: Path,
    ) -> None:
        import mlflow.lightgbm

        return mlflow.lightgbm.save_model(model_object, path=directory.as_posix())

    def load_model_from_directory(self, directory: Path) -> TrainedModelObject:
        import mlflow.lightgbm

        return mlflow.lightgbm.load_model(directory.as_uri())
