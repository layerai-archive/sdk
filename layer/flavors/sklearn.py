from pathlib import Path

from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.contracts.models import TrainedModelObject

from .base import ModelFlavor


class ScikitLearnModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Scikit Learn Models."""

    MODULE_KEYWORD = "sklearn"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_SKLEARN")

    def save_model_to_directory(
        self, model_object: TrainedModelObject, directory: Path
    ) -> None:
        import mlflow.sklearn

        # Added serialization_format specifically to disable pickle5 which causes errors on Python 3.7
        return mlflow.sklearn.save_model(
            model_object, path=directory.as_posix(), serialization_format="pickle"
        )

    def load_model_from_directory(self, directory: Path) -> TrainedModelObject:
        import mlflow.sklearn

        return mlflow.sklearn.load_model(directory.as_uri())
