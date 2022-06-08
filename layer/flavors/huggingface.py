from pathlib import Path

import pandas as pd
from layerapi.api.value.model_flavor_pb2 import ModelFlavor as PbModelFlavor

from layer.types import ModelObject

from .base import ModelFlavor, ModelRuntimeObjects


class HuggingFaceModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Hugging Face Transformer Models."""

    MODULE_KEYWORD = "transformers.models"
    PROTO_FLAVOR = PbModelFlavor.MODEL_FLAVOR_HUGGINGFACE

    HF_TYPE_FILE = "model.hf_type"

    def save_model_to_directory(
        self,
        model_object: ModelObject,
        directory: Path,
    ) -> None:
        model_object.save_pretrained(directory.as_posix())  # type: ignore

        with open(directory / self.HF_TYPE_FILE, "w") as f:
            f.write(type(model_object).__name__)

    def load_model_from_directory(self, directory: Path) -> ModelRuntimeObjects:
        with open(directory / self.HF_TYPE_FILE) as f:
            transformer_type = f.readlines()[0]

            mod = __import__("transformers", fromlist=[transformer_type])
            architecture_class = getattr(mod, transformer_type)

            model = architecture_class.from_pretrained(directory.as_posix())

            return ModelRuntimeObjects(
                model, lambda input_df: self.__predict(model, input_df)
            )

    @staticmethod
    def __predict(model: ModelObject, input_df: pd.DataFrame) -> pd.DataFrame:
        raise Exception("Not implemented")
