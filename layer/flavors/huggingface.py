from pathlib import Path
from typing import Any, Callable, Tuple

import pandas as pd
from layerapi.api.entity.model_version_pb2 import ModelVersion

from layer.contracts.models import TrainedModelObject

from .base import ModelFlavor


class HuggingFaceModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Hugging Face Transformer Models."""

    MODULE_KEYWORD = "transformers.models"
    PROTO_FLAVOR = ModelVersion.ModelFlavor.Value("MODEL_FLAVOR_HUGGINGFACE")

    HF_TYPE_FILE = "model.hf_type"

    def save_model_to_directory(
        self,
        model_object: Any,
        directory: Path,
    ) -> None:
        model_object.save_pretrained(directory.as_posix())

        with open(directory / self.HF_TYPE_FILE, "w") as f:
            f.write(type(model_object).__name__)

    def load_model_from_directory(
        self, directory: Path
    ) -> Tuple[TrainedModelObject, Callable[[pd.DataFrame], pd.DataFrame]]:
        with open(directory / self.HF_TYPE_FILE) as f:
            transformer_type = f.readlines()[0]

            mod = __import__("transformers", fromlist=[transformer_type])
            architecture_class = getattr(mod, transformer_type)
            model = architecture_class.from_pretrained(directory.as_posix())

            return model, lambda input_df: self.__predict(model, input_df)

    @staticmethod
    def __predict(model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
        raise Exception("Not implemented")
