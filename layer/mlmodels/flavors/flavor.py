import inspect
import pathlib
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import mlflow.keras
from yarl import URL

from layer.projects.tracker.project_progress_tracker import ProjectProgressTracker
from layer.projects.tracker.resource_transfer_state import ResourceTransferState

from ...cache import Cache
from ...s3 import S3Util
from .. import ModelObject
from .model_definition import ModelDefinition


class ModelFlavorMetaData(NamedTuple):
    """NamedTuple containing flavor module and class names"""

    module_name: str
    class_name: str


class ModelFlavor(metaclass=ABCMeta):
    """Represents a machine learning model flavor for a specific framework.

    Implementations provide methods for checking for membership, saving and loading
    of machine learning models within their flavor.
    """

    def __init__(
        self, no_cache: bool = False, cache_dir: Optional[Path] = None
    ) -> None:
        self._no_cache = no_cache
        self._cache_dir = cache_dir
        self._from_cache = False

    @property
    def from_cache(self) -> bool:
        """If true, the model was loaded from the local cache, rather than the remote location.
        The initial value is false. The value is updated after the call
        to `layer.mlmodels.flavors.flavor.ModelFlavor.load`.
        """
        return self._from_cache

    @abstractmethod
    def module_keyword(self) -> str:
        """Defines a keyword as part of an object's module name that matches this flavor.

        Returns:
            keyword: a str
        """

    @abstractmethod
    def log_model_impl(self) -> Any:
        """Defines the method that this Model Flavor uses to log(/store) a model.

        Returns:
             A method reference of the model log implementation.
        """

    @abstractmethod
    def load_model_impl(self) -> Any:
        """Defines the method that this Model Flavor uses to load a model.

        Returns:
             A method reference of the model loader implementation.
        """

    def log_model_args(self) -> Dict[str, Any]:
        """Defines the kwargs that this Model Flavor uses to log a model.

        Returns:
             Dictionary of kwargs.
        """
        return {}

    def load_model_args(self) -> Dict[str, Any]:
        """Defines the kwargs that this Model Flavor uses to load a model.

        Returns:
             Dictionary of kwargs.
        """
        return {}

    def can_interpret_object(self, model_object: ModelObject) -> bool:
        """Checks whether supplied model object has flavor of this class.

        Args:
            model_object: A machine learning model which could be originated from any framework.

        Returns:
            bool: true if this ModelFlavor can interpret the given model instance.
        """
        for hierarchy_class in inspect.getmro(type(model_object)):
            parent_module = inspect.getmodule(hierarchy_class)
            if (
                parent_module is not None
                and self.module_keyword() in parent_module.__name__
            ):
                return True

        return False

    def save(
        self,
        model_definition: ModelDefinition,
        model_object: ModelObject,
        s3_endpoint_url: Optional[URL] = None,
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> None:
        """Stores the given machine learning model definition to the backing store.

        Args:
            model_definition: Model metadata object which describes the model instance
            model_object: A machine learning model which could be originated from any framework
        """
        from mlflow.utils.file_utils import TempDir

        if not tracker:
            tracker = ProjectProgressTracker()

        model_impl = self.log_model_impl()
        with TempDir() as tmp:
            local_path = pathlib.Path((tmp.path("model")))
            local_path_str = local_path.as_posix()

            args = self.log_model_args()
            model_impl(model_object, path=local_path_str, **args)
            state = ResourceTransferState()
            tracker.mark_model_saving_result(model_definition.model_raw_name, state)
            S3Util.upload_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
                state=state,
            )

    def load(
        self, model_definition: ModelDefinition, s3_endpoint_url: Optional[URL] = None
    ) -> ModelObject:
        """Loads the given machine learning model definition from the backing store and
        returns an instance of it

        Args:
            model_definition: Model metadata object which describes the model instance

        Returns:
            A machine learning model object
        """
        cache = Cache(cache_dir=self._cache_dir).initialise()
        model_train_id = model_definition.model_train_id.value
        model_cache_dir = cache.get_path_entry(model_train_id)

        if model_cache_dir is None or self._no_cache:
            self._from_cache = False
            from mlflow.utils.file_utils import TempDir

            with TempDir() as tmp:
                local_path = pathlib.Path((tmp.path("model")))
                S3Util.download_dir(
                    local_dir=local_path,
                    credentials=model_definition.credentials,
                    s3_path=model_definition.s3_path,
                    endpoint_url=s3_endpoint_url,
                )
                if self._no_cache:
                    return self._load_model(local_path)
                model_cache_dir = cache.put_path_entry(model_train_id, local_path)
        else:
            self._from_cache = True

        return self._load_model(model_cache_dir)  # type: ignore

    def _load_model(self, model_dir: Path) -> ModelObject:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            args = self.load_model_args()
            model_impl = self.load_model_impl()
            return model_impl(f"file://{model_dir.absolute().as_posix()}", **args)

    @property
    def metadata(self) -> ModelFlavorMetaData:
        """Returns metadata which contains flavor module and classname to be used when loading an instance of a flavor

        Returns:
            ModelFlavorMetaData: a metadata dict
        """
        return ModelFlavorMetaData(
            module_name=self.__module__, class_name=self.__class__.__qualname__
        )


class CatBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of CaBoost Models."""

    def module_keyword(self) -> str:
        return "catboost"

    def log_model_impl(self) -> Any:
        import mlflow.catboost

        return mlflow.catboost.save_model

    def load_model_impl(self) -> Any:
        import mlflow.catboost

        return mlflow.catboost.load_model


class ScikitLearnModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Scikit Learn Models."""

    def module_keyword(self) -> str:
        return "sklearn"

    def log_model_args(self) -> Dict[str, Any]:
        # Added serialization_format specifically to disable pickle5 which causes errors on Python 3.7
        return {"serialization_format": "pickle"}

    def log_model_impl(self) -> Any:
        import mlflow.sklearn

        return mlflow.sklearn.save_model

    def load_model_impl(self) -> Any:
        import mlflow.sklearn

        return mlflow.sklearn.load_model


class PyTorchModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of PyTorch Models."""

    def module_keyword(self) -> str:
        return "torch"

    def log_model_impl(self) -> Any:
        import mlflow.pytorch

        return mlflow.pytorch.save_model

    def load_model_impl(self) -> Any:
        import mlflow.pytorch

        return mlflow.pytorch.load_model

    def load_model_args(self) -> Dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {"map_location": torch.device("cpu")}
        return {}


class XGBoostModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of XGBoost Models.

    Uses XGBoost model (an instance of xgboost.Booster).

    """

    def module_keyword(self) -> str:
        return "xgboost"

    def log_model_impl(self) -> Any:
        import mlflow.xgboost

        return mlflow.xgboost.save_model

    def load_model_impl(self) -> Any:
        import mlflow.xgboost

        return mlflow.xgboost.load_model


class LightGBMModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of LightGBM Models.
    Uses LightGBM model (an instance of lightgbm.Booster).

    """

    def module_keyword(self) -> str:
        return "lightgbm"

    def log_model_impl(self) -> Any:
        import mlflow.lightgbm

        return mlflow.lightgbm.save_model

    def load_model_impl(self) -> Any:
        import mlflow.lightgbm

        return mlflow.lightgbm.load_model


class TensorFlowModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of TensorFlow Models."""

    def save(
        self,
        model_definition: ModelDefinition,
        model_object: ModelObject,
        s3_endpoint_url: Optional[URL] = None,
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> None:
        """See ModelFlavor.save()'s documentation.

        Due to peculiar arguments required for mlflow.tensorflow.save_model, we need to duplicate a lot of code here.

        TODO: Refactor this
        """
        import tensorflow  # type: ignore
        from mlflow.utils.file_utils import TempDir

        if not tracker:
            tracker = ProjectProgressTracker()

        model_impl = self.log_model_impl()

        with TempDir() as tmp:
            local_path = pathlib.Path((tmp.path("model")))
            local_path_str = local_path.as_posix()
            tmp_save_path = pathlib.Path(tmp.path(model_definition.model_name))
            tensorflow.saved_model.save(model_object, tmp_save_path)

            model_impl(
                tf_saved_model_dir=tmp_save_path,
                tf_meta_graph_tags=None,
                tf_signature_def_key="serving_default",
                path=local_path_str,
            )
            state = ResourceTransferState()
            tracker.mark_model_saving_result(model_definition.model_raw_name, state)
            S3Util.upload_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
                state=state,
            )

    def module_keyword(self) -> str:
        return "tensorflow"

    def log_model_impl(self) -> Any:
        import mlflow.tensorflow

        return mlflow.tensorflow.save_model

    def load_model_impl(self) -> Any:
        import mlflow.tensorflow

        return mlflow.tensorflow.load_model


class KerasModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Keras Models."""

    TOKENIZER_FILE = "tokenizer.pickle"

    def module_keyword(self) -> str:
        return "keras"

    def log_model_impl(self) -> Any:
        raise NotImplementedError("Use Keras specific `save` method")

    def load_model_impl(self) -> Any:
        raise NotImplementedError("Use Keras specific `load` method")

    def can_interpret_object(self, model_object: ModelObject) -> bool:
        # Check if model is a tensorflow keras
        try:
            import tensorflow.keras.models  # type: ignore

            if isinstance(model_object, tensorflow.keras.models.Model):
                return True
        except ImportError:
            pass

        # Check if model is a pure keras
        try:
            import keras.engine.network  # type: ignore

            if isinstance(model_object, keras.engine.network.Network):
                return True
        except ImportError:
            pass

        # Check if model is a tensorflow/pure keras tokenizer
        try:
            import keras

            if isinstance(model_object, keras.preprocessing.text.Tokenizer):
                return True
        except ImportError:
            pass

        return False

    def save(
        self,
        model_definition: ModelDefinition,
        model_object: ModelObject,
        s3_endpoint_url: Optional[URL] = None,
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> None:
        import os

        import cloudpickle  # type: ignore
        import keras
        from mlflow.utils.file_utils import TempDir

        if not tracker:
            tracker = ProjectProgressTracker()

        with TempDir() as tmp:
            local_path = pathlib.Path((tmp.path("model")))
            local_path_str = local_path.as_posix()
            if isinstance(model_object, keras.preprocessing.text.Tokenizer):
                os.makedirs(local_path)
                with open(local_path / KerasModelFlavor.TOKENIZER_FILE, "wb") as handle:
                    cloudpickle.dump(model_object, handle)
            else:
                args = self.log_model_args()
                mlflow.keras.save_model(model_object, path=local_path_str, **args)

            state = ResourceTransferState()
            tracker.mark_model_saving_result(model_definition.model_raw_name, state)
            S3Util.upload_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
                state=state,
            )

    def load(
        self, model_definition: ModelDefinition, s3_endpoint_url: Optional[URL] = None
    ) -> ModelObject:
        import os
        import warnings

        from mlflow.utils.file_utils import TempDir

        with TempDir() as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            local_path = pathlib.Path((tmp.path("model")))
            S3Util.download_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
            )

            if os.path.exists(local_path / KerasModelFlavor.TOKENIZER_FILE):
                import cloudpickle

                # loading
                with open(local_path / KerasModelFlavor.TOKENIZER_FILE, "rb") as handle:
                    return cloudpickle.load(handle)
            else:
                args = self.load_model_args()
                local_path_str = local_path.as_posix()
                return mlflow.keras.load_model(f"file://{local_path_str}", **args)


class HuggingFaceModelFlavor(ModelFlavor):
    """An ML Model flavor implementation which handles persistence of Hugging Face Transformer Models."""

    HF_TYPE_FILE = "model.hf_type"

    def module_keyword(self) -> str:
        return "transformers.models"

    def log_model_impl(self) -> Any:
        raise NotImplementedError("Use HuggingFace specific `save` method")

    def load_model_impl(self) -> Any:
        raise NotImplementedError("Use HuggingFace specific `load` method")

    def save_transformer(self, model_object: Any, local_path: Path) -> None:
        local_path_str = local_path.as_posix()
        model_object.save_pretrained(local_path_str)

        with open(local_path / HuggingFaceModelFlavor.HF_TYPE_FILE, "w") as f:
            f.write(type(model_object).__name__)

    def load_transformer(self, model_path: Path) -> Any:
        with open(model_path / HuggingFaceModelFlavor.HF_TYPE_FILE) as f:
            transformer_type = f.readlines()[0]

            mod = __import__("transformers", fromlist=[transformer_type])
            architecture_class = getattr(mod, transformer_type)

            model_path_str = model_path.as_posix()
            return architecture_class.from_pretrained(model_path_str)

    def save(
        self,
        model_definition: ModelDefinition,
        model_object: Any,
        s3_endpoint_url: Optional[URL] = None,
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> None:
        from mlflow.utils.file_utils import TempDir

        if not tracker:
            tracker = ProjectProgressTracker()

        with TempDir() as tmp:
            local_path = pathlib.Path((tmp.path()))

            self.save_transformer(model_object, local_path)
            state = ResourceTransferState()
            tracker.mark_model_saving_result(model_definition.model_raw_name, state)
            S3Util.upload_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
                state=state,
            )

    def load(
        self, model_definition: ModelDefinition, s3_endpoint_url: Optional[URL] = None
    ) -> ModelObject:
        import warnings

        from mlflow.utils.file_utils import TempDir

        with TempDir() as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            local_path = pathlib.Path((tmp.path("model")))
            S3Util.download_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
            )

            return self.load_transformer(local_path)
