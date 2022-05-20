import inspect
import tempfile
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from pathlib import Path
from typing import NamedTuple, Optional

from layerapi.api.entity.model_version_pb2 import (  # pylint: disable=unused-import
    ModelVersion,
)
from yarl import URL

from layer.cache import Cache
from layer.contracts.models import TrainedModelObject
from layer.contracts.runs import ResourceTransferState
from layer.tracker.project_progress_tracker import ProjectProgressTracker
from layer.utils.s3 import S3Util

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

    @abstractproperty
    def MODULE_KEYWORD(self) -> str:  # pylint: disable=invalid-name
        """Defines a keyword as part of an object's module name that matches this flavor.

        Returns:
            keyword: a str
        """

    @abstractproperty
    def PROTO_FLAVOR(  # pylint: disable=invalid-name
        self,
    ) -> "ModelVersion.ModelFlavor":
        """Defines the proto flavor that this Model Flavor uses.

        Returns:
            The proto flavor
        """

    @property
    def from_cache(self) -> bool:
        """If true, the model was loaded from the local cache, rather than the remote location.
        The initial value is false. The value is updated after the call
        to `layer.flavors.base.ModelFlavor.load`.
        """
        return self._from_cache

    @property
    def metadata(self) -> ModelFlavorMetaData:
        """Returns metadata which contains flavor module and classname to be used when loading an instance of a flavor

        Returns:
            ModelFlavorMetaData: a metadata dict
        """
        return ModelFlavorMetaData(
            module_name=self.__module__, class_name=self.__class__.__qualname__
        )

    def can_interpret_object(self, model_object: TrainedModelObject) -> bool:
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
                and self.MODULE_KEYWORD in parent_module.__name__
            ):
                return True

        return False

    @abstractmethod
    def save_model_to_directory(
        self,
        model_object: TrainedModelObject,
        directory: Path,
    ) -> None:
        """Defines the method that this Model Flavor uses to save a model to a directory.

        Returns:
             A callable to save the model.
        """

    @abstractmethod
    def load_model_from_directory(self, directory: Path) -> TrainedModelObject:
        """Defines the method that this Model Flavor uses to load a model from a directory.

        Returns:
             A callable to load the model.
        """

    def save_to_s3(
        self,
        model_definition: ModelDefinition,
        model_object: TrainedModelObject,
        s3_endpoint_url: Optional[URL] = None,
        tracker: Optional[ProjectProgressTracker] = None,
    ) -> None:
        """Stores the given machine learning model definition to the backing store.

        Args:
            model_definition: Model metadata object which describes the model instance
            model_object: A machine learning model which could be originated from any framework
        """

        if not tracker:
            tracker = ProjectProgressTracker()

        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / "model"

            self.save_model_to_directory(model_object, local_path)
            state = ResourceTransferState()
            tracker.mark_model_saving_result(model_definition.model_raw_name, state)
            S3Util.upload_dir(
                local_dir=local_path,
                credentials=model_definition.credentials,
                s3_path=model_definition.s3_path,
                endpoint_url=s3_endpoint_url,
                state=state,
            )

    def load_from_s3(
        self,
        model_definition: ModelDefinition,
        state: Optional[ResourceTransferState],
        s3_endpoint_url: Optional[URL] = None,
    ) -> TrainedModelObject:
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
        if self._no_cache or not self.is_cached(model_definition):
            self._from_cache = False

            assert state

            with tempfile.TemporaryDirectory() as tmp:
                local_path = Path(tmp) / "model"

                S3Util.download_dir(
                    local_dir=local_path,
                    credentials=model_definition.credentials,
                    s3_path=model_definition.s3_path,
                    endpoint_url=s3_endpoint_url,
                    state=state,
                )
                if self._no_cache:
                    return self._load_model(local_path)
                model_cache_dir = cache.put_path_entry(model_train_id, local_path)
        else:
            self._from_cache = True

        assert model_cache_dir
        return self._load_model(model_cache_dir)

    def _load_model(self, model_dir: Path) -> TrainedModelObject:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.load_model_from_directory(model_dir)

    def is_cached(self, model_definition: ModelDefinition) -> bool:
        cache = Cache(cache_dir=self._cache_dir).initialise()
        model_train_id = model_definition.model_train_id.value
        model_cache_dir = cache.get_path_entry(model_train_id)
        return model_cache_dir is not None
