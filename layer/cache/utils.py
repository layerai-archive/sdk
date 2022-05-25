from typing import Union

from layer.contracts.datasets import Dataset
from layer.contracts.models import Model

from .cache import Cache


def is_cached(asset: Union[Dataset, Model]) -> bool:
    cache = Cache(cache_dir=None).initialise()
    cache_dir = cache.get_path_entry(str(asset.id))
    return cache_dir is not None


def clear_cache() -> None:
    """
    Clear all cached objects fetched by the Layer SDK on this machine.

    The Layer SDK locally stores all datasets and models by default on your computer.
    When you fetch a dataset with :func:``layer.get_dataset``, or load the model with ``layer.get_model``,
    the first call will fetch the artifact from remote storage,
    but subsequent calls will read it from the local disk.
    """
    Cache().clear()
