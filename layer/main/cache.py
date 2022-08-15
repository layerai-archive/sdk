from layer.cache.cache import Cache

from .utils import sdk_function


@sdk_function
def clear_cache() -> None:
    """
    Clear all cached objects fetched by the Layer SDK on this machine.
    The Layer SDK locally stores all datasets and models by default on your computer.
    When you fetch a dataset with :func:``layer.get_dataset``, or load the model with ``layer.get_model``,
    the first call will fetch the artifact from remote storage,
    but subsequent calls will read it from the local disk.
    """
    Cache().clear()
