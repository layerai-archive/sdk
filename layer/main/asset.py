import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from layer.cache.utils import is_cached
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.context import (
    Context,
    get_active_context,
    reset_active_context,
    set_active_context,
)
from layer.contracts.assets import AssetPath, AssetType
from layer.contracts.datasets import Dataset
from layer.contracts.models import Model
from layer.contracts.project_full_name import ProjectFullName
from layer.contracts.tracker import ResourceTransferState
from layer.exceptions.exceptions import ProjectInitializationException
from layer.global_context import current_account_name
from layer.projects.utils import get_current_project_full_name
from layer.tracker.utils import get_progress_tracker
from layer.utils.async_utils import asyncio_run_in_thread

from .utils import sdk_function


if TYPE_CHECKING:
    import pandas

logger = logging.getLogger(__name__)


@sdk_function
def get_dataset(name: str, no_cache: bool = False) -> Dataset:
    """
    :param name: Name or path of the dataset.
    :param no_cache: if True, force dataset fetch from the remote location.
    :return: A dataset object defined in a Layer project.

    Retrieves a Layer dataset object from the Discover > Datasets tab.

    Guest users can use this function to access public datasets without logging in to Layer.

    By default, this function caches dataset contents locally when possible.

    .. code-block:: python

        # The simplest application of this method just returns a list of dataset information.
        layer.get_dataset("titanic")

        # This example gets the titanic dataset, turns it into a Pandas dataframe, and then displays it.
        dataset = layer.get_dataset("titanic")
        df = dataset.to_pandas()
        df.head()

        # You can also get datasets from other projects you have access to, including public projects.
        layer.get_dataset("the-project/datasets/titanic")
    """
    config: Config = asyncio_run_in_thread(ConfigManager().refresh(allow_guest=True))
    asset_path = AssetPath.parse(name, expected_asset_type=AssetType.DATASET)
    asset_path = _ensure_asset_path_is_absolute(asset_path)

    def fetch_dataset() -> "pandas.DataFrame":
        context = get_active_context()
        with LayerClient(config.client, logger).init() as client:
            within_run = (
                True if context else False
            )  # if layer.get_dataset is called within an @dataset decorated func or not
            callback = lambda: client.data_catalog.fetch_dataset(  # noqa: E731
                asset_path, no_cache=no_cache
            )
            if not within_run:
                try:
                    tracker = get_progress_tracker(
                        url=config.url,
                        account_name=asset_path.must_org_name(),
                        project_name=asset_path.must_project_name(),
                    )
                    with Context(
                        asset_type=AssetType.DATASET,
                        asset_name=asset_path.asset_name,
                        tracker=tracker,
                    ) as context:
                        set_active_context(context)
                        with tracker.track():
                            dataset = _ui_progress_with_tracker(
                                callback,
                                asset_path.asset_name,
                                False,  # Datasets are fetched per partition, no good way to show caching per partition
                                within_run,
                                context,
                                AssetType.DATASET,
                            )
                finally:
                    reset_active_context()  # Reset only if outside layer func, as the layer func logic will reset it
            else:
                assert context
                dataset = _ui_progress_with_tracker(
                    callback,
                    asset_path.asset_name,
                    False,  # Datasets are fetched per partition, no good way to show caching per partition
                    within_run,
                    context,
                    AssetType.DATASET,
                )

            return dataset

    return Dataset(
        asset_path=asset_path,
        _pandas_df_factory=fetch_dataset,
    )


@sdk_function
def get_model(name: str, no_cache: bool = False) -> Model:
    """
    :param name: Name or path of the model. You can pass additional parameters in the name to retrieve a specific version of the model with format: ``model_name:major_version.minor_version``
    :param no_cache: if True, force model fetch from the remote location.
    :return: The model object.

    Retrieves a Layer model object by its name.

    Guest users can use this function to access public models without logging in to Layer.

    By default, this function caches models locally when possible.

    .. code-block:: python

        # Loads the default version of the model.
        layer.get_model("churn_model")

        # Loads the latest train of version 2.
        layer.get_model("churn_model:2")

        # Loads a specific train of the model version 2.
        layer.get_model("churn_model:2.12")

        # Load a model from a project you aren't logged in to.
        layer.get_model("the-project/models/churn_model")
    """
    config = asyncio_run_in_thread(ConfigManager().refresh(allow_guest=True))
    asset_path = AssetPath.parse(name, expected_asset_type=AssetType.MODEL)
    asset_path = _ensure_asset_path_is_absolute(asset_path)
    context = get_active_context()

    with LayerClient(config.client, logger).init() as client:
        within_run = (
            True if context else False
        )  # if layer.get_model is called within a @model decorated func of not
        model = client.model_catalog.load_model_by_path(path=asset_path.path())
        from_cache = not no_cache and is_cached(model)
        state = ResourceTransferState(model.name)

        def callback() -> Model:
            return _load_model_runtime_objects(client, model, state, no_cache)

        if not within_run:
            try:
                tracker = get_progress_tracker(
                    url=config.url,
                    account_name=asset_path.must_org_name(),
                    project_name=asset_path.must_project_name(),
                )
                with Context(
                    asset_type=AssetType.MODEL,
                    asset_name=asset_path.asset_name,
                    tracker=tracker,
                ) as context:
                    set_active_context(context)
                    with tracker.track():
                        model = _ui_progress_with_tracker(
                            callback,
                            asset_path.asset_name,
                            from_cache,
                            within_run,
                            context,
                            AssetType.MODEL,
                            state,
                        )
            finally:
                reset_active_context()  # Reset only if outside layer func, as the layer func logic will reset it
        else:
            assert context
            model = _ui_progress_with_tracker(
                callback,
                asset_path.asset_name,
                from_cache,
                within_run,
                context,
                AssetType.MODEL,
                state,
            )

        return model


def _load_model_runtime_objects(
    client: LayerClient,
    model: Model,
    state: ResourceTransferState,
    no_cache: bool,
) -> Model:
    model_runtime_objects = client.model_catalog.load_model_runtime_objects(
        model,
        state=state,
        no_cache=no_cache,
    )
    model.set_model_runtime_objects(model_runtime_objects)
    return model


def _ui_progress_with_tracker(
    callback: Callable[[], Any],
    getting_asset_name: str,
    from_cache: bool,
    within_run: bool,
    context: Context,
    getting_asset_type: AssetType,
    state: Optional[ResourceTransferState] = None,
) -> Any:
    asset_name = context.asset_name()
    assert asset_name
    tracker = context.tracker()
    assert tracker
    asset_type = context.asset_type()
    assert asset_type
    if asset_type == AssetType.MODEL:
        if getting_asset_type == AssetType.DATASET:
            tracker.mark_model_getting_dataset(
                asset_name, getting_asset_name, from_cache
            )
        elif getting_asset_type == AssetType.MODEL:
            tracker.mark_model_getting_model(
                asset_name, getting_asset_name, state, from_cache
            )
    elif asset_type == AssetType.DATASET:
        if getting_asset_type == AssetType.DATASET:
            tracker.mark_dataset_getting_dataset(
                asset_name, getting_asset_name, from_cache
            )
        elif getting_asset_type == AssetType.MODEL:
            tracker.mark_dataset_getting_model(
                asset_name, getting_asset_name, state, from_cache
            )
    result = callback()
    if within_run:
        if asset_type == AssetType.MODEL:
            tracker.mark_model_training(asset_name)
        elif asset_type == AssetType.DATASET:
            tracker.mark_dataset_building(asset_name)
    elif asset_type == AssetType.MODEL:
        tracker.mark_model_loaded(asset_name)
    elif asset_type == AssetType.DATASET:
        tracker.mark_dataset_loaded(asset_name)
    return result


def _ensure_asset_path_is_absolute(
    path: AssetPath,
) -> AssetPath:
    if not path.is_relative():
        return path
    project_name = (
        path.project_name
        if path.has_project()
        else get_current_project_full_name().project_name
    )
    account_name = (
        path.org_name if path.org_name is not None else current_account_name()
    )

    if not project_name or not account_name:
        raise ProjectInitializationException(
            "Please specify the project full name globally with layer.init('account-name/project-name')"
            "or have it in the asset full name like 'the-account/the-project/models/the-model-name'"
        )

    path = path.with_project_full_name(
        ProjectFullName(account_name=account_name, project_name=project_name)
    )
    return path


@sdk_function
def save_model(model: Any) -> None:
    """
    :param model: The model object to save.
    :return: None.

    Saves a model object to Layer under the currently decorated function.

    .. code-block:: python

        # Saves the model to the current train.
        my_model = train()
        layer.save_model(my_model)
    """
    active_context = get_active_context()
    if not active_context:
        raise RuntimeError(
            "Saving model only allowed inside functions decorated with @model"
        )
    train = active_context.train()
    tracker = active_context.tracker()
    if not train or not tracker:
        raise RuntimeError(
            "Saving model only allowed inside functions decorated with @model"
        )
    train.save_model(model, tracker=tracker)
