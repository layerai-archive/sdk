import logging
import os
from typing import Any, Callable

import layer
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.global_context import reset_to, set_has_shown_update_message
from layer.logged_data.queuing_logged_data_destination import (
    QueueingLoggedDataDestination,
)
from layer.tracker.utils import get_progress_tracker
from layer.utils.async_utils import asyncio_run_in_thread


ENV_LAYER_API_URL = "LAYER_API_URL"
ENV_LAYER_API_KEY = "LAYER_API_KEY"
ENV_LAYER_API_TOKEN = "LAYER_API_TOKEN"
ENV_LAYER_RUN_ID = "LAYER_RUN_ID"
ENV_LAYER_FABRIC = "LAYER_FABRIC"

RunnerFunction = Callable[[FunctionDefinition], Any]
RunFunction = Any  # TODO(volkan) build a better argument typing to pass in

logger = logging.getLogger(__name__)


def make_runner(run_function: RunFunction) -> RunnerFunction:
    def runner(definition: FunctionDefinition) -> Any:
        def inner() -> Any:
            _initialize(definition)
            config: Config = asyncio_run_in_thread(ConfigManager().refresh())
            progress_tracker = get_progress_tracker(
                url=config.url,
                project_name=definition.project_name,
                account_name=definition.account_name,
            )
            with LayerClient(
                config.client, logger
            ).init() as client, progress_tracker.track() as tracker:
                with QueueingLoggedDataDestination(
                    client=client.logged_data_service_client
                ) as logged_data_destination:
                    tracker.add_asset(definition.asset_type, definition.asset_name)
                    return run_function(
                        config.url,
                        definition,
                        client,
                        tracker,
                        _get_display_fabric(),
                        _get_run_id(),
                        logged_data_destination=logged_data_destination,
                    )

        return inner

    return runner


def _initialize(definition: FunctionDefinition) -> None:
    set_has_shown_update_message(True)
    reset_to(definition.project_full_name.path)

    api_url = os.environ.get(ENV_LAYER_API_URL)
    api_key = os.environ.get(ENV_LAYER_API_KEY)
    api_token = os.environ.get(ENV_LAYER_API_TOKEN)

    if api_url:
        if api_key:
            layer.login_with_api_key(api_key, url=api_url)
        elif api_token:
            layer.login_with_access_token(api_token, url=api_url)


def _get_display_fabric() -> Fabric:
    return Fabric.find(os.getenv(ENV_LAYER_FABRIC, "f-local"))


def _get_run_id() -> str:
    return os.getenv(ENV_LAYER_RUN_ID, "")
