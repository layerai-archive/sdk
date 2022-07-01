import logging
import os
import pickle  # nosec import_pickle
from typing import Any
from uuid import UUID

import layer
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.contracts.assets import AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.executables.model.model_trainer import LocalTrainContext, ModelTrainer
from layer.global_context import set_has_shown_update_message
from layer.projects.utils import (
    get_current_project_full_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.settings import LayerSettings
from layer.tracker.utils import get_progress_tracker
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)
set_has_shown_update_message(True)


def _run(user_function: Any) -> Any:
    settings: LayerSettings = user_function.layer
    current_project_full_name_ = get_current_project_full_name()
    config: Config = asyncio_run_in_thread(ConfigManager().refresh())
    with LayerClient(config.client, logger).init() as client:
        progress_tracker = get_progress_tracker(
            url=config.url,
            project_name=current_project_full_name_.project_name,
            account_name=current_project_full_name_.account_name,
        )

        with progress_tracker.track() as tracker:
            tracker.add_asset(AssetType.MODEL, settings.get_asset_name())
            model_definition = FunctionDefinition(
                func=user_function,
                asset_name=settings.get_asset_name(),
                asset_type=AssetType.MODEL,
                fabric=settings.get_fabric(),
                asset_dependencies=[],
                pip_dependencies=[],
                resource_paths=settings.get_resource_paths(),
                assertions=settings.get_assertions(),
                project_name=current_project_full_name_.project_name,
                account_name=current_project_full_name_.account_name,
            )
            assert model_definition.project_name is not None
            verify_project_exists_and_retrieve_project_id(
                client, model_definition.project_full_name
            )

            model_version = client.model_catalog.create_model_version(
                model_definition.asset_path,
                model_definition.description,
                model_definition.source_code_digest.hexdigest(),
                model_definition.get_fabric(True),
            ).model_version
            train_id = client.model_catalog.create_model_train_from_version_id(
                model_version.id
            )
            train = client.model_catalog.get_model_train(train_id)

            context = LocalTrainContext(  # noqa: F841
                logger=logger,
                model_name=model_definition.asset_name,
                model_version=model_version.name,
                train_id=UUID(train_id.value),
                function=user_function,
                train_index=str(train.index),
            )
            trainer = ModelTrainer(
                client=client,
                train_context=context,
                logger=logger,
                tracker=tracker,
            )
            result = trainer.train()

            tracker.mark_model_trained(
                name=model_definition.asset_name,
                train_index=str(train.index),
                version=model_version.name,
            )

            return result


LAYER_CLIENT_AUTH_URL = os.environ["LAYER_CLIENT_AUTH_URL"]
LAYER_CLIENT_AUTH_TOKEN = os.environ["LAYER_CLIENT_AUTH_TOKEN"]
LAYER_PROJECT_NAME = os.environ["LAYER_PROJECT_NAME"]

layer.login_with_access_token(
    access_token=LAYER_CLIENT_AUTH_TOKEN, url=LAYER_CLIENT_AUTH_URL
)
layer.init(LAYER_PROJECT_NAME)

# load the entrypoint function
with open("function.pkl", "rb") as file:
    user_function = pickle.load(file)  # nosec pickle

_run(user_function)
