import logging
import os
import pickle  # nosec import_pickle
from uuid import UUID

import layer
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.contracts.assets import AssetType
from layer.contracts.runs import FunctionDefinition
from layer.executables.model.model_trainer import LocalTrainContext, ModelTrainer
from layer.global_context import set_has_shown_update_message
from layer.projects.utils import (
    get_current_project_full_name,
    verify_project_exists_and_retrieve_project_id,
)
from layer.tracker.progress_tracker import RunProgressTracker
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)
set_has_shown_update_message(True)


# load the entrypoint function
with open("function.pkl", "rb") as file:
    layer.init(os.environ["LAYER_PROJECT_NAME"])
    user_function = pickle.load(file)  # nosec pickle
    current_project_full_name_ = get_current_project_full_name()
    config: Config = asyncio_run_in_thread(ConfigManager().refresh())
    with LayerClient(config.client, logger).init() as client:
        progress_tracker = RunProgressTracker(
            url=config.url,
            project_name=current_project_full_name_.project_name,
            account_name=current_project_full_name_.account_name,
        )

        with progress_tracker.track() as tracker:
            tracker.add_asset(AssetType.MODEL, user_function.layer.get_asset_name())
            model_definition = FunctionDefinition(
                func=user_function,
                asset_name=user_function.layer.get_asset_name(),
                asset_type=AssetType.MODEL,
                fabric=user_function.layer.get_fabric(),
                asset_dependencies=[],
                pip_dependencies=[],
                resource_paths=user_function.layer.get_resource_paths(),
                assertions=user_function.layer.get_assertions(),
                project_name=current_project_full_name_.project_name,
                account_name=current_project_full_name_.account_name,
            )
            assert model_definition.project_name is not None
            verify_project_exists_and_retrieve_project_id(
                client, model_definition.project_full_name
            )

            model_version = client.model_catalog.create_model_version(
                model_definition.project_full_name,
                model_definition,
                True,
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
