import logging
import os
from typing import Any
from uuid import UUID

from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.contracts.assets import AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.executables.model.model_trainer import LocalTrainContext, ModelTrainer
from layer.global_context import reset_to, set_has_shown_update_message
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
from layer.tracker.utils import get_progress_tracker
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)
set_has_shown_update_message(True)


def runner(model_definition: FunctionDefinition) -> Any:
    def inner() -> Any:
        return _run(model_definition)

    return inner


def _run(model_definition: FunctionDefinition) -> Any:
    config: Config = asyncio_run_in_thread(ConfigManager().refresh())

    reset_to(model_definition.project_full_name.path)

    with LayerClient(config.client, logger).init() as client:
        progress_tracker = get_progress_tracker(
            url=config.url,
            project_name=model_definition.project_name,
            account_name=model_definition.account_name,
        )

        with progress_tracker.track() as tracker:
            tracker.add_asset(AssetType.MODEL, model_definition.asset_name)
            assert model_definition.project_name is not None
            verify_project_exists_and_retrieve_project_id(
                client, model_definition.project_full_name
            )

            model_version = client.model_catalog.create_model_version(
                model_definition.asset_path,
                model_definition.description,
                model_definition.source_code_digest,
                _get_display_fabric(),
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
                function=model_definition.func,
                args=model_definition.args,
                kwargs=model_definition.kwargs,
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


def _get_display_fabric() -> str:
    return os.getenv("LAYER_FABRIC", Fabric.F_LOCAL.value)
