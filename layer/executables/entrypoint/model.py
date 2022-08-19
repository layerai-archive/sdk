import logging
from typing import Any
from uuid import UUID

from layerapi.api.entity.task_pb2 import Task
from layerapi.api.ids_pb2 import RunId

from layer.clients.layer import LayerClient
from layer.config.config import Config
from layer.contracts.assets import AssetType
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
from layer.tracker.utils import get_progress_tracker

from .common import make_runner
from .model_trainer import LocalTrainContext, ModelTrainer


logger = logging.getLogger(__name__)


def _run(
    model_definition: FunctionDefinition, config: Config, fabric: Fabric, run_id: str
) -> Any:

    with LayerClient(config.client, logger).init() as client:
        progress_tracker = get_progress_tracker(
            url=config.url,
            project_name=model_definition.project_name,
            account_name=model_definition.account_name,
        )

        with progress_tracker.track() as tracker:
            tracker.add_asset(AssetType.MODEL, model_definition.asset_name)

            verify_project_exists_and_retrieve_project_id(
                client, model_definition.project_full_name
            )

            model_version = client.model_catalog.create_model_version(
                model_definition.asset_path,
                model_definition.description,
                model_definition.source_code_digest,
                fabric.value,
            ).model_version
            train_id = client.model_catalog.create_model_train_from_version_id(
                model_version.id
            )
            if run_id:
                client.flow_manager.update_run_metadata(
                    run_id=RunId(value=run_id),
                    task_id=model_definition.asset_path.path(),
                    task_type=Task.Type.TYPE_MODEL_TRAIN,
                    key="train-id",
                    value=str(train_id.value),
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


RUNNER = make_runner(_run)
