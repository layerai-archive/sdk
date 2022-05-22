import importlib.util
import os
import pickle  # nosec blacklist
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

from layerapi.api.entity.model_train_status_pb2 import ModelTrainStatus
from layerapi.api.ids_pb2 import ModelTrainId

from layer.clients.model_catalog import ModelCatalogClient


def import_function(
    source_folder: Path, source_entrypoint: str, function_name: Optional[str] = None
) -> Callable[..., Any]:
    """
    Example:
    source_folder: ~/source
    source_entrypoint: models/survival_model/model.py
    function_name: train_model

    If the source_entrypoint is a pickle file, then the function will be deserialized from this file and
    the function_name parameter is unused and should be None.
    Otherwise, the soure_entrypoint.py is loaded as a Python module and the function_name function is
    extracted from that module and returned.
    """
    _, extension = os.path.splitext(source_entrypoint.replace(os.sep, "."))
    if extension.lower() == ".pickle" or extension.lower() == ".pkl":
        return _get_user_function_from_pickle_file(source_folder, source_entrypoint)
    else:
        assert function_name
        return _get_user_function_from_python_source(
            source_folder, source_entrypoint, function_name
        )


def _get_user_function_from_pickle_file(
    source_folder: Path, source_entrypoint: str
) -> Callable[..., Any]:
    expanded = os.path.expanduser(source_folder)
    full_entrypoint_path = os.path.join(expanded, source_entrypoint)
    with open(full_entrypoint_path, "rb") as f:
        return pickle.load(f)  # nosec pickle


def _get_user_function_from_python_source(
    source_folder: Path, source_entrypoint: str, function_name: str
) -> Callable[..., Any]:
    """
    Example input params:
    source_folder: ~/source
    source_entrypoint: models/survival_model/model.py
    function_name: train_model

    Then these vars would have these values:
    module_path: source.survival_model.model
    package: source.survival_model
    """
    expanded = os.path.expanduser(source_folder)
    source_dir = os.path.dirname(expanded)
    sys.path.append(source_dir)  # source becomes a top-level package
    module_path = (
        os.path.basename(source_folder)
        + "."
        + str(Path(source_entrypoint).with_suffix("")).replace(os.path.sep, ".")
    )
    package = (
        os.path.basename(source_folder)
        + "."
        + os.path.dirname(source_entrypoint).replace(os.path.sep, ".")
    )
    user_code_module = importlib.import_module(
        module_path,
        package=package,
    )
    sys.path.remove(source_dir)
    return getattr(user_code_module, function_name)


def update_train_status(
    model_catalog_client: ModelCatalogClient,
    train_id: UUID,
    train_status: "ModelTrainStatus.TrainStatus.V",
    logger: Logger,
    info: str = "",
) -> None:
    try:
        model_catalog_client.update_model_train_status(
            ModelTrainId(value=str(train_id)),
            ModelTrainStatus(train_status=train_status, info=info),
        )
    except Exception as e:
        reason = getattr(e, "message", repr(e))
        logger.error(
            f"Failure while trying to update the status of train ID {str(train_id)}: {reason}"
        )
