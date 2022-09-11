import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import grpc

import layer
from layer import Context, global_context
from layer.clients.layer import LayerClient
from layer.config import ConfigManager
from layer.config.config import Config
from layer.contracts.definitions import FunctionDefinition
from layer.contracts.fabrics import Fabric
from layer.exceptions.exceptions import (
    LayerFailedAssertionsException,
    RuntimeMemoryException,
)
from layer.global_context import set_has_shown_update_message
from layer.logged_data.logged_data_destination import LoggedDataDestination
from layer.logged_data.queuing_logged_data_destination import (
    QueueingLoggedDataDestination,
)
from layer.logged_data.system_metrics import SystemMetrics
from layer.projects.utils import verify_project_exists_and_retrieve_project_id
from layer.tracker.progress_tracker import RunProgressTracker
from layer.tracker.utils import get_progress_tracker
from layer.utils.async_utils import asyncio_run_in_thread


logger = logging.getLogger(__name__)

ENV_LAYER_API_URL = "LAYER_API_URL"
ENV_LAYER_API_KEY = "LAYER_API_KEY"
ENV_LAYER_API_TOKEN = "LAYER_API_TOKEN"
ENV_LAYER_RUN_ID = "LAYER_RUN_ID"
ENV_LAYER_FABRIC = "LAYER_FABRIC"


class FunctionRunner(ABC):
    config: Config
    client: LayerClient
    logged_data_destination: LoggedDataDestination
    tracker: RunProgressTracker
    project_id: uuid.UUID

    def __init__(self, definition: FunctionDefinition) -> None:
        self.definition = definition

    def __call__(self) -> Any:
        self._run_prep()
        self.config: Config = asyncio_run_in_thread(ConfigManager().refresh())
        progress_tracker = get_progress_tracker(
            url=self.config.url,
            project_name=self.definition.project_name,
            account_name=self.definition.account_name,
        )
        with LayerClient(
            self.config.client, logger
        ).init() as client, progress_tracker.track() as tracker:
            self.client = client
            self.tracker = tracker
            self.project_id = verify_project_exists_and_retrieve_project_id(
                self.client, self.definition.project_full_name
            )
            with QueueingLoggedDataDestination(
                client=client.logged_data_service_client
            ) as logged_data_destination:
                self.logged_data_destination = logged_data_destination
                return self._run()

    def _run(self) -> Any:
        self.tracker.add_asset(self.definition.asset_type, self.definition.asset_name)

        tag, context_kwargs = self._create_asset()

        self.tracker.mark_running(
            asset_type=self.definition.asset_type,
            name=self.definition.asset_name,
            tag=tag,
        )

        with Context(
            url=self.config.url,
            asset_path=self.definition.asset_path,
            client=self.client,
            tracker=self.tracker,
            logged_data_destination=self.logged_data_destination,
            **context_kwargs,
        ) as ctx:
            ctx._label_asset_with(  # pylint: disable=W0212
                global_context.current_label_names()
            )
            self._mark_start()

            try:
                output = self._run_main(ctx)
            except Exception as exc:
                failure_exc = exc
                if isinstance(exc, grpc.RpcError):
                    failure_exc = _transform_grpc_exception(exc)
                elif isinstance(exc, MemoryError):
                    failure_exc = _transform_memory_error_exception(exc)
                tag = self._mark_failure(failure_exc)
                self.tracker.mark_failed(
                    self.definition.asset_type,
                    self.definition.asset_name,
                    tag=tag,
                    reason=str(failure_exc),
                )
                raise failure_exc
            except SystemExit as exc:
                failure_exc = _transform_system_exit_error(exc)
                tag = self._mark_failure(failure_exc)
                self.tracker.mark_failed(
                    self.definition.asset_type,
                    self.definition.asset_name,
                    tag=tag,
                    reason=str(failure_exc),
                )
                raise exc
            else:
                tag = self._mark_success()
                self.tracker.mark_done(
                    self.definition.asset_type,
                    self.definition.asset_name,
                    tag=tag,
                    warnings=self.logged_data_destination.close_and_get_errors(),
                )

        return output

    def _run_prep(self) -> None:
        # do not show update warnings
        set_has_shown_update_message(True)

        # TODO This is too deep, why do we need to alter global context from inside?
        global_context.set_current_project_full_name(self.definition.project_full_name)

        # login
        api_url = os.environ.get(ENV_LAYER_API_URL)
        api_key = os.environ.get(ENV_LAYER_API_KEY)
        api_token = os.environ.get(ENV_LAYER_API_TOKEN)

        if api_url:
            if api_key:
                layer.login_with_api_key(api_key, url=api_url)
            elif api_token:
                layer.login_with_access_token(api_token, url=api_url)

    def _run_main(self, ctx: Context) -> Any:
        logger.info("Executing the function")

        work_dir = ctx.get_working_directory()
        os.chdir(work_dir)

        with SystemMetrics(logger):
            output = self.definition.func(
                *self.definition.args, **self.definition.kwargs
            )

        logger.info("Executed function successfully")

        if output is not None:
            output = self._transform_output(output)
            self._run_assertions(output)
            self._save_artifact(ctx, output)

        return output

    def _run_assertions(self, output: Any) -> None:
        failed_assertions = []
        self.tracker.mark_asserting(
            self.definition.asset_type, self.definition.asset_name
        )
        for assertion in reversed(self.definition.assertions):
            try:
                self.tracker.mark_asserting(
                    self.definition.asset_type,
                    self.definition.asset_name,
                    assertion=assertion,
                )
                assertion.function(output)
            except Exception:
                failed_assertions.append(assertion)
        if len(failed_assertions) > 0:
            self.tracker.mark_failed_assertions(
                self.definition.asset_type,
                self.definition.asset_name,
                failed_assertions,
            )
            raise LayerFailedAssertionsException(failed_assertions)
        else:
            self.tracker.mark_asserted(
                self.definition.asset_type, self.definition.asset_name
            )

    @property
    def fabric(self) -> Fabric:
        return Fabric.find(os.getenv(ENV_LAYER_FABRIC, "f-local"))

    @property
    def run_id(self) -> Optional[uuid.UUID]:
        run_id_str = os.getenv(ENV_LAYER_RUN_ID)
        if run_id_str is None:
            return None
        return uuid.UUID(run_id_str)

    def _transform_output(self, output: Any) -> Any:
        return output

    @abstractmethod
    def _create_asset(self) -> Tuple[str, Dict[str, Any]]:
        ...

    @abstractmethod
    def _mark_start(self) -> None:
        ...

    @abstractmethod
    def _mark_success(self) -> str:
        ...

    @abstractmethod
    def _mark_failure(self, failure_exc: Exception) -> str:
        ...

    @abstractmethod
    def _save_artifact(self, ctx: Context, artifact: Any) -> None:
        ...


def _transform_system_exit_error(e: SystemExit) -> Exception:
    # To populate stacktrace
    try:
        raise Exception(f"System exit with code:{e.code}")
    except Exception as ee:
        return ee


def _transform_memory_error_exception(e: MemoryError) -> Exception:
    # To populate stacktrace
    try:
        raise RuntimeMemoryException(str(e))
    except Exception as ee:
        return ee


def _transform_grpc_exception(exc: grpc.RpcError) -> Exception:
    """
    Done to populate sys.exc_info() with an exception containing the stacktrace
    of the input param exception, so that traceback.print_exc(), which we use
    to print debug logs, would print the complete exception stacktrace as intended.
    :param exc: The exception, whose error message we want to extract + retain its stacktrace
    """
    failure_reason = exc.details()
    try:
        raise Exception(failure_reason) from exc
    except Exception as e:
        return e
