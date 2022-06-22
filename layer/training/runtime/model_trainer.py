import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
from time import sleep
from types import TracebackType
from typing import Any, Callable, List, Optional
from uuid import UUID

import pandas as pd
import polling
from layerapi.api.entity.model_train_status_pb2 import ModelTrainStatus

from layer import Context
from layer.clients.layer import LayerClient
from layer.contracts.assertions import Assertion
from layer.exceptions.exception_handler import exception_handler
from layer.exceptions.exceptions import LayerFailedAssertionsException
from layer.global_context import (
    current_project_full_name,
    reset_active_context,
    set_active_context,
)
from layer.logged_data.log_data_runner import LogDataRunner
from layer.resource_manager import ResourceManager
from layer.tracker.progress_tracker import RunProgressTracker
from layer.training.train import Train

from .common import import_function, update_train_status
from .model_train_failure_reporter import ModelTrainFailureReporter


@dataclass
class TrainContextDataclassMixin:
    model_name: str
    model_version: str
    train_id: UUID
    source_entrypoint: str
    source_folder: Path
    logger: Logger
    train_index: Optional[str] = None


class TrainContext(ABC, TrainContextDataclassMixin):
    def init_or_save_context(self, context: Context) -> None:
        set_active_context(context)

    @abstractmethod
    def __enter__(self) -> None:
        pass

    @abstractmethod
    def get_working_directory(self) -> Path:
        pass

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        reset_active_context()


@dataclass
class LocalTrainContext(TrainContext):
    initial_cwd: Optional[str] = None

    def __enter__(self) -> None:
        super().__enter__()
        self.initial_cwd = os.getcwd()

    def get_working_directory(self) -> Path:
        assert self.initial_cwd
        return Path(self.initial_cwd)

    def __exit__(
        self,
        exc_type: Optional[BaseException],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        assert self.initial_cwd
        os.chdir(
            self.initial_cwd
        )  # Important for local execution to have no such side effect
        return None


start_cpu_used = None


@dataclass(frozen=True)
class ModelTrainer:
    client: LayerClient
    train_context: TrainContext
    logger: Logger
    failure_reporter: ModelTrainFailureReporter
    tracker: RunProgressTracker

    def train(self) -> Any:
        self.tracker.mark_model_training(
            self.train_context.model_name,
            version=self.train_context.model_version,
            train_idx=self.train_context.train_index,
        )
        try:
            with self.train_context:
                return self._train(callback=self.failure_reporter.report_failure)
        except Exception:
            self.logger.error(
                f"Error performing model training with id: {self.train_context.train_id}",
                exc_info=True,
            )
            import sys
            import traceback

            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

    def _run_assertions(self, model: Any, assertions: List[Assertion]) -> None:
        failed_assertions = []
        self.tracker.mark_model_running_assertions(self.train_context.model_name)
        for assertion in reversed(assertions):
            try:
                self.tracker.mark_model_running_assertion(
                    self.train_context.model_name, assertion
                )
                assertion.function(model)
            except Exception:
                failed_assertions.append(assertion)
        if len(failed_assertions) > 0:
            self.tracker.mark_model_failed_assertions(
                self.train_context.model_name, failed_assertions
            )
            raise LayerFailedAssertionsException(failed_assertions)
        else:
            self.tracker.mark_model_completed_assertions(self.train_context.model_name)

    @exception_handler(stage="Training run")
    def _train(
        self, callback: Optional[Callable[[str, Exception], None]] = None
    ) -> Any:
        self.logger.info(
            f"Importing user code({self.train_context.source_entrypoint}) from {self.train_context.source_folder}"
        )
        train_model_func = import_function(
            self.train_context.source_folder,
            self.train_context.source_entrypoint,
            "train_model",
        )
        self.logger.info("train_model_func function imported successfully")
        project_full_name = current_project_full_name()
        if not project_full_name:
            raise Exception("Internal Error: missing current project full name")
        with Context() as context:
            with Train(
                layer_client=self.client,
                name=self.train_context.model_name,
                project_full_name=project_full_name,
                version=self.train_context.model_version,
                train_id=self.train_context.train_id,
            ) as train:
                context.with_train(train)
                context.with_tracker(self.tracker)
                context.with_asset_name(self.train_context.model_name)
                self.train_context.init_or_save_context(context)
                update_train_status(
                    self.client.model_catalog,
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_FETCHING_FEATURES,
                    self.logger,
                )

                log_data_runner = LogDataRunner(
                    client=self.client,
                    train_id=self.train_context.train_id,
                    logger=self.logger,
                )
                tag = "Fabric Performance"

                def get_metrics(start_time):
                    global start_cpu_used

                    def read_value_from_file(path: str) -> int:
                        with open(path, "r") as f:
                            return int(f.read())

                    def get_mem_used() -> int:
                        return read_value_from_file(
                            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
                        )

                    def get_mem_available() -> int:
                        return read_value_from_file(
                            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
                        )

                    def get_mem_used_percent(used: int, available: int) -> float:
                        if not used or not available:
                            print("Memory metric 0")
                            return 0
                        return round((100 * float(used) / float(available)), 2)

                    def get_cpu_used() -> int:
                        # Time in nanoseconds
                        # Multiply by 1000000000 to get to seconds
                        return read_value_from_file(
                            "/sys/fs/cgroup/cpu/cpuacct.usage_user"
                        )

                    def get_cpu_available() -> float:
                        # Times in microseconds
                        cpu_quota = read_value_from_file(
                            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
                        )
                        cpu_period = read_value_from_file(
                            "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
                        )
                        return float(cpu_quota / cpu_period)

                    start_cpu_used = start_cpu_used or get_cpu_used()
                    now_time = int(time.time())
                    now_cpu_used = get_cpu_used()
                    diff_time = now_time - start_time
                    diff_cpu = now_cpu_used - start_cpu_used
                    cpus_available = get_cpu_available()
                    cpus_used = float(diff_cpu / diff_time / 1000000000)
                    print(f"CPUs used: {cpus_used:.2f} ({cpus_available} available)")
                    fabric_cpu_utilisation_percent = cpus_used * 100 / cpus_available
                    print(
                        f"Fabric's CPU Utilisation: {fabric_cpu_utilisation_percent:.2f} %"
                    )
                    local_now = datetime.now().strftime("%Y%m%dT%H%M%S")
                    mem_used = get_mem_used()
                    mem_used_percent = get_mem_used_percent(
                        mem_used, get_mem_available()
                    )

                    dataframe = pd.DataFrame(
                        [
                            [
                                local_now,
                                mem_used,
                                mem_used_percent,
                                cpus_used,
                                fabric_cpu_utilisation_percent,
                            ],
                        ],
                        columns=[
                            "Timestamp",
                            "Used Memory",
                            "Memory Utilisation",
                            "Used CPUs",
                            "Fabric's CPU Utilisation %",
                        ],
                        index_col="Timestamp",
                    )
                    log_data_runner.log({tag: dataframe})

                def monitor_system_metrics(stop):
                    start_time = int(time.time())
                    sleep(1)  # helps keep things simple
                    polling.poll(
                        lambda: get_metrics(start_time),
                        step=1,  # anything under 1 second risks divisions by zero, stick with >=1
                        check_success=stop,
                        poll_forever=True,
                    )

                stop_system_metrics_thread = False
                system_metrics_thread = threading.Thread(
                    target=monitor_system_metrics,
                    args=(lambda x: stop_system_metrics_thread,),
                )
                system_metrics_thread.start()

                update_train_status(
                    self.client.model_catalog,
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_IN_PROGRESS,
                    self.logger,
                )
                self.logger.info("Executing the train_model_func")
                work_dir = self.train_context.get_working_directory()
                os.chdir(work_dir)
                self.logger.info("Downloading resources")
                ResourceManager(self.client).wait_resource_download(
                    project_full_name,
                    train_model_func.__name__,
                    target_dir=str(work_dir),
                )
                model = train_model_func()
                self.tracker.mark_model_trained(
                    self.train_context.model_name,
                )
                self.logger.info("Executed train_model_func successfully")
                self._run_assertions(
                    model,
                    train_model_func.layer.get_assertions(),  # type: ignore
                )
                self.tracker.mark_model_saving(self.train_context.model_name)
                self.logger.info(f"Saving model artifact {model} to model registry")
                train.save_model(model, tracker=self.tracker)
                update_train_status(
                    self.client.model_catalog,
                    self.train_context.train_id,
                    ModelTrainStatus.TRAIN_STATUS_SUCCESSFUL,
                    self.logger,
                )
                self.logger.info(
                    f"Saved model artifact {model} to model registry successfully"
                )
                self.tracker.mark_model_saved(self.train_context.model_name)
                stop_system_metrics_thread = True
                system_metrics_thread.join()
                return model
