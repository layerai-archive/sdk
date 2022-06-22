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
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

import nvsmi  # type: ignore
import pandas as pd
import polling  # type: ignore
import psutil  # type: ignore
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


metrics_data = []
cpu_used_temp = None  # pylint: disable=C0103
start_time_temp = None  # pylint: disable=C0103


class TrainContext(ABC, TrainContextDataclassMixin):
    is_remote = True

    def init_or_save_context(self, context: Context) -> None:
        set_active_context(context)

    @abstractmethod
    def __enter__(self) -> None:
        pass

    @abstractmethod
    def get_working_directory(self) -> Path:
        pass

    @staticmethod
    def get_metrics(start_time: int, start_cpu_used: int) -> Dict[str, pd.DataFrame]:
        tag = "Remote Fabric Stats"
        global metrics_data  # pylint: disable=invalid-name disable=global-variable-not-assigned
        global cpu_used_temp  # pylint: disable=invalid-name
        global start_time_temp  # pylint: disable=invalid-name

        def read_value_from_file(path: str) -> int:
            with open(path, "r") as f:
                return int(f.read())

        def get_mem_used() -> int:
            return read_value_from_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")

        def get_mem_available() -> int:
            return read_value_from_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")

        def get_used_percent(used: float, available: float) -> float:
            if not used or not available:
                print("System metric 0")
                return 0
            return round((100 * used / available), 2)

        def get_cpu_used() -> int:
            # Time in nanoseconds
            # Multiply by 1000000000 to get to seconds
            return read_value_from_file("/sys/fs/cgroup/cpu/cpuacct.usage_user")

        def get_cpu_available() -> float:
            # Times in microseconds
            cpu_quota = read_value_from_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            cpu_period = read_value_from_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            return float(cpu_quota / cpu_period)

        now_time = int(time.time())
        now_cpu_used = get_cpu_used()
        if start_time_temp is None:
            diff_time = now_time - start_time
        else:
            diff_time = now_time - start_time_temp  # type: ignore

        if cpu_used_temp is None:
            diff_cpu = now_cpu_used - start_cpu_used
        else:
            diff_cpu = now_cpu_used - cpu_used_temp  # type: ignore
        start_time_temp = now_time
        cpu_used_temp = now_cpu_used
        cpus_available = get_cpu_available()
        cpus_used = diff_cpu / diff_time / 1000000000
        fabric_cpu_utilisation_percent = get_used_percent(cpus_used, cpus_available)
        local_now = datetime.now().strftime("%Y%m%dT%H%M%S")
        mem_used = get_mem_used()
        mem_available = get_mem_available()
        mem_used_percent = get_used_percent(mem_used, mem_available)

        gpu_present = nvsmi.is_nvidia_smi_on_path() is not None
        if gpu_present:
            gpus = nvsmi.get_gpus()
            gpu0 = next(gpus, None)  # Only handle a single GPU for now
            gpu0_utilisation = gpu0.gpu_util  # type: ignore
            gpu0_mem_utilisation = gpu0.mem_util  # type: ignore
            gpu0_mem_used = gpu0.mem_used  # type: ignore
            gpu0_mem_total = gpu0.mem_total  # type: ignore
        else:
            gpu0_utilisation = -1
            gpu0_mem_utilisation = -1
            gpu0_mem_used = -1
            gpu0_mem_total = -1

        metrics_data.append(
            [
                local_now,
                round(float(mem_used / 1024 / 1024), 2),
                round(float(mem_available / 1024 / 1024), 2),
                mem_used_percent,
                round(cpus_used, 4),
                round(cpus_available, 2),
                fabric_cpu_utilisation_percent,
                gpu0_utilisation,
                gpu0_mem_used,
                gpu0_mem_total,
                round(gpu0_mem_utilisation, 2),
            ]
        )
        dataframe = pd.DataFrame(
            metrics_data,
            columns=[
                "Timestamp",
                "Memory Used (MB)",
                "Memory Allocated (MB)",
                "Memory Utilisation %",
                "CPUs Used",
                "CPUs Allocated",
                "CPU Utilisation %",
                "GPU Utilisation %",
                "GPU Memory Used (MB)",
                "GPU Memory Allocated (MB)",
                "GPU Memory Utilisation %",
            ],
        )
        dataframe.set_index("Timestamp", inplace=True)  # type: ignore
        return {tag: dataframe}

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
    is_remote = False

    def __enter__(self) -> None:
        super().__enter__()
        self.initial_cwd = os.getcwd()

    def get_working_directory(self) -> Path:
        assert self.initial_cwd
        return Path(self.initial_cwd)

    @staticmethod
    def get_metrics(start_time: int, start_cpu_used: int) -> Dict[str, pd.DataFrame]:
        tag = "Local System Stats"
        local_now = datetime.now().strftime("%Y%m%dT%H%M%S")
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_used = cpu_count * cpu_percent / 100

        print("cpu_used:", round(cpu_used, 2), "CPUs")
        print("cpu_count:", cpu_count, "CPUs")
        print("cpu_percent:", round(cpu_percent, 2), "%")

        mem = psutil.virtual_memory()
        mem_allocated = mem.total
        mem_used = mem_allocated - mem.available
        mem_utilisation = mem.percent

        print("mem_used:", round((mem_used / 1024 / 1024), 2), "MB")
        print("mem_allocated:", round((mem_allocated / 1024 / 1024), 2), "MB")
        print("mem_utilisation:", round(mem_utilisation, 2), "%")

        metrics_data.append(
            [
                local_now,
                round((mem_used / 1024 / 1024), 2),
                round((mem_allocated / 1024 / 1024), 2),
                round(mem_utilisation, 2),
                round(cpu_used, 2),
                cpu_count,
                round(cpu_percent, 2),
                -1,
                -1,
                -1,
                -1,
            ]
        )
        dataframe = pd.DataFrame(
            metrics_data,
            columns=[
                "Timestamp",
                "Memory Used (MB)",
                "Memory Allocated (MB)",
                "Memory Utilisation %",
                "CPUs Used",
                "CPUs Allocated",
                "CPU Utilisation %",
                "GPU Utilisation %",
                "GPU Memory Used (MB)",
                "GPU Memory Allocated (MB)",
                "GPU Memory Utilisation %",
            ],
        )
        dataframe.set_index("Timestamp", inplace=True)  # type: ignore
        return {tag: dataframe}

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

                def monitor_system_metrics(stop: bool) -> None:
                    start_time = 0
                    start_cpu_used = 0
                    log_data_runner = LogDataRunner(
                        client=self.client,
                        train_id=self.train_context.train_id,
                        logger=self.logger,
                    )
                    if self.train_context.is_remote:
                        start_time = int(time.time())
                        with open("/sys/fs/cgroup/cpu/cpuacct.usage_user", "r") as f:
                            start_cpu_used = int(f.read())

                    sleep(1)  # helps keep things simple
                    polling.poll(
                        lambda: log_data_runner.log(
                            self.train_context.get_metrics(start_time, start_cpu_used)  # type: ignore
                        ),
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
                sleep(1)  # temporarily make things slower
                self.logger.info("Executing the train_model_func")
                work_dir = self.train_context.get_working_directory()
                os.chdir(work_dir)
                self.logger.info("Downloading resources")
                ResourceManager(self.client).wait_resource_download(
                    project_full_name,
                    train_model_func.__name__,
                    target_dir=str(work_dir),
                )
                sleep(1)  # temporarily make things slower
                model = train_model_func()
                sleep(1)  # temporarily make things slower
                self.tracker.mark_model_trained(
                    self.train_context.model_name,
                )
                sleep(1)  # temporarily make things slower
                self.logger.info("Executed train_model_func successfully")
                self._run_assertions(
                    model,
                    train_model_func.layer.get_assertions(),  # type: ignore
                )
                self.tracker.mark_model_saving(self.train_context.model_name)
                self.logger.info(f"Saving model artifact {model} to model registry")
                train.save_model(model, tracker=self.tracker)
                sleep(1)  # temporarily make things slower
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
