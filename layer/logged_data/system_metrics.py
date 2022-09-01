import os
import pathlib
import subprocess  # nosec: import_subprocess
import threading
import time
from logging import Logger
from typing import Any, Dict, Optional

import nvsmi  # type: ignore
import polling  # type: ignore
import psutil  # type: ignore

from layer.context import get_active_context
from layer.contracts.logged_data import XCoordinateType
from layer.logged_data.log_data_runner import LogDataRunner


class MetricsCollector:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def get_cpu_metrics(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError()

    def get_memory_metrics(self) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError()

    def get_gpu_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        gpu_present = nvsmi.is_nvidia_smi_on_path() is not None
        if gpu_present:
            try:
                gpus = nvsmi.get_gpus()
            except subprocess.CalledProcessError:
                self._logger.info(
                    "Nvidia driver not running despite nvidia-smi being on the path. No GPU stats collected."
                )
                return metrics
            for gpu in gpus:
                metrics.setdefault("gpu_utilization", {})
                metrics["gpu_utilization"].update(
                    {
                        f"gpu{gpu.id}_utilization_percent": gpu.gpu_util,
                    }
                )
                metrics.setdefault("gpu_memory_utilization", {})
                metrics["gpu_memory_utilization"].update(
                    {
                        f"gpu{gpu.id}_mem_utilisation_percent": round(gpu.mem_util, 2),
                    }
                )
                metrics.setdefault("gpu_memory", {})
                metrics["gpu_memory"].update(
                    {
                        f"gpu{gpu.id}_mem_used_mb": gpu.mem_used,
                        f"gpu{gpu.id}_mem_allocated_mb": gpu.mem_total,
                    }
                )

        return metrics


class PsUtilMetricsCollector(MetricsCollector):
    def get_cpu_metrics(self) -> Dict[str, Dict[str, float]]:
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_used = cpu_count * cpu_percent / 100
        return {
            "cpu": {
                "cpu_used": round(cpu_used, 2),
                "cpu_allocated": cpu_count,
            }
        }

    def get_memory_metrics(self) -> Dict[str, Dict[str, float]]:
        mem = psutil.virtual_memory()
        mem_allocated = mem.total
        mem_used = mem_allocated - mem.available
        return {
            "memory": {
                "mem_used_gb": round(float(mem_used / 1024**3), 2),
                "mem_allocated_gb": round(float(mem_allocated / 1024**3), 2),
            }
        }


class DockerMetricsCollector(MetricsCollector):
    """
    For Docker metrics collection, we calculate metrics based off cgroup CPU usage data.
    For each iteration, we compare the current usage data with the previous iteration's data, allowing us to calculate the difference.
    We capture the data which allows the first loop to run successfully, keeping the logic executed within the loop lean.
    We also wait a second before going into the first loop to ensure the CPU usage data has changed.
    """

    METRICS_ROOT = pathlib.Path("/sys/fs/cgroup/")

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self._cpu_usage_time = time.time()
        self._cpu_usage = self._get_cpu_usage()

    def _read_docker_metric(self, metric_path: str) -> int:
        return _read_int_from_file(self.METRICS_ROOT / metric_path)

    def get_cpu_metrics(self) -> Dict[str, Dict[str, float]]:
        now_time = time.time()
        diff_time = now_time - self._cpu_usage_time
        if diff_time <= 0:
            return {}
        self._cpu_usage_time = now_time

        now_cpu_usage = self._get_cpu_usage()
        diff_cpu = now_cpu_usage - self._cpu_usage
        self._cpu_usage = now_cpu_usage

        cpu_available = self._get_cpu_available()
        cpu_used = diff_cpu / diff_time / 1_000_000_000
        return {
            "cpu": {
                "cpu_used": round(cpu_used, 4),
                "cpu_allocated": round(cpu_available, 2),
            }
        }

    def _get_cpu_usage(self) -> int:
        # Time in nanoseconds
        # Multiply by 1000000000 to get to seconds
        return self._read_docker_metric("cpu/cpuacct.usage_user")

    def _get_cpu_available(self) -> float:
        # Times in microseconds
        cpu_quota = self._read_docker_metric("cpu/cpu.cfs_quota_us")
        cpu_period = self._read_docker_metric("cpu/cpu.cfs_period_us")
        return float(cpu_quota / cpu_period)

    def get_memory_metrics(self) -> Dict[str, Dict[str, float]]:
        mem_used = self._get_mem_used()
        mem_available = self._get_mem_available()
        return {
            "memory": {
                "mem_used_gb": round(float(mem_used / 1024**3), 2),
                "mem_allocated_gb": round(float(mem_available / 1024**3), 2),
            }
        }

    def _get_mem_used(self) -> int:
        return self._read_docker_metric("memory/memory.usage_in_bytes")

    def _get_mem_available(self) -> int:
        return self._read_docker_metric("memory/memory.limit_in_bytes")


def _read_int_from_file(path: pathlib.Path) -> int:
    return int(path.read_text())


class SystemMetrics:
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger: Logger = logger
        self._log_data_runner: Optional[LogDataRunner] = None
        self._metrics_collector: Optional[MetricsCollector] = None

        self._stop_system_metrics_thread: bool = False
        self._start_time: float = time.time()
        self._elapsed_seconds: int = 0
        self._system_metrics_thread: threading.Thread = threading.Thread(
            target=self.monitor_system_metrics,
        )

    def __enter__(self) -> None:
        # Build the log_data_runner
        active_context = get_active_context()
        if not active_context:
            raise RuntimeError(
                "System stats logging only allowed inside functions decorated with @model or @dataset"
            )
        logged_data_destination = active_context.logged_data_destination()
        if not logged_data_destination:
            raise RuntimeError(
                "System stats logging only allowed with a logged data destination"
            )
        train = active_context.train()
        train_id = train.get_id() if train is not None else None
        dataset_build = active_context.dataset_build()
        dataset_build_id = dataset_build.id if dataset_build is not None else None
        self._log_data_runner = LogDataRunner(
            train_id=train_id,
            dataset_build_id=dataset_build_id,
            logger=self._logger,
            logged_data_destination=logged_data_destination,
        )

        # By default, we get metrics from psutil
        if "LAYER_FABRIC" in os.environ:
            self._metrics_collector = DockerMetricsCollector(self._logger)
        else:
            self._metrics_collector = PsUtilMetricsCollector(self._logger)

        # Start collecting metrics
        self._system_metrics_thread.start()

    def __exit__(
        self, exception_type: Any, exception_value: Any, traceback: Any
    ) -> None:
        self._log_system_metrics()  # log one last time before exiting
        self._stop_system_metrics_thread = True
        self._system_metrics_thread.join()

    def monitor_system_metrics(self) -> None:

        polling.poll(
            self._log_system_metrics,
            check_success=lambda x: self._stop_system_metrics_thread,
            poll_forever=True,
            step=1,
            step_function=self._metrics_step_function,
        )

    def _log_system_metrics(self) -> None:
        assert self._metrics_collector
        assert self._log_data_runner
        self._elapsed_seconds = round(time.time() - self._start_time)
        for group_metrics in [
            self._metrics_collector.get_cpu_metrics(),
            self._metrics_collector.get_memory_metrics(),
            self._metrics_collector.get_gpu_metrics(),
        ]:
            for group, metrics in group_metrics.items():
                self._log_data_runner.log(
                    metrics,
                    x_coordinate=self._elapsed_seconds,
                    x_coordinate_type=XCoordinateType.TIME,
                    group_tag=group,
                    category="System",
                )

    def _metrics_step_function(self, step: int) -> int:
        # ensure that short runs still get at least a couple of data points by polling
        # more frequently in the beginning
        if self._elapsed_seconds < 15:
            step = 1
        elif self._elapsed_seconds < 60:
            step = 5
        else:
            step = 15
        return step
