import os
import pathlib
import re
import subprocess  # nosec: import_subprocess
import threading
import time
from enum import Enum
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


class CGroupsMetricsCollectorVersioned:
    METRICS_ROOT = pathlib.Path("/sys/fs/cgroup/")

    def get_cpu_usage(self) -> int:
        raise NotImplementedError()

    def get_cpu_available(self) -> float:
        raise NotImplementedError()

    def get_mem_used(self) -> int:
        raise NotImplementedError()

    def get_mem_available(self) -> int:
        raise NotImplementedError()

    def _read_cgroup_metric(self, metric_path: str) -> Any:
        return _read_from_file(self.METRICS_ROOT / metric_path)


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


class CGroupsMetricsCollectorVersionedV1(CGroupsMetricsCollectorVersioned):
    """
    Use cgroups-v1 to collect system metrics
    """

    def get_cpu_usage(self) -> int:
        # Time in nanoseconds
        return int(self._read_cgroup_metric("cpu/cpuacct.usage_user"))

    def get_cpu_available(self) -> float:
        # Times in microseconds
        cpu_quota = int(self._read_cgroup_metric("cpu/cpu.cfs_quota_us"))
        cpu_period = int(self._read_cgroup_metric("cpu/cpu.cfs_period_us"))
        return float(cpu_quota / cpu_period)

    def get_mem_used(self) -> int:
        return int(self._read_cgroup_metric("memory/memory.usage_in_bytes"))

    def get_mem_available(self) -> int:
        return int(self._read_cgroup_metric("memory/memory.limit_in_bytes"))


class CGroupsMetricsCollectorVersionedV2(CGroupsMetricsCollectorVersioned):
    """
    Use cgroups-v2 to collect system metrics
    """

    def get_cpu_usage(self) -> int:
        # Time expected in nanoseconds
        cpu_usage = str(self._read_cgroup_metric("user.slice/cpu.stat"))
        regex = r"user_usec (\d*)\n"
        extracted_cpu_usage = re.search(regex, cpu_usage)
        if extracted_cpu_usage is None:
            return 0
        else:
            # Time in microseconds, converting to nanoseconds
            return int(extracted_cpu_usage.group(1)) * 1000

    def get_cpu_available(self) -> float:
        max_cpu = str(self._read_cgroup_metric("user.slice/cpu.max"))
        cpu_quota = max_cpu.split()[0]
        cpu_period = max_cpu.split()[1]
        if str(cpu_quota) == "max":
            return 1
        else:
            return float(int(cpu_quota) / int(cpu_period))

    def get_mem_used(self) -> int:
        return int(self._read_cgroup_metric("user.slice/memory.current"))

    def get_mem_available(self) -> int:
        memory_high = self._read_cgroup_metric("user.slice/memory.high")
        if memory_high == "max":
            return int(psutil.virtual_memory().total)
        else:
            return int(memory_high)


class CGroupsVersion(Enum):
    V1 = "v1"
    V2 = "v2"
    UNKNOWN = "unknown"


class CGroupsMetricsCollector(MetricsCollector):
    """
    For Docker metrics collection, we calculate metrics based off cgroup CPU usage data.
    For each iteration, we compare the current usage data with the previous iteration's data,
    allowing us to calculate the difference.
    We capture the data allowing the first loop to run successfully, keeping the logic executed within the loop lean.
    We also wait a second before going into the first loop to ensure the CPU usage data has changed.
    """

    def __init__(
        self, logger: Logger, cgroup_metrics_collector: CGroupsMetricsCollectorVersioned
    ) -> None:
        super().__init__(logger)
        self._cpu_usage_time = time.time()
        self._cgroup_metrics_collector = cgroup_metrics_collector
        self._cpu_usage = self._cgroup_metrics_collector.get_cpu_usage()

    def get_cpu_metrics(self) -> Dict[str, Dict[str, float]]:
        now_time = time.time()
        diff_time = now_time - self._cpu_usage_time
        if diff_time <= 0:
            return {}
        self._cpu_usage_time = now_time

        now_cpu_usage = self._cgroup_metrics_collector.get_cpu_usage()
        diff_cpu = now_cpu_usage - self._cpu_usage
        self._cpu_usage = now_cpu_usage

        cpu_available = self._cgroup_metrics_collector.get_cpu_available()
        cpu_used = diff_cpu / diff_time / 1_000_000_000
        return {
            "cpu": {
                "cpu_used": round(cpu_used, 4),
                "cpu_allocated": round(cpu_available, 2),
            }
        }

    def get_memory_metrics(self) -> Dict[str, Dict[str, float]]:
        mem_used = self._cgroup_metrics_collector.get_mem_used()
        mem_available = self._cgroup_metrics_collector.get_mem_available()
        return {
            "memory": {
                "mem_used_gb": round(float(mem_used / 1024**3), 2),
                "mem_allocated_gb": round(float(mem_available / 1024**3), 2),
            }
        }


def _read_from_file(path: pathlib.Path) -> Any:
    return path.read_text()


def _get_cgroup_version() -> CGroupsVersion:
    """
    Find out the cgroup version used on the machine by finding out the filesystem used for the cgroup mount point.
    """
    path = os.path.abspath("/sys/fs/cgroup/")
    partitions = [
        p for p in psutil.disk_partitions(all=True) if p.mountpoint == path.__str__()
    ]
    if len(partitions) == 1:
        cgroup_mount_type = partitions[0].fstype
        if cgroup_mount_type == "tmpfs":
            return CGroupsVersion.V1
        elif cgroup_mount_type == "cgroup2":
            return CGroupsVersion.V2
        else:
            return CGroupsVersion.UNKNOWN
    else:
        return CGroupsVersion.UNKNOWN


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

        cgroup_version = _get_cgroup_version()
        if cgroup_version == CGroupsVersion.V1:
            self._metrics_collector = CGroupsMetricsCollector(
                self._logger, CGroupsMetricsCollectorVersionedV1()
            )
        elif cgroup_version == CGroupsVersion.V2:
            self._metrics_collector = CGroupsMetricsCollector(
                self._logger, CGroupsMetricsCollectorVersionedV2()
            )
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
