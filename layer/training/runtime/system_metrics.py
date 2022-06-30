import subprocess  # nosec: import_subprocess
import threading
import time
from logging import Logger
from time import sleep
from typing import Any, Dict, Tuple
from uuid import UUID

import nvsmi  # type: ignore
import polling  # type: ignore
import psutil  # type: ignore

from layer.clients.layer import LayerClient
from layer.logged_data.log_data_runner import LogDataRunner


class SystemMetrics:
    def __init__(
        self,
        client: LayerClient,
        train_id: UUID,
        logger: Logger,
        local: bool,
    ) -> None:
        self._client: LayerClient = client
        self._train_id: UUID = train_id
        self._logger: Logger = logger
        self._local: bool = local
        self._cpu_used_temp: int = 0
        self._cpu_used: int = 0
        self._start_time_temp: int = 0
        self._start_time: int = 0
        self._start_cpu_used: int = 0
        self._step_value: int = 0
        self._stop_system_metrics_thread: bool = False
        self._system_metrics_thread: threading.Thread = threading.Thread(
            target=self.monitor_system_metrics,
        )

    def __enter__(self) -> None:
        self._system_metrics_thread.start()

    def __exit__(
        self, exception_type: Any, exception_value: Any, traceback: Any
    ) -> None:
        self._stop_system_metrics_thread = True
        self._system_metrics_thread.join()

    def _get_gpu_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Any] = {}
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
                metrics[gpu.id] = {
                    "utilisation": gpu.gpu_util,
                    "mem_utilisation": round(gpu.mem_util, 2),
                    "mem_used": gpu.mem_used,
                    "mem_total": gpu.mem_total,
                    "id": gpu.id,
                    "name": gpu.name,
                }

        return metrics

    @staticmethod
    def _generate_system_metrics_dict(
        mem_used: float,
        mem_allocated: float,
        mem_used_percent: float,
        cpus_used: float,
        cpus_available: float,
        cpu_utilisation_percent: float,
        gpu_metrics: Dict[Any, Any],
    ) -> Dict[str, float]:
        metrics = {
            "Memory Used (MB)": mem_used,
            "Memory Allocated (MB)": mem_allocated,
            "Memory Utilisation %": mem_used_percent,
            "CPUs Used": cpus_used,
            "CPUs Allocated": cpus_available,
            "CPU Utilisation %": cpu_utilisation_percent,
        }

        for gpu in gpu_metrics:
            gpu_id = gpu_metrics[gpu]["id"]
            gpu_name = gpu_metrics[gpu]["name"]
            gpu_id_name_suffix = f" - gpu{gpu_id} - {gpu_name}"
            metrics.update(
                {
                    "GPU Utilisation %"
                    + gpu_id_name_suffix: gpu_metrics[gpu]["utilisation"],
                    "GPU Memory Used (MB)"
                    + gpu_id_name_suffix: gpu_metrics[gpu]["mem_used"],
                    "GPU Memory Allocated (MB)"
                    + gpu_id_name_suffix: gpu_metrics[gpu]["mem_total"],
                    "GPU Memory Utilisation %"
                    + gpu_id_name_suffix: gpu_metrics[gpu]["mem_utilisation"],
                }
            )
        return metrics

    def _generate_remote_system_metrics(self) -> Tuple[Dict[str, float], int]:
        def _read_value_from_file(path: str) -> int:
            with open(path, "r") as f:
                return int(f.read())

        def _get_mem_used() -> int:
            return _read_value_from_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")

        def _get_mem_available() -> int:
            return _read_value_from_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")

        def _get_used_percent(used: float, available: float) -> float:
            if not available:
                # If we try to divide by 0, return 0
                return 0
            return round((100 * used / available), 2)

        def _get_cpu_used() -> int:
            # Time in nanoseconds
            # Multiply by 1000000000 to get to seconds
            return _read_value_from_file("/sys/fs/cgroup/cpu/cpuacct.usage_user")

        def _get_cpu_available() -> float:
            # Times in microseconds
            cpu_quota = _read_value_from_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            cpu_period = _read_value_from_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            return float(cpu_quota / cpu_period)

        now_time = int(time.time())
        now_cpu_used = _get_cpu_used()
        if self._start_time_temp == 0:
            diff_time = now_time - self._start_time
        else:
            diff_time = now_time - self._start_time_temp

        if self._cpu_used_temp == 0:
            diff_cpu = now_cpu_used - self._start_cpu_used
        else:
            diff_cpu = now_cpu_used - self._cpu_used_temp
        self._start_time_temp = now_time
        self._cpu_used_temp = now_cpu_used
        cpus_available = _get_cpu_available()
        cpus_used = diff_cpu / diff_time / 1000000000
        fabric_cpu_utilisation_percent = _get_used_percent(cpus_used, cpus_available)
        mem_used = _get_mem_used()
        mem_available = _get_mem_available()
        mem_used_percent = _get_used_percent(mem_used, mem_available)

        metrics = self._generate_system_metrics_dict(
            round(float(mem_used / 1024 / 1024), 2),
            round(float(mem_available / 1024 / 1024), 2),
            mem_used_percent,
            round(cpus_used, 4),
            round(cpus_available, 2),
            fabric_cpu_utilisation_percent,
            self._get_gpu_metrics(),
        )

        return (metrics, self._step_value)

    def _generate_local_system_metrics(self) -> Tuple[Dict[str, float], int]:
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_used = cpu_count * cpu_percent / 100

        mem = psutil.virtual_memory()
        mem_allocated = mem.total
        mem_used = mem_allocated - mem.available
        mem_utilisation = mem.percent

        metrics = self._generate_system_metrics_dict(
            round((mem_used / 1024 / 1024), 2),
            round((mem_allocated / 1024 / 1024), 2),
            round(mem_utilisation, 2),
            round(cpu_used, 2),
            cpu_count,
            round(cpu_percent, 2),
            self._get_gpu_metrics(),
        )
        return (metrics, self._step_value)

    def _generate_system_metrics(
        self,
        log_data_runner: LogDataRunner,
    ) -> None:
        metrics = None
        step = None
        metrics, step = (
            self._generate_local_system_metrics()
            if self._local
            else self._generate_remote_system_metrics()
        )
        log_data_runner.log(metrics, step)  # type: ignore

    def monitor_system_metrics(
        self,
    ) -> None:
        log_data_runner = LogDataRunner(
            client=self._client,
            train_id=self._train_id,
            logger=self._logger,
        )

        def _metrics_step_function(step: int) -> int:
            step += 1
            self._step_value += step
            return min(step, 15)

        # For remote system metrics collection, we calculate metrics based off cgroup CPU usage data.
        # For each iteration, we compare the current usage data with the previous iteration's data, allowing us to calculate the difference.
        # We capture the data which allows the first loop to run successfully, keeping the logic executed within the loop lean.
        # We also wait a second before going into the first loop to ensure the CPU usage data has changed.
        if not self._local:
            self._start_time = int(time.time())
            with open("/sys/fs/cgroup/cpu/cpuacct.usage_user", "r") as f:
                self._start_cpu_used = int(f.read())
            sleep(1)

        polling.poll(
            lambda: self._generate_system_metrics(log_data_runner),
            check_success=lambda x: self._stop_system_metrics_thread,
            poll_forever=True,
            # get metrics every 1, then 2, then 3, then (...) until 15 seconds, then every 15 seconds
            # This will ensure that short trains still get at least a couple of data points
            step=1,
            step_function=_metrics_step_function,
        )
