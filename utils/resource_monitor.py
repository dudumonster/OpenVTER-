#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
import time
from typing import Any


def _to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def configure_runtime_threads(cpu_threads: int | None = None) -> dict[str, Any]:
    """Limit common native thread pools to the Slurm CPU allocation."""
    resolved = (
        _to_int(os.environ.get("OPENVTER_CPU_THREADS"))
        or _to_int(os.environ.get("SLURM_CPUS_PER_TASK"))
        or _to_int(cpu_threads)
    )
    if resolved is None or resolved <= 0:
        return {"cpu_threads": None, "configured": False}

    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, str(resolved))

    status: dict[str, Any] = {"cpu_threads": resolved, "configured": True}

    try:
        import cv2

        cv2.setNumThreads(resolved)
        status["opencv_threads"] = cv2.getNumThreads()
    except Exception as exc:  # pragma: no cover - depends on local cv2 build.
        status["opencv_threads_error"] = repr(exc)

    try:
        import torch

        torch.set_num_threads(resolved)
        try:
            torch.set_num_interop_threads(max(1, min(2, resolved)))
        except RuntimeError:
            pass
        status["torch_threads"] = torch.get_num_threads()
    except Exception as exc:  # pragma: no cover - torch may be absent locally.
        status["torch_threads_error"] = repr(exc)

    return status


class ResourceMonitor:
    def __init__(self, interval_seconds: float | None = None) -> None:
        if interval_seconds is None:
            interval_seconds = float(os.environ.get("OPENVTER_MONITOR_INTERVAL", "60"))
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._last_log_time = 0.0
        self._process = None
        self._psutil = None
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
            self._process = None

    def should_log(self, force: bool = False) -> bool:
        if force:
            self._last_log_time = time.time()
            return True
        if self.interval_seconds <= 0:
            return False
        now = time.time()
        if now - self._last_log_time >= self.interval_seconds:
            self._last_log_time = now
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self._psutil is not None and self._process is not None:
            try:
                mem = self._process.memory_info()
                data["rss_gb"] = round(mem.rss / (1024**3), 3)
                data["vms_gb"] = round(mem.vms / (1024**3), 3)
                data["proc_cpu_pct"] = round(self._process.cpu_percent(interval=None), 1)
                data["system_cpu_pct"] = round(self._psutil.cpu_percent(interval=None), 1)
            except Exception as exc:
                data["psutil_error"] = repr(exc)
        else:
            data["psutil"] = "unavailable"
            rss_gb = self._rss_from_proc_status()
            if rss_gb is not None:
                data["rss_gb"] = rss_gb

        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                data["cuda_device"] = int(device)
                data["torch_allocated_gb"] = round(
                    torch.cuda.memory_allocated(device) / (1024**3), 3
                )
                data["torch_reserved_gb"] = round(
                    torch.cuda.memory_reserved(device) / (1024**3), 3
                )
                try:
                    data["torch_max_allocated_gb"] = round(
                        torch.cuda.max_memory_allocated(device) / (1024**3), 3
                    )
                except Exception:
                    pass
            else:
                data["cuda"] = "unavailable"
        except Exception as exc:
            data["torch_error"] = repr(exc)

        data.update(self._nvidia_smi())
        return data

    def format(self, **extra: Any) -> str:
        data = self.snapshot()
        data.update({key: value for key, value in extra.items() if value is not None})
        parts = [f"{key}={value}" for key, value in data.items()]
        return "[resource] " + " ".join(parts)

    @staticmethod
    def _nvidia_smi() -> dict[str, Any]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return {}

        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        fields = [item.strip() for item in first_line.split(",")]
        if len(fields) < 3:
            return {}
        try:
            mem_used_mb = float(fields[1])
            mem_total_mb = float(fields[2])
            return {
                "gpu_util_pct": float(fields[0]),
                "gpu_mem_used_gb": round(mem_used_mb / 1024, 3),
                "gpu_mem_total_gb": round(mem_total_mb / 1024, 3),
            }
        except ValueError:
            return {}

    @staticmethod
    def _rss_from_proc_status() -> float | None:
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return round(float(parts[1]) / (1024**2), 3)
        except Exception:
            return None
        return None
