"""Memory protection: probes and the watchdog loop (section 8, ring 2).

The watchdog polls free device memory every few tens of milliseconds. Below the
low watermark it frees slabs at once, standbys first, until the high watermark is
restored. A free needs no save and no network call.
"""

from __future__ import annotations

import logging
import threading
import time

from .slabs import SlabManager

log = logging.getLogger(__name__)


class SimulatedProbe:
    """A GPU with ``total`` bytes and a host job using ``host_used``. For CPU runs,
    tests and the reclaim benchmark. ``host_activity`` models a host whose work rate
    drops by ``sensitivity`` times our duty cycle (for exercising the AIMD loop)."""

    def __init__(self, total: int, host_used: int = 0, sensitivity: float = 0.0):
        self.total = total
        self.host_used = host_used
        self.sensitivity = sensitivity
        self._duty = 0.0

    def free_bytes(self, ours: int) -> int:
        return self.total - self.host_used - ours

    def set_duty(self, duty: float):
        self._duty = duty

    def host_activity(self) -> float | None:
        return max(0.0, 1.0 - self.sensitivity * self._duty)


class NvmlProbe:
    """Real device memory via NVML. Our slabs are already counted as used.
    Untested in this repo's CI (no GPU); exercised by bench/spike.py on hardware."""

    def __init__(self, index: int, our_pid: int | None = None):
        import os

        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(self.h).total
        self.pid = our_pid or os.getpid()
        self._last_ts = 0

    def free_bytes(self, ours: int) -> int:
        return self.nvml.nvmlDeviceGetMemoryInfo(self.h).free

    def set_duty(self, duty: float):
        pass

    def host_activity(self) -> float | None:
        """Mean SM utilization of other processes since the last call (per-process
        kernel-active time). Returns None when the driver does not expose it."""
        try:
            samples = self.nvml.nvmlDeviceGetProcessUtilization(self.h, self._last_ts)
        except self.nvml.NVMLError:
            return None
        others = [s for s in samples if s.pid != self.pid]
        if samples:
            self._last_ts = max(s.timeStamp for s in samples)
        return sum(s.smUtil for s in others) / max(1, len(others)) if others else 0.0


class Watchdog:
    def __init__(self, probe, slabs: SlabManager, guard_bytes: int, slack_bytes: int,
                 poll_s: float = 0.02, on_evict=None):
        self.probe, self.slabs = probe, slabs
        self.low = guard_bytes
        self.high = guard_bytes + slack_bytes
        self.poll_s = poll_s
        self.on_evict = on_evict
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.evicted_total: list[int] = []

    def budget(self) -> int:
        """M: capacity minus the host's live usage minus the guard (our slabs count as ours)."""
        ours = self.slabs.used_bytes()
        return max(0, self.probe.free_bytes(ours) + ours - self.high)

    def check(self) -> list[int]:
        free = lambda: self.probe.free_bytes(self.slabs.used_bytes())  # noqa: E731
        if free() >= self.low:
            return []
        out = self.slabs.evict_until(self.high, free)
        if out:
            self.evicted_total.extend(out)
            log.warning("watchdog evicted %s (free=%d low=%d)", out, free(), self.low)
            if self.on_evict:
                self.on_evict(out)
        return out

    def start(self):
        def _run():
            while not self._stop.wait(self.poll_s):
                try:
                    self.check()
                except Exception:
                    log.exception("watchdog check failed")

        self.thread = threading.Thread(target=_run, name="watchdog", daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
