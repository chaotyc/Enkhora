"""Compute protection: token bucket, kernel-length cap, AIMD on κ (section 8).

The host is the tenant and we are the guest. κ is spent in kernel-milliseconds;
an empty bucket means the executor refuses and the dispatcher goes elsewhere.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque


class TokenBucket:
    """Refills at κ ms of kernel time per wall ms, capped at κ * window_ms.

    The window bounds bursts: at most κ * window_ms of back-to-back kernel time.

    Calls are admitted while tokens > 0 and charged their measured time afterwards
    (small debt allowed: the kernel-length cap bounds it).
    """

    def __init__(self, kappa: float, window_ms: float = 100.0, clock=time.monotonic):
        self.clock = clock
        self.window_ms = window_ms
        self._lock = threading.Lock()
        self.kappa = max(0.0, min(1.0, kappa))
        self.tokens = self.capacity
        self._last = clock()
        self.paused = False

    @property
    def capacity(self) -> float:
        return self.kappa * self.window_ms

    def _refill(self):
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * 1000.0 * self.kappa)
        self._last = now

    def set_kappa(self, kappa: float):
        with self._lock:
            self._refill()
            self.kappa = max(0.0, min(1.0, kappa))
            self.tokens = min(self.tokens, self.capacity)

    def try_acquire(self) -> bool:
        with self._lock:
            if self.paused:
                return False
            self._refill()
            return self.tokens > 0

    def charge(self, ms: float):
        with self._lock:
            self._refill()
            self.tokens -= ms

    def remaining_ms(self) -> float:
        with self._lock:
            self._refill()
            return 0.0 if self.paused else self.tokens


class AIMD:
    """Additive increase while the host shows no slowdown; multiplicative cut when it does.
    Same shape as TCP congestion control, with host slowdown playing packet loss."""

    def __init__(self, kappa0: float = 0.01, add: float = 0.01, mult: float = 0.5,
                 kmin: float = 0.0, kmax: float = 1.0, tolerance: float = 0.05):
        self.kappa = kappa0
        self.add, self.mult, self.kmin, self.kmax, self.tolerance = add, mult, kmin, kmax, tolerance

    def update(self, slowdown: float) -> float:
        if slowdown > self.tolerance:
            self.kappa = max(self.kmin, self.kappa * self.mult)
        else:
            self.kappa = min(self.kmax, self.kappa + self.add)
        return self.kappa


class OnOffEstimator:
    """Slowdown without a baseline: alternate randomized windows with lending allowed
    and paused, and compare the host's own activity rate across them."""

    def __init__(self, pairs: int = 4, rng: random.Random | None = None,
                 min_window_s: float = 0.5, max_window_s: float = 2.0):
        self.on: deque[float] = deque(maxlen=pairs)
        self.off: deque[float] = deque(maxlen=pairs)
        self.rng = rng or random.Random()
        self.min_w, self.max_w = min_window_s, max_window_s

    def next_window(self) -> tuple[bool, float]:
        """Return (lending_on, seconds). Order within each pair is randomized."""
        lend = self.rng.random() < 0.5 if len(self.on) == len(self.off) else len(self.on) < len(self.off)
        return lend, self.rng.uniform(self.min_w, self.max_w)

    def record(self, lending_on: bool, host_rate: float):
        (self.on if lending_on else self.off).append(host_rate)

    def ready(self) -> bool:
        return len(self.on) > 0 and len(self.off) > 0

    def slowdown(self) -> float:
        on = sum(self.on) / len(self.on)
        off = sum(self.off) / len(self.off)
        return 0.0 if off <= 0 else max(0.0, 1.0 - on / off)
