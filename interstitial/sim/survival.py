"""Lifetime prediction (section 9): empirical conditional survival per bucket.

S(t | elapsed a) = P(L > a + t) / P(L > a), hard-capped at limit - a. Buckets are
(kind, partition, time-limit band, account-history band). No learning needed; a
learned survival model replaces this only if shadow results justify it.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass


def limit_band(limit_s: float) -> str:
    h = limit_s / 3600
    return "<=4h" if h <= 4 else "<=12h" if h <= 12 else "<=24h" if h <= 24 else ">24h"


def ratio_band(r: float | None) -> str:
    if r is None:
        return "new"
    return "lo" if r < 0.4 else "mid" if r < 0.7 else "hi"


@dataclass
class EmpiricalCurve:
    durations: list[float]       # sorted
    elapsed: float
    cap_s: float

    def _surv(self, x: float) -> int:
        return len(self.durations) - bisect.bisect_right(self.durations, x)

    def p_alive(self, t: float) -> float:
        if t >= self.cap_s:
            return 0.0
        base = self._surv(self.elapsed)
        if base == 0:
            return math.exp(-t / 1800)            # outlived every example: assume short
        return self._surv(self.elapsed + t) / base

    def expected_min(self, horizon: float, steps: int = 32) -> float:
        h = min(horizon, self.cap_s)
        if h <= 0:
            return 0.0
        dt = h / steps
        return sum(self.p_alive((i + 0.5) * dt) for i in range(steps)) * dt


@dataclass
class OracleCurve:
    remaining: float

    def p_alive(self, t: float) -> float:
        return 1.0 if t < self.remaining else 0.0

    def expected_min(self, horizon: float) -> float:
        return max(0.0, min(horizon, self.remaining))


class SurvivalModel:
    def __init__(self, min_bucket: int = 30):
        self.min_bucket = min_bucket
        self.buckets: dict[tuple, list[float]] = {}
        self.account_ratio: dict[str, float] = {}

    def fit(self, jobs: list[dict], idle_gaps: list[float]):
        """jobs: dicts with partition, limit_s, duration_s, account."""
        ratios: dict[str, list[float]] = {}
        for j in jobs:
            if j["limit_s"] > 0:
                ratios.setdefault(j["account"], []).append(j["duration_s"] / j["limit_s"])
        self.account_ratio = {a: sum(r) / len(r) for a, r in ratios.items() if len(r) >= 3}
        raw: dict[tuple, list[float]] = {}
        for j in jobs:
            rb = ratio_band(self.account_ratio.get(j["account"]))
            for key in self._keys("job", j["partition"], j["limit_s"], rb):
                raw.setdefault(key, []).append(j["duration_s"])
        for key in self._keys("idle", "", 0, "new"):
            raw.setdefault(key, []).extend(idle_gaps)
        self.buckets = {k: sorted(v) for k, v in raw.items()}
        return self

    @staticmethod
    def _keys(kind, partition, limit_s, rband):
        lb = limit_band(limit_s) if kind == "job" else ""
        return [(kind, partition, lb, rband), (kind, partition, lb), (kind,)]

    def curve(self, kind: str, partition: str = "", limit_s: float = 0.0, elapsed_s: float = 0.0,
              account: str = "") -> EmpiricalCurve:
        rb = ratio_band(self.account_ratio.get(account))
        cap = max(0.0, limit_s - elapsed_s) if kind == "job" and limit_s > 0 else math.inf
        for key in self._keys(kind, partition, limit_s, rb):
            d = self.buckets.get(key)
            if d and len(d) >= self.min_bucket:
                return EmpiricalCurve(d, elapsed_s, cap)
        return EmpiricalCurve(self.buckets.get((kind,), [3600.0]), elapsed_s, cap)
