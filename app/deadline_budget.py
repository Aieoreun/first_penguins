"""Monotonic per-request deadline budget manager preventing 30s competition timeouts."""

from __future__ import annotations

import time
from typing import Optional


class DeadlineBudget:
    """Tracks remaining wall-clock budget for a single request across multi-agent nodes."""

    def __init__(self, total_budget_sec: float = 27.5):
        self.t_start = time.monotonic()
        self.total_budget = total_budget_sec

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t_start

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.elapsed)

    def has_at_least(self, min_required_sec: float) -> bool:
        return self.remaining >= min_required_sec

    def check_or_degrade(self, min_required_sec: float, fallback_route: str) -> Optional[str]:
        """Return fallback_route if remaining budget is below min_required_sec."""
        if self.remaining < min_required_sec:
            return fallback_route
        return None

    def step_timeout(self, default_sec: float, reserve_sec: float = 1.0) -> float:
        """Compute a safe timeout for an external call leaving reserve_sec for post-processing."""
        return max(0.5, min(default_sec, self.remaining - reserve_sec))
