"""Compute budget guard.

Training stops before the configured GPU-hour budget is exhausted. Usage is
measured as wall-clock hours multiplied by the number of GPUs, which is how
the shared allocation is accounted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ComputeBudget:
    gpu_hours: float
    num_gpus: int = 1
    safety_margin: float = 0.05
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _start: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.gpu_hours <= 0:
            raise ValueError("gpu_hours must be positive")
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError("safety_margin must be in [0, 1)")

    def start(self) -> None:
        self._start = self.clock()

    @property
    def used_gpu_hours(self) -> float:
        if self._start is None:
            return 0.0
        return (self.clock() - self._start) / 3600.0 * self.num_gpus

    @property
    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.gpu_hours - self.used_gpu_hours)

    def can_continue(self, next_step_estimate_hours: float = 0.0) -> bool:
        """True while usage plus the next step stays within the budget minus the margin."""
        limit = self.gpu_hours * (1.0 - self.safety_margin)
        return self.used_gpu_hours + next_step_estimate_hours * self.num_gpus <= limit
