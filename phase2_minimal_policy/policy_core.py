"""Pure validation and health state for the no-command Phase II baseline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable


ARMS = ("left", "right")


def validate_joint_state(names: list[str], positions: list[float]) -> tuple[float, ...]:
    if len(names) != 7 or len(positions) != 7:
        raise ValueError("exactly seven joint names and positions are required")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("joint names must be nonempty strings")
    if len(set(names)) != 7:
        raise ValueError("joint names must be unique")
    values = tuple(float(value) for value in positions)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("joint positions must be finite")
    return values


@dataclass
class HoldMonitor:
    stale_after_seconds: float = 1.0
    clock: Callable[[], float] = time.monotonic
    received_at: dict[str, float] = field(default_factory=dict)
    samples: dict[str, int] = field(default_factory=lambda: {arm: 0 for arm in ARMS})

    def observe(self, arm: str, names: list[str], positions: list[float]) -> None:
        if arm not in ARMS:
            raise ValueError("unknown arm")
        validate_joint_state(names, positions)
        self.received_at[arm] = self.clock()
        self.samples[arm] += 1

    def active(self) -> bool:
        now = self.clock()
        return all(
            arm in self.received_at and now - self.received_at[arm] <= self.stale_after_seconds
            for arm in ARMS
        )

    @staticmethod
    def commands_sent() -> int:
        return 0
