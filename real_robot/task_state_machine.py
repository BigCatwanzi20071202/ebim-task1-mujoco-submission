"""Observation-gated semantic task sequence; it never calls robot drivers."""

from dataclasses import dataclass
from enum import Enum


class TaskState(str, Enum):
    LOCATE = "LOCATE"
    PLAN = "PLAN"
    PREWRAP = "PREWRAP"
    WRAP = "WRAP"
    SETTLE = "SETTLE"
    VERIFY = "VERIFY"
    MOUTH = "MOUTH"
    SEAT = "SEAT"
    RETAIN = "RETAIN"
    TRANSPORT = "TRANSPORT"
    INSPECTION = "INSPECTION"
    HOLD = "HOLD"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class TaskStateMachine:
    stages: tuple[str, ...]
    state: TaskState = TaskState.LOCATE
    reason: str = ""

    def dry_run_states(self) -> list[str]:
        paths = {"o": ["LOCATE", "PLAN", "PREWRAP", "WRAP", "SETTLE", "VERIFY"],
                 "c": ["LOCATE", "PLAN", "MOUTH", "SEAT", "RETAIN", "VERIFY"],
                 "y": ["LOCATE", "PLAN", "TRANSPORT", "INSPECTION", "HOLD", "VERIFY"]}
        return [f"{stage.upper()}:{state}" for stage in self.stages for state in paths[stage]] + ["TASK:SUCCEEDED"]
