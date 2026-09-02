"""Safe command-line entry point for the non-moving FR3 migration scaffold."""

import argparse

from .config_real import ARMS, CAMERA_TOPICS, MOTION_ENABLED
from .fr3_interface import FR3Interface
from .perception import check_perception_runtime
from .task_state_machine import TaskStateMachine


def _stages(value: str) -> tuple[str, ...]:
    return {"o": ("o",), "c": ("c",), "oc": ("o", "c"), "ocy": ("o", "c", "y")}[value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="plan/log only; send no commands")
    parser.add_argument("--check-robot", action="store_true")
    parser.add_argument("--check-perception", action="store_true")
    parser.add_argument("--stage", choices=("o", "c", "oc", "ocy"), default="ocy")
    args = parser.parse_args(argv)
    print(f"motion_enabled={MOTION_ENABLED}")
    print("movement_implementation=DISABLED (PTPMotion schema unconfirmed)")
    if args.check_robot:
        result = FR3Interface(False).check()
        print(f"robot_check={'AVAILABLE' if result.ros2_available else 'UNAVAILABLE'}: {result.detail}")
    if args.check_perception:
        ok, detail = check_perception_runtime()
        print(f"perception_check={'AVAILABLE' if ok else 'UNAVAILABLE'}: {detail}")
        print(f"camera_topics={CAMERA_TOPICS}")
    print(f"arms={{{', '.join(f'{name}:{cfg.robot_ip}' for name, cfg in ARMS.items())}}}")
    machine = TaskStateMachine(_stages(args.stage))
    mode = "DRY_RUN" if args.dry_run else "NO_MOTION_PREVIEW"
    print(f"mode={mode} stage={args.stage}")
    for state in machine.dry_run_states():
        print(state)
    print("commands_sent=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
