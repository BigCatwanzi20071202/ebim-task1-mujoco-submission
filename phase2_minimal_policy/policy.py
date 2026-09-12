#!/usr/bin/env python3
"""ROS 2 state monitor that deliberately exposes no robot command path."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from policy_core import HoldMonitor, validate_joint_state


TOPICS = {
    "left": "/left/franka_robot_state_broadcaster/measured_joint_states",
    "right": "/right/franka_robot_state_broadcaster/measured_joint_states",
}


def emit(status: str, **fields: object) -> None:
    print(json.dumps({"status": status, "commands_sent": 0, **fields}, sort_keys=True), flush=True)


def self_test() -> int:
    names = [f"joint{i}" for i in range(1, 8)]
    validate_joint_state(names, [0.0] * 7)
    monitor = HoldMonitor()
    monitor.observe("left", names, [0.0] * 7)
    monitor.observe("right", names, [0.0] * 7)
    if not monitor.active() or monitor.commands_sent() != 0:
        return 1
    emit("SELF_TEST_OK", publishers_created=0, action_clients_created=0, service_clients_created=0)
    return 0


def run(startup_timeout: float, stale_after: float) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except Exception as exc:
        emit("ROS_IMPORT_ERROR", error=str(exc))
        return 2

    rclpy.init(args=None)
    monitor = HoldMonitor(stale_after_seconds=stale_after)
    node = Node("ggboy_task1_safe_hold_monitor")

    def callback(arm: str):
        def receive(message: JointState) -> None:
            try:
                monitor.observe(arm, list(message.name), list(message.position))
            except ValueError as exc:
                emit("INVALID_JOINT_STATE", arm=arm, error=str(exc))
        return receive

    subscriptions = [
        node.create_subscription(JointState, topic, callback(arm), 10)
        for arm, topic in TOPICS.items()
    ]
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    emit("WAITING_FOR_JOINT_STATES", topics=TOPICS, startup_timeout_seconds=startup_timeout)
    started = time.monotonic()
    was_active = False
    exit_code = 0
    try:
        while not stop:
            rclpy.spin_once(node, timeout_sec=0.1)
            active = monitor.active()
            if active and not was_active:
                emit("ACTIVE_SAFE_HOLD_NO_COMMAND", samples=monitor.samples)
                was_active = True
            elif was_active and not active:
                emit("STALE_JOINT_STATE_FAIL_CLOSED", samples=monitor.samples)
                exit_code = 4
                break
            elif not was_active and time.monotonic() - started > startup_timeout:
                emit("STARTUP_TIMEOUT_FAIL_CLOSED", samples=monitor.samples)
                exit_code = 3
                break
    finally:
        subscriptions.clear()
        node.destroy_node()
        rclpy.shutdown()
    emit("STOPPED", samples=monitor.samples)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("self-test")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--startup-timeout", type=float, default=20.0)
    run_parser.add_argument("--stale-after", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.mode == "self-test":
        return self_test()
    if args.startup_timeout <= 0 or args.stale_after <= 0:
        parser.error("timeouts must be positive")
    return run(args.startup_timeout, args.stale_after)


if __name__ == "__main__":
    sys.exit(main())
