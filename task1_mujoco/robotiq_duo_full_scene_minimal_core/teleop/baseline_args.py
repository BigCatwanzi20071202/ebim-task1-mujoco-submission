"""Baseline sequence argument parsing utilities.

This module is deliberately MuJoCo-free so its parsing helpers can be
unit-tested even when the simulator bindings are unavailable.
"""

from __future__ import annotations

from . import config


def parse_baseline_props(args) -> list[str]:
    """Return the ordered list of target names for a baseline sequence."""
    if getattr(args, "baseline_props", None):
        return [prop.strip() for prop in args.baseline_props.split(",") if prop.strip()]
    repeats = int(getattr(args, "baseline_repeats", 1) or 1)
    base_name = getattr(args, "baseline_prop", config.C_BASELINE_CLIP)
    if repeats <= 1:
        return [base_name]
    return [base_name] * repeats
