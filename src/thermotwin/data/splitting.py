"""Chronological split boundaries with purge gaps protecting lag windows and targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitRange:
    """Half-open raw timestep range assigned to one supervised partition."""

    name: str
    start: int
    end: int


def chronological_ranges(n_steps: int, window_steps: int, max_horizon_steps: int) -> list[SplitRange]:
    """Create train/validation/calibration/test blocks separated by a full leakage purge gap."""
    gap = window_steps + max_horizon_steps
    usable = n_steps - 3 * gap
    minimum = 4 * (window_steps + max_horizon_steps + 2)
    if usable < minimum:
        raise ValueError(f"trajectory needs at least {minimum + 3 * gap} steps for four purged partitions")
    weights = (0.50, 0.17, 0.16, 0.17)
    lengths = [int(usable * weight) for weight in weights]
    lengths[-1] = usable - sum(lengths[:-1])
    names = ("train", "validation", "calibration", "test")
    ranges, cursor = [], 0
    for index, (name, length) in enumerate(zip(names, lengths)):
        ranges.append(SplitRange(name, cursor, cursor + length))
        cursor += length
        if index < 3:
            cursor += gap
    return ranges


def assert_partition_local(input_start: int, origin: int, target_end: int, split: SplitRange) -> None:
    """Raise if a sample's lag window or targets leave their assigned raw block."""
    if not (split.start <= input_start <= origin < target_end < split.end):
        raise ValueError("input/target sample crosses a split boundary")

