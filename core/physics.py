"""Physics utilities for the Newtonian universe simulation."""

from __future__ import annotations

import math
import random
from typing import Sequence, Tuple


def wrap_coordinate(value: float, limit: float) -> float:
    """Wrap a coordinate value within the given limit."""
    half = limit / 2.0
    while value < -half:
        value += limit
    while value > half:
        value -= limit
    return value


def vector_length(vector: Sequence[float]) -> float:
    """Return the Euclidean length of the vector."""
    return math.sqrt(sum(component * component for component in vector))


def normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    """Return the normalized version of the vector."""
    length = vector_length(vector)
    if length == 0.0:
        return 0.0, 0.0, 0.0
    return tuple(component / length for component in vector)


def cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Return the cross product of vectors *a* and *b*."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def random_unit_vector() -> Tuple[float, float, float]:
    """Return a random unit vector."""
    z = random.uniform(-1.0, 1.0)
    theta = random.uniform(0.0, math.tau)
    radius = math.sqrt(1.0 - z * z)
    return radius * math.cos(theta), radius * math.sin(theta), z


def wrapped_delta(origin: float, target: float, limit: float) -> float:
    """Return the wrapped delta between *origin* and *target*."""
    diff = target - origin
    half = limit / 2.0
    if diff > half:
        diff -= limit
    elif diff < -half:
        diff += limit
    return diff
