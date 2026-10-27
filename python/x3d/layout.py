"""Choosing the cheapest stored layout for a read.

Cost = (requests / concurrency) * latency + bytes / bandwidth, with the request and byte counts
planned by the Rust core from the chunk geometry. The constants are rough: only the ordering matters,
and the layouts differ by orders of magnitude for the access patterns they are built for.
"""

from __future__ import annotations

LATENCY_S = 0.05
CONCURRENCY = 16
BANDWIDTH_BPS = 100e6


def cost(requests: int, nbytes: int) -> float:
    return requests / CONCURRENCY * LATENCY_S + nbytes / BANDWIDTH_BPS


def choose(plans: dict[str, tuple[int, int]]) -> str:
    """Name of the layout with the lowest cost; ``plans`` maps layout -> (requests, bytes)."""
    return min(plans, key=lambda name: (cost(*plans[name]), name))
