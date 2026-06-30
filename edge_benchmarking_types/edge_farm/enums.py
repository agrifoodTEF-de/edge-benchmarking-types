from enum import Enum


class OptimizationFactor(str, Enum):
    """The metric a device recommendation is optimized (minimized) for."""

    COST = "cost"
    ENERGY = "energy"
    LATENCY = "latency"


class LatencyPercentile(str, Enum):
    """Which inference-latency statistic the latency constraint is applied to."""

    AVG = "avg"
    P95 = "p95"
    P99 = "p99"
