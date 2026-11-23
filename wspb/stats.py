"""Uncertainty for benchmark numbers.

A score is the mean over tasks of each task's mean over runs. Its interval comes from a two-stage bootstrap:
resample tasks, then resample runs within each task, so both task choice and run-to-run noise count. Two
configurations are compared on the tasks they share, resampling the same tasks for both (a paired
difference), which is much tighter than two separate intervals. Seeds are fixed so reports reproduce.
"""

from __future__ import annotations

import random
from typing import Any

B = 2000
LEVEL = 0.90


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _quantiles(xs: list[float], level: float) -> tuple[float, float]:
    xs = sorted(xs)
    lo = xs[int((1 - level) / 2 * (len(xs) - 1))]
    hi = xs[int((1 + level) / 2 * (len(xs) - 1))]
    return lo, hi


def _resampled_task_mean(rng: random.Random, runs: list[float]) -> float:
    return _mean([runs[rng.randrange(len(runs))] for _ in runs])


def interval(by_task: dict[str, list[float]], *, b: int = B, level: float = LEVEL, seed: int = 0) -> dict[str, Any]:
    """by_task: task -> per-run values (None values dropped). Mean over tasks with a two-stage bootstrap interval."""
    data = {t: [float(x) for x in xs if x is not None] for t, xs in by_task.items()}
    data = {t: xs for t, xs in data.items() if xs}
    if not data:
        return {"mean": None, "lo": None, "hi": None, "tasks": 0}
    tasks = sorted(data)
    rng = random.Random(seed)
    boots = []
    for _ in range(b):
        pick = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        boots.append(_mean([_resampled_task_mean(rng, data[t]) for t in pick]))
    lo, hi = _quantiles(boots, level)
    return {"mean": _mean([_mean(data[t]) for t in tasks]), "lo": lo, "hi": hi, "tasks": len(tasks)}


def paired(a: dict[str, list[float]], b_: dict[str, list[float]], *, b: int = B, level: float = LEVEL,
           seed: int = 0) -> dict[str, Any]:
    """Difference a - b over the tasks both have, resampling the same tasks for both sides."""
    a = {t: [float(x) for x in xs if x is not None] for t, xs in a.items()}
    b_ = {t: [float(x) for x in xs if x is not None] for t, xs in b_.items()}
    tasks = sorted(t for t in a if t in b_ and a[t] and b_[t])
    if not tasks:
        return {"diff": None, "lo": None, "hi": None, "tasks": 0}
    per = {t: _mean(a[t]) - _mean(b_[t]) for t in tasks}
    rng = random.Random(seed)
    boots = []
    for _ in range(b):
        pick = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        boots.append(_mean([_resampled_task_mean(rng, a[t]) - _resampled_task_mean(rng, b_[t]) for t in pick]))
    lo, hi = _quantiles(boots, level)
    return {"diff": _mean(list(per.values())), "lo": lo, "hi": hi, "tasks": len(tasks),
            "a_better": sum(v > 0 for v in per.values()), "b_better": sum(v < 0 for v in per.values()),
            "per_task": per}


def contrast(a1: dict[str, list[float]], b1: dict[str, list[float]], a2: dict[str, list[float]],
             b2: dict[str, list[float]], *, b: int = B, level: float = LEVEL, seed: int = 0) -> dict[str, Any]:
    """(a1 - b1) minus (a2 - b2): how much larger a's lead over b is on one group of tasks than on another. Each
    group's tasks are resampled on their own, the same tasks for both sides; `below` is the share of resamples in
    which the lead was no larger."""
    groups = []
    for x, y in ((a1, b1), (a2, b2)):
        x = {t: [float(v) for v in vs if v is not None] for t, vs in x.items()}
        y = {t: [float(v) for v in vs if v is not None] for t, vs in y.items()}
        groups.append((x, y, sorted(t for t in x if t in y and x[t] and y[t])))
    if not all(tasks for _, _, tasks in groups):
        return {"diff": None, "lo": None, "hi": None, "below": None, "tasks": (0, 0)}
    rng = random.Random(seed)

    def lead(x, y, tasks, resample):
        pick = [tasks[rng.randrange(len(tasks))] for _ in tasks] if resample else tasks
        one = _resampled_task_mean if resample else lambda _, runs: _mean(runs)
        return _mean([one(rng, x[t]) - one(rng, y[t]) for t in pick])
    boots = [lead(*groups[0], True) - lead(*groups[1], True) for _ in range(b)]
    lo, hi = _quantiles(boots, level)
    return {"diff": lead(*groups[0], False) - lead(*groups[1], False), "lo": lo, "hi": hi,
            "below": sum(v <= 0 for v in boots) / b, "tasks": tuple(len(tasks) for _, _, tasks in groups)}
