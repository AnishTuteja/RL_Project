"""Local multi-seed, multi-scenario evaluation loop shared across notebooks.

Implements the TODO left in the official starter_notebook.ipynb ("Build the
local evaluation loop"): run a policy across disclosed validation seeds and
declared scenario families, and aggregate the official unscaled episode cost
and its components (Section 11 of the project spec).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from industrial_inventory_env import IndustrialInventoryEnv


def evaluate_policy(
    policy: Callable[[dict], Iterable[int]],
    student_config: dict[str, Any],
    seeds: Iterable[int],
    scenario_modes: Iterable[str] = ("random",),
    domain_randomization: bool = True,
) -> pd.DataFrame:
    """Run ``policy`` over every (scenario_mode, seed) pair and collect cost breakdowns.

    ``policy`` must return actual order quantities (e.g. the return value of a
    run_policy(observation) function), not internal action indices.
    """
    rows: list[dict[str, Any]] = []
    for scenario_mode in scenario_modes:
        env = IndustrialInventoryEnv(
            student_config,
            scenario_mode=scenario_mode,
            domain_randomization=domain_randomization,
        )
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            cost_totals = {"holding": 0.0, "stockout": 0.0, "ordering": 0.0, "discarding": 0.0}
            unfulfilled_total = 0
            started = time.perf_counter()
            while True:
                quantities = policy(observation)
                action = env.quantities_to_action_indices(quantities)
                observation, _, terminated, truncated, step_info = env.step(action)
                for key in cost_totals:
                    cost_totals[key] += step_info["costs"][key]
                unfulfilled_total += int(np.sum(step_info["unfulfilled_demand"]))
                if truncated or terminated:
                    break
            rows.append(
                {
                    "scenario_mode": scenario_mode,
                    "seed": seed,
                    "total_cost": sum(cost_totals.values()),
                    **cost_totals,
                    "unfulfilled_units": unfulfilled_total,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.Series:
    """Aggregate evaluate_policy() output into headline comparison metrics."""
    return pd.Series(
        {
            "mean_total_cost": results["total_cost"].mean(),
            "std_total_cost": results["total_cost"].std(),
            "mean_unfulfilled_units": results["unfulfilled_units"].mean(),
            "mean_wall_seconds": results["wall_seconds"].mean(),
        }
    )