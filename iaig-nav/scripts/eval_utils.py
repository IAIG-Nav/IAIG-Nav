"""Utilities for deterministic evaluation scenarios and seeding."""

from typing import List, Tuple
import math
import random

import numpy as np


DEFAULT_EVAL_SCENARIO_SEED = 12345


def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def generate_eval_scenarios(env, num_scenarios: int, seed: int = DEFAULT_EVAL_SCENARIO_SEED) -> List[Tuple[float, float, float, float, float]]:
    """Generate deterministic (robot_x, robot_y, yaw, goal_x, goal_y) scenarios."""
    rng = np.random.default_rng(seed)
    scenarios: List[Tuple[float, float, float, float, float]] = []

    # Avoid influence from previous episode pedestrians.
    old_peds = getattr(env, "pedestrians", None)
    env.pedestrians = []
    try:
        robot_attempts = 100
        goal_attempts = 100
        scenario_retries = 20

        for _ in range(num_scenarios):
            scenario_found = False
            for _ in range(scenario_retries):
                x = y = yaw = 0.0
                goal_x = goal_y = 0.0

                found_pair = False
                for _ in range(robot_attempts):
                    x = float(rng.uniform(env.world_min + 1.0, env.world_max - 1.0))
                    y = float(rng.uniform(env.world_min + 1.0, env.world_max - 1.0))
                    yaw = float(rng.uniform(-math.pi, math.pi))
                    extra_margin = getattr(env, "robot_spawn_clearance", 0.0)
                    if not env._is_position_safe(x, y, env.robot_radius, extra_margin=extra_margin):
                        continue

                    goal_margin = getattr(env, "goal_margin", 0.5)
                    goal_clearance = getattr(env, "goal_spawn_clearance", 0.0)
                    min_goal_distance = getattr(env, "min_goal_distance", 2.0)
                    for _ in range(goal_attempts):
                        goal_x = float(rng.uniform(env.world_min + goal_margin, env.world_max - goal_margin))
                        goal_y = float(rng.uniform(env.world_min + goal_margin, env.world_max - goal_margin))
                        if math.hypot(goal_x - x, goal_y - y) <= min_goal_distance:
                            continue
                        if not env._is_position_safe(goal_x, goal_y, 0.3, extra_margin=goal_clearance):
                            continue

                        found_pair = True
                        break

                    if found_pair:
                        break

                if not found_pair:
                    continue

                scenarios.append((x, y, yaw, goal_x, goal_y))
                scenario_found = True
                break

            if not scenario_found:
                raise RuntimeError(
                    "Failed to sample a safe eval scenario after retries."
                )

        return scenarios
    finally:
        if old_peds is not None:
            env.pedestrians = old_peds


def filter_safe_scenarios(env, scenarios: List[Tuple[float, float, float, float, float]], seed_base: int = DEFAULT_EVAL_SCENARIO_SEED):
    """Drop scenarios that violate safety constraints in env.reset()."""
    safe: List[Tuple[float, float, float, float, float]] = []
    dropped = 0
    for idx, scenario in enumerate(scenarios):
        try:
            env.reset(scenario=scenario, eval_seed=seed_base + idx)
        except Exception:
            dropped += 1
            continue
        safe.append(scenario)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} unsafe fixed scenarios.")
    return safe
