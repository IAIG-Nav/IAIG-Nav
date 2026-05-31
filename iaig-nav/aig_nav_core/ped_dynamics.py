"""Training-parity pedestrian dynamics.

Single source of truth for the reactive-SFM math that powers both the
training env (`pygame_env_aig_nav.PygameNavEnvAIGNav._update_pedestrians`)
and the Gazebo deployment mover (`iaig_nav_ros.hunav.agent_manager` when
running with `mode: training_parity`).

The model is the simplified reactive SFM described in Section III of the
IAIG-Nav paper:

  v_goal  = s · (g - p) / |g - p|                         # attraction to goal
  v_react = reactivity · (1 - d_robot / r_avoid)          # repulsion from
          · (p - p_robot) / d_robot    (if d_robot < r_avoid)   robot
  v       = v_goal + v_react, capped at 1.5 · s
  p'      = p + v · dt, reflected off world bounds with 0.8 damping

No ped-ped terms, no wall terms beyond the bounding box — matches what
the policy saw during training. Full Helbing-Molnár is in
`iaig_nav_ros/hunav/sfm.py` (used when `mode: full_sfm`).

The state is a mutable `PedestrianState` object, modified in place, so
this file can be called from both the pygame env (which holds
`DynamicAgent` instances) and the Gazebo manager (holds `Agent` msgs).
Both callers first copy their pedestrian into `PedestrianState`, tick,
and copy back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PedestrianState:
    """Minimal state a single pedestrian step needs.

    `speed` is the per-agent desired speed in m/s; `radius` is the
    agent's collision radius (for the reflecting-boundary clamp).
    """
    x: float
    y: float
    vx: float
    vy: float
    goal_x: float
    goal_y: float
    speed: float
    radius: float


def update_pedestrian_state(
    ped: PedestrianState,
    robot_x: float,
    robot_y: float,
    dt: float,
    reactivity: float,
    world_min: float,
    world_max: float,
    *,
    avoidance_radius: float = 2.0,
    speed_cap_mul: float = 1.5,
    goal_reach_tol: float = 0.5,
    boundary_damp: float = 0.8,
    new_goal_fn: Optional[Callable[[], tuple[float, float]]] = None,
) -> None:
    """Advance `ped` by one `dt` step in place. See module docstring."""
    # Goal-attraction base velocity.
    dx_goal = ped.goal_x - ped.x
    dy_goal = ped.goal_y - ped.y
    dist_goal = math.hypot(dx_goal, dy_goal) + 1e-6

    if dist_goal < goal_reach_tol:
        if new_goal_fn is not None:
            ped.goal_x, ped.goal_y = new_goal_fn()
        else:
            # Default: uniform random point inside world bounds.
            import random
            margin = 0.5
            ped.goal_x = random.uniform(world_min + margin, world_max - margin)
            ped.goal_y = random.uniform(world_min + margin, world_max - margin)
        dx_goal = ped.goal_x - ped.x
        dy_goal = ped.goal_y - ped.y
        dist_goal = math.hypot(dx_goal, dy_goal) + 1e-6

    vx_base = ped.speed * dx_goal / dist_goal
    vy_base = ped.speed * dy_goal / dist_goal

    # Reactive avoidance of the robot.
    dx_robot = ped.x - robot_x
    dy_robot = ped.y - robot_y
    dist_robot = math.hypot(dx_robot, dy_robot) + 1e-6

    if dist_robot < avoidance_radius:
        strength = reactivity * (1.0 - dist_robot / avoidance_radius)
        vx_avoid = strength * dx_robot / dist_robot
        vy_avoid = strength * dy_robot / dist_robot
    else:
        vx_avoid = 0.0
        vy_avoid = 0.0

    ped.vx = vx_base + vx_avoid
    ped.vy = vy_base + vy_avoid

    # Per-agent speed cap.
    speed = math.hypot(ped.vx, ped.vy)
    cap = ped.speed * speed_cap_mul
    if speed > cap:
        ped.vx = ped.vx / speed * cap
        ped.vy = ped.vy / speed * cap

    new_x = ped.x + ped.vx * dt
    new_y = ped.y + ped.vy * dt

    # Reflecting boundary with `boundary_damp` energy loss.
    lo = world_min + ped.radius
    hi = world_max - ped.radius
    if new_x < lo or new_x > hi:
        ped.vx *= -boundary_damp
        new_x = max(lo, min(new_x, hi))
    if new_y < lo or new_y > hi:
        ped.vy *= -boundary_damp
        new_y = max(lo, min(new_y, hi))

    ped.x = new_x
    ped.y = new_y
