"""
Pygame Navigation Environment for AIG-Nav (Action-conditioned Interaction Graph)

This environment extends the basic navigation environment to support:
1. Trajectory history tracking for all agents
2. Reactive pedestrian behavior (pedestrians respond to robot)
3. Rich observation for trajectory prediction

Key differences from standard env:
- Tracks history of all agent positions
- Pedestrians have reactive avoidance behavior
- Returns detailed agent information for prediction
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


@dataclass
class DynamicAgent:
    """Represents a dynamic agent (pedestrian) with history tracking"""
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    agent_id: int
    # Goal position for intentional movement
    goal_x: float = 0.0
    goal_y: float = 0.0
    # Per-pedestrian speed (for domain randomization)
    speed: float = 0.5
    # History tracking
    history_x: deque = field(default_factory=lambda: deque(maxlen=20))
    history_y: deque = field(default_factory=lambda: deque(maxlen=20))
    history_vx: deque = field(default_factory=lambda: deque(maxlen=20))
    history_vy: deque = field(default_factory=lambda: deque(maxlen=20))

    def record_history(self):
        """Record current state to history"""
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        self.history_vx.append(self.vx)
        self.history_vy.append(self.vy)

    def get_history_array(self, length: int = 10) -> np.ndarray:
        """Get history as numpy array [T, 4] (x, y, vx, vy)"""
        hist_len = min(len(self.history_x), length)
        if hist_len == 0:
            return np.zeros((length, 4), dtype=np.float32)

        history = np.zeros((length, 4), dtype=np.float32)
        for i in range(hist_len):
            idx = -(hist_len - i)
            history[length - hist_len + i, 0] = self.history_x[idx]
            history[length - hist_len + i, 1] = self.history_y[idx]
            history[length - hist_len + i, 2] = self.history_vx[idx]
            history[length - hist_len + i, 3] = self.history_vy[idx]

        # Pad earlier timesteps with first known position
        if hist_len < length:
            for i in range(length - hist_len):
                history[i] = history[length - hist_len]

        return history


@dataclass
class StaticObstacle:
    """Static obstacle"""
    x: float
    y: float
    radius: float


class PygameNavEnvAIGNav:
    """
    Navigation environment for Action-conditioned Interaction Graph training.

    Key features:
    1. Tracks trajectory history for all agents
    2. Pedestrians have reactive behavior (avoid robot)
    3. Returns rich observations for trajectory prediction
    """

    def __init__(
        self,
        render: bool = False,
        seed: Optional[int] = None,
        world_min: float = -4.0,
        world_max: float = 4.0,
        dt: float = 0.1,
        scan_beams: int = 32,
        scan_min: float = 0.12,
        scan_max: float = 3.5,
        render_fps: int = 60,
        history_length: int = 10,
        prediction_horizon: int = 12,
        num_pedestrians: int = 4,
        pedestrian_reactivity: float = 0.0,  # How much pedestrians react to robot
        robot_radius: float = 0.25,
        scan_fov: float = math.pi / 2,       # Half-FOV in radians; π/2 = front 180°
        scan_noise_std: float = 0.0,          # LiDAR Gaussian noise std (meters)
        # Domain randomization parameters
        randomize_obstacles: bool = False,     # Randomize static obstacles each reset
        num_obstacles_range: tuple = (2, 6),   # (min, max) number of obstacles
        obstacle_radius_range: tuple = (0.3, 0.8),  # (min, max) obstacle radius
        randomize_num_pedestrians: bool = False,  # Randomize pedestrian count each reset
        num_pedestrians_range: tuple = (2, 6),    # (min, max) pedestrian count
        ped_speed_range: tuple = (0.5, 1.0),      # (min, max) per-pedestrian speed
    ):
        self.render_enabled = render
        self.world_min = world_min
        self.world_max = world_max
        self.dt = dt
        self.scan_beams = scan_beams
        self.scan_min = scan_min
        self.scan_max = scan_max
        self.render_fps = render_fps
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.num_pedestrians = num_pedestrians
        self.pedestrian_reactivity = pedestrian_reactivity

        self.world_radius = 5.0
        self.max_velocity = 1.5
        self.max_goal_dist = 6.0

        self.robot_radius = robot_radius
        self.scan_fov = scan_fov
        self.scan_noise_std = scan_noise_std
        self.randomize_obstacles = randomize_obstacles
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_radius_range = obstacle_radius_range
        self.randomize_num_pedestrians = randomize_num_pedestrians
        self.num_pedestrians_range = num_pedestrians_range
        self.ped_speed_range = ped_speed_range
        self.goal_threshold = 0.35
        self.pedestrian_radius = 0.3
        self.pedestrian_speed = 0.5
        # Extra clearance around the robot at spawn to avoid immediate collisions.
        self.robot_spawn_clearance = 0.5
        # Goal placement constraints.
        self.goal_spawn_clearance = 0.5
        self.goal_margin = 0.5
        self.min_goal_distance = 2.0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Static obstacles
        self.static_obstacles: List[StaticObstacle] = [
            StaticObstacle(2.5, -2.5, 0.6),
            StaticObstacle(-2.5, 2.5, 0.5),
            StaticObstacle(-2.5, -1.0, 0.4),
        ]
        self._validate_static_obstacles()

        # Dynamic agents (pedestrians)
        self.pedestrians: List[DynamicAgent] = []

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0

        # Robot history
        self.robot_history_x = deque(maxlen=history_length)
        self.robot_history_y = deque(maxlen=history_length)
        self.robot_history_vx = deque(maxlen=history_length)
        self.robot_history_vy = deque(maxlen=history_length)

        # Goal
        self.goal_x = 2.0
        self.goal_y = 0.0

        self.last_distance = None
        self.steps = 0

        # Scan
        self.beam_angles = np.linspace(-self.scan_fov, self.scan_fov, self.scan_beams, endpoint=True)
        self.last_scan_raw = None

        # Rendering
        self.screen = None
        self.clock = None
        self.scale = 100.0
        self.screen_size = int((self.world_max - self.world_min) * self.scale)

        if self.render_enabled:
            if pygame is None:
                raise RuntimeError("pygame is not installed")
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("AIG-Nav Navigation Environment")
            self.clock = pygame.time.Clock()

    def close(self):
        if self.render_enabled and pygame is not None:
            pygame.quit()

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = int((x - self.world_min) * self.scale)
        sy = int((self.world_max - y) * self.scale)
        return sx, sy

    def _wrap_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _is_position_safe(
        self,
        x: float,
        y: float,
        radius: float,
        exclude_pedestrian_id: int = -1,
        extra_margin: float = 0.0,
    ) -> bool:
        """Check if position is safe (not colliding with obstacles or other pedestrians)"""
        margin = 0.3 + max(0.0, extra_margin)
        if x < self.world_min + radius + margin or x > self.world_max - radius - margin:
            return False
        if y < self.world_min + radius + margin or y > self.world_max - radius - margin:
            return False

        for obs in self.static_obstacles:
            if math.hypot(x - obs.x, y - obs.y) < (radius + obs.radius + margin):
                return False

        for ped in self.pedestrians:
            if ped.agent_id == exclude_pedestrian_id:
                continue
            if math.hypot(x - ped.x, y - ped.y) < (radius + ped.radius + margin):
                return False

        return True

    def _validate_static_obstacles(self):
        for i, obs_a in enumerate(self.static_obstacles):
            for j in range(i + 1, len(self.static_obstacles)):
                obs_b = self.static_obstacles[j]
                dist = math.hypot(obs_a.x - obs_b.x, obs_a.y - obs_b.y)
                if dist < (obs_a.radius + obs_b.radius):
                    raise ValueError(
                        f"Static obstacles overlap: {i} and {j} "
                        f"(dist={dist:.2f}, radii={obs_a.radius + obs_b.radius:.2f})"
                    )

    def _generate_random_obstacles(self, rng: Optional[np.random.Generator] = None):
        """Generate random static obstacles for domain randomization."""
        if rng is not None:
            rand = rng.uniform
            rand_int = rng.integers
        else:
            rand = np.random.uniform
            rand_int = np.random.randint
        n = int(rand_int(self.num_obstacles_range[0], self.num_obstacles_range[1] + 1))
        self.static_obstacles = []
        margin = 0.5  # Keep obstacles away from walls
        for _ in range(n):
            for _attempt in range(100):
                r = float(rand(self.obstacle_radius_range[0], self.obstacle_radius_range[1]))
                x = float(rand(self.world_min + r + margin, self.world_max - r - margin))
                y = float(rand(self.world_min + r + margin, self.world_max - r - margin))
                # Check no overlap with existing obstacles
                ok = all(
                    math.hypot(x - o.x, y - o.y) > r + o.radius + 0.3
                    for o in self.static_obstacles
                )
                if ok:
                    self.static_obstacles.append(StaticObstacle(x, y, r))
                    break

    def _create_pedestrians(self, rng: Optional[np.random.Generator] = None):
        """Create pedestrians with random positions and goals"""
        self.pedestrians = []
        rand = rng.uniform if rng is not None else np.random.uniform
        max_attempts_per_ped = 300

        for i in range(self.num_pedestrians):
            # Per-pedestrian random speed
            ped_speed = float(rand(self.ped_speed_range[0], self.ped_speed_range[1]))

            placed = False
            for _ in range(max_attempts_per_ped):
                x = float(rand(self.world_min + 1.0, self.world_max - 1.0))
                y = float(rand(self.world_min + 1.0, self.world_max - 1.0))

                # Don't spawn near robot or goal
                if math.hypot(x - self.x, y - self.y) < 1.5:
                    continue
                if math.hypot(x - self.goal_x, y - self.goal_y) < 1.0:
                    continue

                if self._is_position_safe(x, y, self.pedestrian_radius, exclude_pedestrian_id=-1):
                    # Random goal on opposite side
                    goal_x = float(rand(self.world_min + 0.5, self.world_max - 0.5))
                    goal_y = float(rand(self.world_min + 0.5, self.world_max - 0.5))

                    # Initial velocity toward goal
                    dx = goal_x - x
                    dy = goal_y - y
                    dist = math.hypot(dx, dy) + 1e-6
                    vx = ped_speed * dx / dist
                    vy = ped_speed * dy / dist

                    ped = DynamicAgent(
                        x=x, y=y, vx=vx, vy=vy,
                        radius=self.pedestrian_radius,
                        agent_id=i,
                        goal_x=goal_x, goal_y=goal_y,
                        speed=ped_speed,
                    )
                    ped.record_history()
                    self.pedestrians.append(ped)
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"Failed to place pedestrian {i} safely after {max_attempts_per_ped} attempts. "
                    "Reduce num_pedestrians or adjust clearance."
                )

    def _update_pedestrians(self):
        """Advance all pedestrians one simulation step.

        The reactive-SFM math was extracted to `aig_nav_core.ped_dynamics`
        so the Gazebo deployment can use the same function under
        `mode: training_parity`. Behavior is byte-identical to the
        pre-extraction implementation: goal attraction + robot repulsion
        (within 2 m) + 1.5x per-agent speed cap + reflecting boundary
        with 0.8 damping. See `ped_dynamics.update_pedestrian_state`.
        """
        from .ped_dynamics import PedestrianState, update_pedestrian_state

        def _new_goal():
            gx = float(np.random.uniform(self.world_min + 0.5, self.world_max - 0.5))
            gy = float(np.random.uniform(self.world_min + 0.5, self.world_max - 0.5))
            return gx, gy

        for ped in self.pedestrians:
            ped.record_history()
            state = PedestrianState(
                x=ped.x, y=ped.y, vx=ped.vx, vy=ped.vy,
                goal_x=ped.goal_x, goal_y=ped.goal_y,
                speed=ped.speed, radius=ped.radius,
            )
            update_pedestrian_state(
                state,
                robot_x=self.x, robot_y=self.y,
                dt=self.dt,
                reactivity=self.pedestrian_reactivity,
                world_min=self.world_min, world_max=self.world_max,
                new_goal_fn=_new_goal,
            )
            ped.x = state.x
            ped.y = state.y
            ped.vx = state.vx
            ped.vy = state.vy
            ped.goal_x = state.goal_x
            ped.goal_y = state.goal_y

    def _record_robot_history(self):
        """Record robot state to history"""
        vx = self.lin_vel * math.cos(self.yaw)
        vy = self.lin_vel * math.sin(self.yaw)
        self.robot_history_x.append(self.x)
        self.robot_history_y.append(self.y)
        self.robot_history_vx.append(vx)
        self.robot_history_vy.append(vy)

    def _get_robot_history(self) -> np.ndarray:
        """Get robot history as numpy array [T, 4]"""
        length = self.history_length
        hist_len = len(self.robot_history_x)

        if hist_len == 0:
            return np.zeros((length, 4), dtype=np.float32)

        history = np.zeros((length, 4), dtype=np.float32)
        actual_len = min(hist_len, length)

        for i in range(actual_len):
            idx = -(actual_len - i)
            history[length - actual_len + i, 0] = self.robot_history_x[idx]
            history[length - actual_len + i, 1] = self.robot_history_y[idx]
            history[length - actual_len + i, 2] = self.robot_history_vx[idx]
            history[length - actual_len + i, 3] = self.robot_history_vy[idx]

        # Pad earlier timesteps
        if actual_len < length:
            for i in range(length - actual_len):
                history[i] = history[length - actual_len]

        return history

    def reset(
        self,
        scenario: Optional[Tuple[float, float, float, float, float]] = None,
        eval_seed: Optional[int] = None,
    ):
        """Reset the environment"""
        self.steps = 0
        self.last_distance = None
        # Always clear old pedestrians before sampling a new episode state.
        # Otherwise, previous-episode pedestrians can make robot/goal placement
        # fail intermittently.
        self.pedestrians = []

        rng = np.random.default_rng(eval_seed) if eval_seed is not None else None

        # Domain randomization: randomize obstacles before placing robot/goal
        if self.randomize_obstacles:
            self._generate_random_obstacles(rng)

        # Domain randomization: randomize pedestrian count
        if self.randomize_num_pedestrians:
            if rng is not None:
                self.num_pedestrians = int(rng.integers(
                    self.num_pedestrians_range[0], self.num_pedestrians_range[1] + 1))
            else:
                self.num_pedestrians = int(np.random.randint(
                    self.num_pedestrians_range[0], self.num_pedestrians_range[1] + 1))

        # Clear histories
        self.robot_history_x.clear()
        self.robot_history_y.clear()
        self.robot_history_vx.clear()
        self.robot_history_vy.clear()

        rand = rng.uniform if rng is not None else np.random.uniform

        if scenario is None:
            # Jointly sample robot + goal and then place pedestrians.
            # If any stage fails, retry the full reset state instead of aborting.
            reset_attempts = 20
            robot_attempts = 100
            goal_attempts = 100
            reset_ok = False

            for _ in range(reset_attempts):
                found_pair = False
                for _ in range(robot_attempts):
                    cand_x = float(rand(self.world_min + 1.0, self.world_max - 1.0))
                    cand_y = float(rand(self.world_min + 1.0, self.world_max - 1.0))
                    cand_yaw = float(rand(-math.pi, math.pi))

                    if not self._is_position_safe(
                        cand_x,
                        cand_y,
                        self.robot_radius,
                        extra_margin=self.robot_spawn_clearance,
                    ):
                        continue

                    for _ in range(goal_attempts):
                        cand_goal_x = float(rand(self.world_min + self.goal_margin, self.world_max - self.goal_margin))
                        cand_goal_y = float(rand(self.world_min + self.goal_margin, self.world_max - self.goal_margin))

                        if math.hypot(cand_goal_x - cand_x, cand_goal_y - cand_y) <= self.min_goal_distance:
                            continue
                        if not self._is_position_safe(
                            cand_goal_x,
                            cand_goal_y,
                            0.3,
                            extra_margin=self.goal_spawn_clearance,
                        ):
                            continue

                        self.x = cand_x
                        self.y = cand_y
                        self.yaw = cand_yaw
                        self.goal_x = cand_goal_x
                        self.goal_y = cand_goal_y
                        found_pair = True
                        break

                    if found_pair:
                        break

                if not found_pair:
                    continue

                try:
                    self._create_pedestrians(rng)
                except RuntimeError:
                    # Retry with a new sampled robot/goal pair.
                    continue

                reset_ok = True
                break

            if not reset_ok:
                raise RuntimeError(
                    "Failed to sample a valid reset state (robot/goal/pedestrians)."
                )
        else:
            self.x, self.y, self.yaw, self.goal_x, self.goal_y = scenario
            if not self._is_position_safe(
                self.x,
                self.y,
                self.robot_radius,
                extra_margin=self.robot_spawn_clearance,
            ):
                raise ValueError("Scenario robot spawn is not safe.")
            if math.hypot(self.goal_x - self.x, self.goal_y - self.y) <= self.min_goal_distance:
                raise ValueError("Scenario goal is too close to robot.")
            if self.goal_x < self.world_min + self.goal_margin or self.goal_x > self.world_max - self.goal_margin:
                raise ValueError("Scenario goal violates goal_margin (x).")
            if self.goal_y < self.world_min + self.goal_margin or self.goal_y > self.world_max - self.goal_margin:
                raise ValueError("Scenario goal violates goal_margin (y).")
            if not self._is_position_safe(
                self.goal_x,
                self.goal_y,
                0.3,
                extra_margin=self.goal_spawn_clearance,
            ):
                raise ValueError("Scenario goal is not safe.")

        self.lin_vel = 0.0
        self.ang_vel = 0.0

        # Create pedestrians for scenario-based reset.
        if scenario is not None:
            ped_reset_ok = False
            for _ in range(20):
                try:
                    self._create_pedestrians(rng)
                    ped_reset_ok = True
                    break
                except RuntimeError:
                    continue
            if not ped_reset_ok:
                raise RuntimeError(
                    "Failed to sample pedestrians for scenario reset."
                )

        # Record initial state
        self._record_robot_history()

        return self._get_observation(0.0, 0.0)

    def _world_to_robot(self, x_world: float, y_world: float) -> Tuple[float, float]:
        """Transform world coordinates to robot frame"""
        dx = x_world - self.x
        dy = y_world - self.y
        c = math.cos(-self.yaw)
        s = math.sin(-self.yaw)
        return c * dx - s * dy, s * dx + c * dy

    def _vel_world_to_robot(self, vx_world: float, vy_world: float) -> Tuple[float, float]:
        """Transform world velocity to robot frame"""
        c = math.cos(-self.yaw)
        s = math.sin(-self.yaw)
        return c * vx_world - s * vy_world, s * vx_world + c * vy_world

    def _compute_goal_geometry(self) -> Tuple[float, float, float]:
        """Compute distance and angle to goal"""
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy) + 1e-8
        goal_yaw = math.atan2(dy, dx)
        dtheta = self._wrap_angle(goal_yaw - self.yaw)
        return dist, math.cos(dtheta), math.sin(dtheta)

    def _raycast(self, origin: Tuple[float, float], direction: Tuple[float, float]) -> float:
        """Cast a ray and return distance to nearest obstacle"""
        ox, oy = origin
        dx, dy = direction
        min_dist = self.scan_max

        # Check static obstacles
        for obs in self.static_obstacles:
            cx, cy, r = obs.x, obs.y, obs.radius
            fx = ox - cx
            fy = oy - cy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4.0 * c
            if disc >= 0.0:
                sqrt_disc = math.sqrt(disc)
                for t in [(-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0]:
                    if 0.0 < t < min_dist:
                        min_dist = t

        # Check pedestrians
        for ped in self.pedestrians:
            cx, cy, r = ped.x, ped.y, ped.radius
            fx = ox - cx
            fy = oy - cy
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - r * r
            disc = b * b - 4.0 * c
            if disc >= 0.0:
                sqrt_disc = math.sqrt(disc)
                for t in [(-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0]:
                    if 0.0 < t < min_dist:
                        min_dist = t

        # Check walls
        for wall_check in [
            (self.world_min, 'x', -1), (self.world_max, 'x', 1),
            (self.world_min, 'y', -1), (self.world_max, 'y', 1)
        ]:
            wall_pos, axis, _ = wall_check
            if axis == 'x' and abs(dx) > 1e-6:
                t = (wall_pos - ox) / dx
                if t > 0:
                    y_hit = oy + t * dy
                    if self.world_min <= y_hit <= self.world_max and t < min_dist:
                        min_dist = t
            elif axis == 'y' and abs(dy) > 1e-6:
                t = (wall_pos - oy) / dy
                if t > 0:
                    x_hit = ox + t * dx
                    if self.world_min <= x_hit <= self.world_max and t < min_dist:
                        min_dist = t

        return min_dist

    def _get_scan(self) -> np.ndarray:
        """Get normalized LiDAR scan"""
        scan = np.zeros(self.scan_beams, dtype=np.float32)
        scan_raw = np.zeros(self.scan_beams, dtype=np.float32)

        for i, a in enumerate(self.beam_angles):
            angle = self.yaw + a
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist = self._raycast((self.x, self.y), (dx, dy))
            if self.scan_noise_std > 0:
                dist += np.random.normal(0, self.scan_noise_std)
            dist = max(self.scan_min, min(self.scan_max, dist))
            scan_raw[i] = dist
            scan[i] = (dist - self.scan_min) / (self.scan_max - self.scan_min)

        self.last_scan_raw = scan_raw
        return np.clip(scan, 0.0, 1.0)

    def _check_collision(self) -> bool:
        """Check if robot collided with anything"""
        # Wall collision
        if self.x < self.world_min + self.robot_radius or self.x > self.world_max - self.robot_radius:
            return True
        if self.y < self.world_min + self.robot_radius or self.y > self.world_max - self.robot_radius:
            return True

        # Static obstacle collision
        for obs in self.static_obstacles:
            if math.hypot(self.x - obs.x, self.y - obs.y) < (self.robot_radius + obs.radius):
                return True

        # Pedestrian collision
        for ped in self.pedestrians:
            if math.hypot(self.x - ped.x, self.y - ped.y) < (self.robot_radius + ped.radius):
                return True

        return False

    def _get_reward(self, distance: float, collision: bool, goal: bool,
                    lin_vel: float, ang_vel: float, min_scan: float) -> float:
        """Compute reward"""
        if goal:
            return 100.0
        if collision:
            return -100.0

        # Proximity penalty — quadratic, stronger at close range
        proximity_penalty = 0.5 * (1.5 - min_scan) ** 2 if min_scan < 1.5 else 0.0

        # Progress reward
        progress = 0.0
        if self.last_distance is not None:
            progress = (self.last_distance - distance) * 5.0

        # Step penalty discourages idle hesitation;
        # no lin_vel bonus — progress already drives goal-seeking
        return progress - abs(ang_vel) * 0.2 - proximity_penalty - 0.01

    def _get_pedestrian_states(self) -> Dict:
        """Get all pedestrian states for trajectory prediction"""
        states = {
            'positions': [],        # [N, 2] current positions in robot frame
            'velocities': [],       # [N, 2] current velocities in robot frame
            'histories': [],        # [N, T, 4] history trajectories
            'positions_world': [],  # [N, 2] world frame positions (for aux_pred_loss)
            'num_pedestrians': len(self.pedestrians),
        }

        for ped in self.pedestrians:
            # World frame position (for aux_pred_loss supervision)
            states['positions_world'].append([ped.x, ped.y])

            # Current position in robot frame
            rx, ry = self._world_to_robot(ped.x, ped.y)
            states['positions'].append([rx, ry])

            # Current velocity in robot frame
            rvx, rvy = self._vel_world_to_robot(ped.vx, ped.vy)
            states['velocities'].append([rvx, rvy])

            # History (need to transform each point to robot frame)
            hist = ped.get_history_array(self.history_length)
            hist_robot = np.zeros_like(hist)
            for t in range(len(hist)):
                hx, hy = self._world_to_robot(hist[t, 0], hist[t, 1])
                hvx, hvy = self._vel_world_to_robot(hist[t, 2], hist[t, 3])
                hist_robot[t] = [hx, hy, hvx, hvy]
            states['histories'].append(hist_robot)

        # Convert to numpy arrays
        if len(states['positions']) > 0:
            states['positions'] = np.array(states['positions'], dtype=np.float32)
            states['velocities'] = np.array(states['velocities'], dtype=np.float32)
            states['histories'] = np.array(states['histories'], dtype=np.float32)
            states['positions_world'] = np.array(states['positions_world'], dtype=np.float32)
        else:
            states['positions'] = np.zeros((0, 2), dtype=np.float32)
            states['velocities'] = np.zeros((0, 2), dtype=np.float32)
            states['histories'] = np.zeros((0, self.history_length, 4), dtype=np.float32)
            states['positions_world'] = np.zeros((0, 2), dtype=np.float32)

        return states

    def _get_static_obstacle_states(self) -> np.ndarray:
        """Get static obstacles in robot frame [N, 3] (x, y, radius)"""
        states = []
        for obs in self.static_obstacles:
            rx, ry = self._world_to_robot(obs.x, obs.y)
            states.append([rx, ry, obs.radius])
        return np.array(states, dtype=np.float32) if states else np.zeros((0, 3), dtype=np.float32)

    def _get_observation(self, lin_velocity: float, ang_velocity: float):
        """Get complete observation"""
        scan = self._get_scan()
        distance, cos_v, sin_v = self._compute_goal_geometry()
        collision = self._check_collision()
        goal_reached = distance < self.goal_threshold

        min_scan = float(self.last_scan_raw.min()) if self.last_scan_raw is not None else self.scan_max
        reward = self._get_reward(distance, collision, goal_reached, lin_velocity, ang_velocity, min_scan)

        self.last_distance = distance

        # Get detailed states for AIG
        pedestrian_states = self._get_pedestrian_states()
        static_obstacles = self._get_static_obstacle_states()
        robot_history = self._get_robot_history()

        # Goal in robot frame
        goal_rx, goal_ry = self._world_to_robot(self.goal_x, self.goal_y)

        # Observation dictionary
        obs = {
            'scan': scan,                           # [32] normalized scan
            'distance': distance,                    # scalar
            'cos_v': cos_v,                         # scalar
            'sin_v': sin_v,                         # scalar
            'collision': collision,                  # bool
            'goal_reached': goal_reached,           # bool
            'reward': reward,                       # scalar
            'robot_vel': np.array([lin_velocity, ang_velocity], dtype=np.float32),
            'robot_history': robot_history,          # [T, 4]
            'goal_position': np.array([goal_rx, goal_ry], dtype=np.float32),
            'pedestrian_states': pedestrian_states,  # dict with positions, velocities, histories
            'static_obstacles': static_obstacles,    # [N_static, 3]
            'robot_pose': np.array([self.x, self.y, self.yaw], dtype=np.float32),
        }

        return obs

    def step(self, lin_velocity: float, ang_velocity: float):
        """Execute one step"""
        self.lin_vel = float(lin_velocity)
        self.ang_vel = float(ang_velocity)

        # Record robot history before update
        self._record_robot_history()

        # Update robot pose
        self.yaw = self._wrap_angle(self.yaw + self.ang_vel * self.dt)
        self.x += self.lin_vel * math.cos(self.yaw) * self.dt
        self.y += self.lin_vel * math.sin(self.yaw) * self.dt

        # Update pedestrians (with reactive behavior)
        self._update_pedestrians()

        self.steps += 1

        obs = self._get_observation(self.lin_vel, self.ang_vel)

        if self.render_enabled:
            self.render()

        return obs

    def render(self):
        """Render the environment"""
        if not self.render_enabled or pygame is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill((30, 30, 40))

        # Draw walls
        pygame.draw.rect(
            self.screen, (90, 90, 100),
            pygame.Rect(0, 0, self.screen_size, self.screen_size), width=6
        )

        # Draw static obstacles
        for obs in self.static_obstacles:
            sx, sy = self._world_to_screen(obs.x, obs.y)
            pygame.draw.circle(self.screen, (120, 120, 120), (sx, sy), int(obs.radius * self.scale))

        # Draw pedestrians and their trajectories
        for ped in self.pedestrians:
            # Draw history trail
            if len(ped.history_x) > 1:
                points = []
                for hx, hy in zip(ped.history_x, ped.history_y):
                    points.append(self._world_to_screen(hx, hy))
                if len(points) > 1:
                    pygame.draw.lines(self.screen, (180, 100, 100), False, points, 2)

            # Draw pedestrian
            px, py = self._world_to_screen(ped.x, ped.y)
            pygame.draw.circle(self.screen, (220, 100, 100), (px, py), int(ped.radius * self.scale))

            # Draw velocity vector
            vx_screen = int(ped.vx * self.scale * 0.5)
            vy_screen = int(-ped.vy * self.scale * 0.5)
            pygame.draw.line(self.screen, (255, 150, 150), (px, py), (px + vx_screen, py + vy_screen), 2)

        # Draw scan
        if self.last_scan_raw is not None:
            for i, dist in enumerate(self.last_scan_raw):
                angle = self.yaw + self.beam_angles[i]
                ex = self.x + dist * math.cos(angle)
                ey = self.y + dist * math.sin(angle)
                sx, sy = self._world_to_screen(self.x, self.y)
                exs, eys = self._world_to_screen(ex, ey)
                intensity = max(0.2, min(1.0, dist / self.scan_max))
                color = (int(40 + 120 * intensity), int(160 + 80 * intensity), int(220 * intensity))
                pygame.draw.line(self.screen, color, (sx, sy), (exs, eys), 1)

        # Draw goal
        gx, gy = self._world_to_screen(self.goal_x, self.goal_y)
        pygame.draw.circle(self.screen, (80, 200, 120), (gx, gy), int(0.15 * self.scale))

        # Draw robot
        rx, ry = self._world_to_screen(self.x, self.y)
        pygame.draw.circle(self.screen, (80, 160, 240), (rx, ry), int(self.robot_radius * self.scale))

        # Draw robot heading
        hx = rx + int(math.cos(self.yaw) * self.robot_radius * self.scale * 1.5)
        hy = ry - int(math.sin(self.yaw) * self.robot_radius * self.scale * 1.5)
        pygame.draw.line(self.screen, (240, 240, 240), (rx, ry), (hx, hy), 3)

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.render_fps)


if __name__ == "__main__":
    # Test the environment
    env = PygameNavEnvAIGNav(render=True, num_pedestrians=4)
    obs = env.reset()

    print("Environment test:")
    print(f"  Scan shape: {obs['scan'].shape}")
    print(f"  Robot history shape: {obs['robot_history'].shape}")
    print(f"  Pedestrian positions shape: {obs['pedestrian_states']['positions'].shape}")
    print(f"  Pedestrian histories shape: {obs['pedestrian_states']['histories'].shape}")
    print(f"  Static obstacles shape: {obs['static_obstacles'].shape}")

    for step in range(500):
        # Random action
        lin_vel = np.random.uniform(-0.2, 0.5)
        ang_vel = np.random.uniform(-0.5, 0.5)

        obs = env.step(lin_vel, ang_vel)

        if obs['collision'] or obs['goal_reached']:
            print(f"Episode ended at step {step}: collision={obs['collision']}, goal={obs['goal_reached']}")
            obs = env.reset()

    env.close()
