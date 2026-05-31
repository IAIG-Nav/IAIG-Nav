"""
Replay Buffer for AIG-Nav.

Stores transitions with pedestrian histories and static obstacles to support
action-conditioned predictive interaction graphs.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
import random
import threading


class AIGNavReplayBuffer:
    """
    Replay buffer for AIG-Nav that stores pedestrian observations
    for trajectory prediction and interaction modeling.
    """

    def __init__(
        self,
        buffer_size: int = 1000000,
        scan_dim: int = 32,
        core_dim: int = 3,
        action_dim: int = 2,
        max_pedestrians: int = 10,
        history_length: int = 10,
        max_static_obstacles: int = 8,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.buffer_size = int(buffer_size)
        self.scan_dim = scan_dim
        self.core_dim = core_dim
        self.action_dim = action_dim
        self.max_pedestrians = max_pedestrians
        self.history_length = history_length
        self.max_static_obstacles = max_static_obstacles
        self.device = device

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Standard RL data
        self.scans = np.zeros((self.buffer_size, scan_dim), dtype=np.float32)
        self.cores = np.zeros((self.buffer_size, core_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_scans = np.zeros((self.buffer_size, scan_dim), dtype=np.float32)
        self.next_cores = np.zeros((self.buffer_size, core_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        # Pedestrian data (padded to max_pedestrians)
        self.ped_positions = np.zeros((self.buffer_size, max_pedestrians, 2), dtype=np.float32)
        self.ped_velocities = np.zeros((self.buffer_size, max_pedestrians, 2), dtype=np.float32)
        self.ped_histories = np.zeros(
            (self.buffer_size, max_pedestrians, history_length, 4), dtype=np.float32
        )
        self.ped_masks = np.zeros((self.buffer_size, max_pedestrians), dtype=np.bool_)
        self.num_peds = np.zeros(self.buffer_size, dtype=np.int32)
        # World coordinates for aux_pred_loss supervision
        self.ped_positions_world = np.zeros((self.buffer_size, max_pedestrians, 2), dtype=np.float32)

        # Robot pose at time t for correct coordinate frame transformation in aux_pred_loss
        self.robot_poses = np.zeros((self.buffer_size, 3), dtype=np.float32)  # (x, y, yaw)

        self.next_ped_positions = np.zeros((self.buffer_size, max_pedestrians, 2), dtype=np.float32)
        self.next_ped_velocities = np.zeros((self.buffer_size, max_pedestrians, 2), dtype=np.float32)
        self.next_ped_histories = np.zeros(
            (self.buffer_size, max_pedestrians, history_length, 4), dtype=np.float32
        )
        self.next_ped_masks = np.zeros((self.buffer_size, max_pedestrians), dtype=np.bool_)
        self.next_num_peds = np.zeros(self.buffer_size, dtype=np.int32)
        # World coordinates for aux_pred_loss supervision
        self.next_ped_positions_world = np.zeros((self.buffer_size, max_pedestrians, 2), dtype=np.float32)

        # Static obstacle data (padded)
        self.static_obstacles = np.zeros(
            (self.buffer_size, max_static_obstacles, 3), dtype=np.float32
        )
        self.static_masks = np.zeros((self.buffer_size, max_static_obstacles), dtype=np.bool_)
        self.next_static_obstacles = np.zeros(
            (self.buffer_size, max_static_obstacles, 3), dtype=np.float32
        )
        self.next_static_masks = np.zeros((self.buffer_size, max_static_obstacles), dtype=np.bool_)

        self.ptr = 0
        self.count = 0

        self._lock = threading.Lock()

    def add(
        self,
        scan: np.ndarray,
        core: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_scan: np.ndarray,
        next_core: np.ndarray,
        done: bool,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        ped_histories: np.ndarray,
        next_ped_positions: np.ndarray,
        next_ped_velocities: np.ndarray,
        next_ped_histories: np.ndarray,
        static_obstacles: np.ndarray,
        next_static_obstacles: np.ndarray,
        ped_positions_world: np.ndarray = None,
        next_ped_positions_world: np.ndarray = None,
        robot_pose: np.ndarray = None,
    ):
        """Add a transition to the buffer."""
        with self._lock:
            idx = self.ptr

            self.scans[idx] = scan
            self.cores[idx] = core
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_scans[idx] = next_scan
            self.next_cores[idx] = next_core
            self.dones[idx] = float(done)

            # Store pedestrian data (pad/truncate to max_pedestrians)
            n_peds = min(len(ped_positions), self.max_pedestrians) if len(ped_positions) > 0 else 0
            self.num_peds[idx] = n_peds
            self.ped_positions[idx] = 0
            self.ped_velocities[idx] = 0
            self.ped_histories[idx] = 0
            self.ped_masks[idx] = False
            self.ped_positions_world[idx] = 0

            if n_peds > 0:
                self.ped_positions[idx, :n_peds] = ped_positions[:n_peds]
                self.ped_velocities[idx, :n_peds] = ped_velocities[:n_peds]
                self.ped_histories[idx, :n_peds] = ped_histories[:n_peds]
                self.ped_masks[idx, :n_peds] = True
                if ped_positions_world is not None:
                    self.ped_positions_world[idx, :n_peds] = ped_positions_world[:n_peds]

            # Robot pose at time t
            if robot_pose is not None:
                self.robot_poses[idx] = robot_pose
            else:
                self.robot_poses[idx] = 0

            # Next pedestrian data
            n_next_peds = min(len(next_ped_positions), self.max_pedestrians) if len(next_ped_positions) > 0 else 0
            self.next_num_peds[idx] = n_next_peds
            self.next_ped_positions[idx] = 0
            self.next_ped_velocities[idx] = 0
            self.next_ped_histories[idx] = 0
            self.next_ped_masks[idx] = False
            self.next_ped_positions_world[idx] = 0

            if n_next_peds > 0:
                self.next_ped_positions[idx, :n_next_peds] = next_ped_positions[:n_next_peds]
                self.next_ped_velocities[idx, :n_next_peds] = next_ped_velocities[:n_next_peds]
                self.next_ped_histories[idx, :n_next_peds] = next_ped_histories[:n_next_peds]
                self.next_ped_masks[idx, :n_next_peds] = True
                if next_ped_positions_world is not None:
                    self.next_ped_positions_world[idx, :n_next_peds] = next_ped_positions_world[:n_next_peds]

            # Static obstacles (pad/truncate)
            self.static_obstacles[idx] = 0
            self.static_masks[idx] = False
            n_static = min(len(static_obstacles), self.max_static_obstacles) if len(static_obstacles) > 0 else 0
            if n_static > 0:
                self.static_obstacles[idx, :n_static] = static_obstacles[:n_static]
                self.static_masks[idx, :n_static] = True

            self.next_static_obstacles[idx] = 0
            self.next_static_masks[idx] = False
            n_next_static = min(len(next_static_obstacles), self.max_static_obstacles) if len(next_static_obstacles) > 0 else 0
            if n_next_static > 0:
                self.next_static_obstacles[idx, :n_next_static] = next_static_obstacles[:n_next_static]
                self.next_static_masks[idx, :n_next_static] = True

            self.ptr = (self.ptr + 1) % self.buffer_size
            self.count = min(self.count + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of transitions uniformly without replacement."""
        with self._lock:
            if self.count < batch_size:
                return None

            indices = np.random.choice(self.count, size=batch_size, replace=False)

            batch = {
                "scan": torch.FloatTensor(self.scans[indices]),
                "core": torch.FloatTensor(self.cores[indices]),
                "action": torch.FloatTensor(self.actions[indices]),
                "reward": torch.FloatTensor(self.rewards[indices]),
                "next_scan": torch.FloatTensor(self.next_scans[indices]),
                "next_core": torch.FloatTensor(self.next_cores[indices]),
                "done": torch.FloatTensor(self.dones[indices]),
                "ped_positions": torch.FloatTensor(self.ped_positions[indices]),
                "ped_velocities": torch.FloatTensor(self.ped_velocities[indices]),
                "ped_histories": torch.FloatTensor(self.ped_histories[indices]),
                "ped_masks": torch.BoolTensor(self.ped_masks[indices]),
                "ped_positions_world": torch.FloatTensor(self.ped_positions_world[indices]),
                "robot_pose": torch.FloatTensor(self.robot_poses[indices]),
                "next_ped_positions": torch.FloatTensor(self.next_ped_positions[indices]),
                "next_ped_velocities": torch.FloatTensor(self.next_ped_velocities[indices]),
                "next_ped_histories": torch.FloatTensor(self.next_ped_histories[indices]),
                "next_ped_masks": torch.BoolTensor(self.next_ped_masks[indices]),
                "next_ped_positions_world": torch.FloatTensor(self.next_ped_positions_world[indices]),
                "static_obstacles": torch.FloatTensor(self.static_obstacles[indices]),
                "static_masks": torch.BoolTensor(self.static_masks[indices]),
                "next_static_obstacles": torch.FloatTensor(self.next_static_obstacles[indices]),
                "next_static_masks": torch.BoolTensor(self.next_static_masks[indices]),
            }

            return batch

    def __len__(self):
        return self.count


if __name__ == "__main__":
    # Test the replay buffer
    buffer = AIGNavReplayBuffer(
        buffer_size=1000,
        scan_dim=32,
        core_dim=3,
        max_pedestrians=10,
        history_length=10,
        max_static_obstacles=5,
    )

    # Add some transitions
    for i in range(100):
        scan = np.random.randn(32).astype(np.float32)
        core = np.random.randn(3).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        reward = np.random.randn()
        next_scan = np.random.randn(32).astype(np.float32)
        next_core = np.random.randn(3).astype(np.float32)
        done = np.random.random() > 0.9

        n_peds = np.random.randint(0, 6)
        ped_pos = np.random.randn(n_peds, 2).astype(np.float32)
        ped_vel = np.random.randn(n_peds, 2).astype(np.float32) * 0.5

        ped_hist = np.random.randn(n_peds, 10, 4).astype(np.float32)
        static_obs = np.random.randn(3, 3).astype(np.float32)

        buffer.add(
            scan, core, action, reward, next_scan, next_core, done,
            ped_pos, ped_vel, ped_hist,
            ped_pos, ped_vel, ped_hist,
            static_obs, static_obs,
        )

    print(f"Buffer size: {len(buffer)}")

    # Sample a batch
    batch = buffer.sample(32)
    print(f"Batch keys: {batch.keys()}")
    print(f"Scan shape: {batch['scan'].shape}")
    print(f"Ped positions shape: {batch['ped_positions'].shape}")
    print(f"Ped masks shape: {batch['ped_masks'].shape}")
    print(f"Ped histories shape: {batch['ped_histories'].shape}")
    print(f"Static obstacles shape: {batch['static_obstacles'].shape}")

    print("\nReplay buffer tests passed!")
