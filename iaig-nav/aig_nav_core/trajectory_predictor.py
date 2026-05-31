"""
Trajectory Predictor for Action-conditioned Interaction Graph (AIG)

This module predicts future trajectories of pedestrians, conditioned on
the robot's planned action. This enables:
1. Predicting where pedestrians will be in the future
2. Modeling pedestrian reactions to robot behavior (reactive modeling)
3. Computing future interaction probabilities

Key Innovation: Action-Conditioned Trajectory Prediction
- Traditional: τ_ped = f(history_ped)
- Ours: τ_ped = f(history_ped, robot_action)
  Pedestrian trajectory depends on what the robot plans to do
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class TrajectoryEncoder(nn.Module):
    """
    Encode trajectory history using LSTM.

    Input: [batch, seq_len, 4] where 4 = (x, y, vx, vy)
    Output: [batch, hidden_dim] encoded representation
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_dim = hidden_dim

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: [batch, seq_len, 4] history trajectory

        Returns:
            encoding: [batch, hidden_dim] encoded representation
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(trajectory)
        # Use last hidden state
        return h_n[-1]  # [batch, hidden_dim]


class TrajectoryDecoder(nn.Module):
    """
    Decode future trajectory from encoded state.

    Input: [batch, hidden_dim] encoded state
    Output: [batch, pred_horizon, 2] predicted future positions (x, y)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        pred_horizon: int = 12,
        output_dim: int = 2,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        encoding: torch.Tensor,
        last_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoding: [batch, hidden_dim] encoded trajectory
            last_position: [batch, 2] last known position

        Returns:
            predictions: [batch, pred_horizon, 2] predicted positions
        """
        batch_size = encoding.size(0)
        device = encoding.device

        # Initialize hidden state from encoding
        h_0 = encoding.unsqueeze(0)  # [1, batch, hidden_dim]
        c_0 = torch.zeros_like(h_0)

        # Autoregressive decoding
        predictions = []
        current_pos = last_position  # [batch, 2]

        hidden = (h_0, c_0)

        for t in range(self.pred_horizon):
            # LSTM step
            lstm_input = current_pos.unsqueeze(1)  # [batch, 1, 2]
            output, hidden = self.lstm(lstm_input, hidden)

            # Predict delta position
            delta = self.output_layer(output.squeeze(1))  # [batch, 2]

            # Update position
            current_pos = current_pos + delta
            predictions.append(current_pos)

        return torch.stack(predictions, dim=1)  # [batch, pred_horizon, 2]


class ReactiveTrajectoryPredictor(nn.Module):
    """
    Predict pedestrian trajectories conditioned on robot action.

    This is the key innovation: pedestrians react to robot's planned action.

    Architecture:
        pedestrian_history → Encoder → ped_encoding
        robot_action → MLP → action_encoding
        [ped_encoding, action_encoding] → Fusion → fused_encoding
        fused_encoding → Decoder → predicted_trajectory
    """

    def __init__(
        self,
        history_len: int = 10,
        pred_horizon: int = 12,
        hidden_dim: int = 64,
        action_dim: int = 2,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim

        # Trajectory encoder
        self.traj_encoder = TrajectoryEncoder(
            input_dim=4,  # x, y, vx, vy
            hidden_dim=hidden_dim,
        )

        # Robot action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Fusion layer: combine pedestrian encoding with robot action
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            pred_horizon=pred_horizon,
            output_dim=2,
        )

        # Output normalization
        self.layer_norm = nn.LayerNorm(2)

    def forward(
        self,
        ped_history: torch.Tensor,
        robot_action: torch.Tensor,
        last_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict pedestrian trajectory conditioned on robot action.

        Args:
            ped_history: [batch, history_len, 4] pedestrian history (x, y, vx, vy)
            robot_action: [batch, 2] robot's planned action (lin_vel, ang_vel)
            last_position: [batch, 2] last known position (optional, uses history if None)

        Returns:
            predicted_trajectory: [batch, pred_horizon, 2] predicted positions
        """
        # Encode pedestrian history
        ped_encoding = self.traj_encoder(ped_history)  # [batch, hidden_dim]

        # Encode robot action
        action_encoding = self.action_encoder(robot_action)  # [batch, hidden_dim]

        # Fuse pedestrian state with robot action (reactive modeling)
        fused = torch.cat([ped_encoding, action_encoding], dim=-1)  # [batch, hidden_dim*2]
        fused_encoding = self.fusion(fused)  # [batch, hidden_dim]

        # Get last position if not provided
        if last_position is None:
            last_position = ped_history[:, -1, :2]  # [batch, 2]

        # Decode future trajectory
        predictions = self.decoder(fused_encoding, last_position)

        return predictions

    def predict_multiple_actions(
        self,
        ped_history: torch.Tensor,
        robot_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict trajectories for multiple candidate robot actions.

        Args:
            ped_history: [batch, history_len, 4] pedestrian history
            robot_actions: [num_actions, 2] candidate robot actions

        Returns:
            predictions: [batch, num_actions, pred_horizon, 2]
        """
        batch_size = ped_history.size(0)
        num_actions = robot_actions.size(0)

        # Expand pedestrian history for all actions
        ped_history_exp = ped_history.unsqueeze(1).expand(-1, num_actions, -1, -1)
        ped_history_exp = ped_history_exp.reshape(batch_size * num_actions, self.history_len, 4)

        # Expand robot actions for all pedestrians
        robot_actions_exp = robot_actions.unsqueeze(0).expand(batch_size, -1, -1)
        robot_actions_exp = robot_actions_exp.reshape(batch_size * num_actions, 2)

        # Get last positions
        last_pos = ped_history[:, -1, :2]  # [batch, 2]
        last_pos_exp = last_pos.unsqueeze(1).expand(-1, num_actions, -1)
        last_pos_exp = last_pos_exp.reshape(batch_size * num_actions, 2)

        # Predict
        predictions = self.forward(ped_history_exp, robot_actions_exp, last_pos_exp)
        predictions = predictions.reshape(batch_size, num_actions, self.pred_horizon, 2)

        return predictions


class MultiAgentTrajectoryPredictor(nn.Module):
    """
    Predict trajectories for multiple pedestrians simultaneously,
    considering interactions between pedestrians and robot.

    This extends ReactiveTrajectoryPredictor to handle:
    1. Multiple pedestrians
    2. Pedestrian-pedestrian interactions
    3. Robot-pedestrian interactions
    """

    def __init__(
        self,
        history_len: int = 10,
        pred_horizon: int = 12,
        hidden_dim: int = 64,
        action_dim: int = 2,
        max_pedestrians: int = 10,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.max_pedestrians = max_pedestrians

        # Individual trajectory encoder
        self.traj_encoder = TrajectoryEncoder(
            input_dim=4,
            hidden_dim=hidden_dim,
        )

        # Robot action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Social interaction attention
        self.social_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Fusion with robot action
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            hidden_dim=hidden_dim,
            pred_horizon=pred_horizon,
            output_dim=2,
        )

    def forward(
        self,
        ped_histories: torch.Tensor,
        robot_action: torch.Tensor,
        ped_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict trajectories for all pedestrians.

        Args:
            ped_histories: [batch, num_peds, history_len, 4] all pedestrian histories
            robot_action: [batch, 2] robot's planned action
            ped_mask: [batch, num_peds] mask for valid pedestrians (optional)

        Returns:
            predictions: [batch, num_peds, pred_horizon, 2] predicted trajectories
        """
        batch_size, num_peds, hist_len, feat_dim = ped_histories.shape
        device = ped_histories.device

        # Encode each pedestrian's history
        ped_histories_flat = ped_histories.reshape(batch_size * num_peds, hist_len, feat_dim)
        ped_encodings_flat = self.traj_encoder(ped_histories_flat)  # [B*N, hidden]
        ped_encodings = ped_encodings_flat.reshape(batch_size, num_peds, self.hidden_dim)

        # Social attention between pedestrians
        social_encodings, _ = self.social_attention(
            ped_encodings,
            ped_encodings,
            ped_encodings,
            key_padding_mask=~ped_mask if ped_mask is not None else None,
        )

        # Encode robot action
        action_encoding = self.action_encoder(robot_action)  # [batch, hidden]

        # Fuse with robot action (same action affects all pedestrians)
        action_encoding_exp = action_encoding.unsqueeze(1).expand(-1, num_peds, -1)
        fused = torch.cat([social_encodings, action_encoding_exp], dim=-1)
        fused_encodings = self.fusion(fused)  # [batch, num_peds, hidden]

        # Decode each pedestrian's trajectory
        fused_flat = fused_encodings.reshape(batch_size * num_peds, self.hidden_dim)
        last_positions = ped_histories[:, :, -1, :2].reshape(batch_size * num_peds, 2)

        predictions_flat = self.decoder(fused_flat, last_positions)
        predictions = predictions_flat.reshape(batch_size, num_peds, self.pred_horizon, 2)

        if ped_mask is not None:
            last_positions = ped_histories[:, :, -1, :2]
            mask = ped_mask.unsqueeze(-1).unsqueeze(-1)
            predictions = torch.where(mask, predictions, last_positions.unsqueeze(2))

        return predictions


class SimpleTrajectoryPredictor(nn.Module):
    """
    Simplified trajectory predictor using constant velocity assumption
    with learned correction for robot interaction.

    This is faster and more stable for initial training.
    """

    def __init__(
        self,
        pred_horizon: int = 12,
        hidden_dim: int = 32,
        dt: float = 0.1,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.dt = dt

        # Robot influence network: how much robot action affects pedestrian
        self.robot_influence = nn.Sequential(
            nn.Linear(6, hidden_dim),  # ped_state (4) + robot_action (2)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # velocity correction
            nn.Tanh(),
        )

        # Scaling factor for robot influence (learnable)
        self.influence_scale = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        robot_action: torch.Tensor,
        robot_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict pedestrian trajectories with robot influence.

        Args:
            ped_positions: [batch, num_peds, 2] current positions
            ped_velocities: [batch, num_peds, 2] current velocities
            robot_action: [batch, 2] robot action (lin_vel, ang_vel)
            robot_position: [batch, 2] robot position

        Returns:
            predictions: [batch, num_peds, pred_horizon, 2] predicted positions
        """
        batch_size, num_peds, _ = ped_positions.shape
        device = ped_positions.device

        # Expand robot action for all pedestrians
        robot_action_exp = robot_action.unsqueeze(1).expand(-1, num_peds, -1)

        # Concatenate pedestrian state with robot action
        ped_state = torch.cat([ped_positions, ped_velocities], dim=-1)  # [B, N, 4]
        combined = torch.cat([ped_state, robot_action_exp], dim=-1)  # [B, N, 6]

        # Compute velocity correction due to robot influence
        combined_flat = combined.reshape(batch_size * num_peds, 6)
        vel_correction = self.robot_influence(combined_flat)  # [B*N, 2]
        vel_correction = vel_correction.reshape(batch_size, num_peds, 2)

        # Apply correction scaled by distance to robot
        robot_pos_exp = robot_position.unsqueeze(1).expand(-1, num_peds, -1)
        dist_to_robot = torch.norm(ped_positions - robot_pos_exp, dim=-1, keepdim=True)
        influence_weight = torch.exp(-dist_to_robot / 2.0)  # Decay with distance

        corrected_vel = ped_velocities + self.influence_scale * influence_weight * vel_correction

        # Predict trajectory using corrected velocity
        predictions = []
        current_pos = ped_positions.clone()

        for t in range(self.pred_horizon):
            current_pos = current_pos + corrected_vel * self.dt
            predictions.append(current_pos.clone())

        return torch.stack(predictions, dim=2)  # [batch, num_peds, pred_horizon, 2]


def compute_collision_probability(
    robot_trajectory: torch.Tensor,
    ped_trajectories: torch.Tensor,
    robot_radius: float = 0.25,
    ped_radius: float = 0.3,
    safety_margin: float = 0.2,
    ped_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute collision probability between robot and pedestrians over time.

    Args:
        robot_trajectory: [batch, pred_horizon, 2] robot predicted trajectory
        ped_trajectories: [batch, num_peds, pred_horizon, 2] pedestrian trajectories
        robot_radius: robot radius
        ped_radius: pedestrian radius
        safety_margin: additional safety distance

    Returns:
        collision_probs: [batch, num_peds] collision probability for each pedestrian
    """
    # Minimum safe distance
    safe_dist = robot_radius + ped_radius + safety_margin

    # Compute distances over time
    # robot_trajectory: [B, T, 2] -> [B, 1, T, 2]
    # ped_trajectories: [B, N, T, 2]
    robot_exp = robot_trajectory.unsqueeze(1)  # [B, 1, T, 2]

    # Distance at each timestep
    distances = torch.norm(robot_exp - ped_trajectories, dim=-1)  # [B, N, T]

    # Minimum distance over prediction horizon
    min_distances, _ = distances.min(dim=-1)  # [B, N]

    # Convert to collision probability using sigmoid
    # P(collision) ≈ σ((safe_dist - min_dist) / temperature)
    temperature = 0.3
    collision_probs = torch.sigmoid((safe_dist - min_distances) / temperature)

    if ped_mask is not None:
        collision_probs = collision_probs * ped_mask.float()

    return collision_probs


def compute_pairwise_collision_probability(
    ped_trajectories: torch.Tensor,
    ped_radius: float = 0.3,
    safety_margin: float = 0.2,
    ped_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute pairwise collision probability between pedestrians.

    Args:
        ped_trajectories: [batch, num_peds, pred_horizon, 2]
        ped_radius: pedestrian radius
        safety_margin: additional safety distance
        ped_mask: [batch, num_peds] optional mask

    Returns:
        collision_probs: [batch, num_peds, num_peds]
    """
    batch_size, num_peds, horizon, _ = ped_trajectories.shape
    device = ped_trajectories.device

    if num_peds == 0:
        return torch.zeros(batch_size, 0, 0, device=device)

    ped_i = ped_trajectories.unsqueeze(2)  # [B, N, 1, T, 2]
    ped_j = ped_trajectories.unsqueeze(1)  # [B, 1, N, T, 2]
    distances = torch.norm(ped_i - ped_j, dim=-1)  # [B, N, N, T]

    min_distances, _ = distances.min(dim=-1)  # [B, N, N]

    safe_dist = 2.0 * ped_radius + safety_margin
    temperature = 0.3
    collision_probs = torch.sigmoid((safe_dist - min_distances) / temperature)

    # Zero self-collisions
    diag_mask = torch.eye(num_peds, device=device).bool().unsqueeze(0)
    collision_probs = collision_probs.masked_fill(diag_mask, 0.0)

    if ped_mask is not None:
        mask_i = ped_mask.unsqueeze(2)  # [B, N, 1]
        mask_j = ped_mask.unsqueeze(1)  # [B, 1, N]
        collision_probs = collision_probs * (mask_i & mask_j).float()

    return collision_probs


def compute_static_collision_probability(
    robot_trajectory: torch.Tensor,
    static_obstacles: torch.Tensor,
    robot_radius: float = 0.25,
    safety_margin: float = 0.2,
    static_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute collision probability between robot trajectory and static obstacles.

    Args:
        robot_trajectory: [batch, pred_horizon, 2]
        static_obstacles: [batch, num_static, 3] (x, y, radius)
        robot_radius: robot radius
        safety_margin: additional safety distance
        static_mask: [batch, num_static] optional mask

    Returns:
        collision_probs: [batch, num_static]
    """
    if static_obstacles.numel() == 0:
        return torch.zeros(static_obstacles.size(0), 0, device=robot_trajectory.device)

    robot_traj = robot_trajectory.unsqueeze(1)  # [B, 1, T, 2]
    obs_pos = static_obstacles[..., :2].unsqueeze(2)  # [B, N, 1, 2]
    obs_r = static_obstacles[..., 2]  # [B, N]

    distances = torch.norm(robot_traj - obs_pos, dim=-1)  # [B, N, T]
    min_distances, _ = distances.min(dim=-1)  # [B, N]

    safe_dist = robot_radius + obs_r + safety_margin
    temperature = 0.3
    collision_probs = torch.sigmoid((safe_dist - min_distances) / temperature)

    if static_mask is not None:
        collision_probs = collision_probs * static_mask.float()

    return collision_probs


def predict_robot_trajectory(
    robot_position: torch.Tensor,
    robot_yaw: torch.Tensor,
    action: torch.Tensor,
    pred_horizon: int = 12,
    dt: float = 0.1,
) -> torch.Tensor:
    """
    Predict robot trajectory given action.

    Args:
        robot_position: [batch, 2] current position
        robot_yaw: [batch] current yaw
        action: [batch, 2] (lin_vel, ang_vel)
        pred_horizon: prediction steps
        dt: time step

    Returns:
        trajectory: [batch, pred_horizon, 2] predicted positions
    """
    batch_size = robot_position.size(0)
    device = robot_position.device

    positions = []
    x = robot_position[:, 0].clone()
    y = robot_position[:, 1].clone()
    yaw = robot_yaw.clone()

    lin_vel = action[:, 0]
    ang_vel = action[:, 1]

    for t in range(pred_horizon):
        # Update yaw
        yaw = yaw + ang_vel * dt

        # Update position
        x = x + lin_vel * torch.cos(yaw) * dt
        y = y + lin_vel * torch.sin(yaw) * dt

        positions.append(torch.stack([x, y], dim=-1))

    return torch.stack(positions, dim=1)  # [batch, pred_horizon, 2]


if __name__ == "__main__":
    # Test the trajectory predictor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Test ReactiveTrajectoryPredictor
    predictor = ReactiveTrajectoryPredictor(
        history_len=10,
        pred_horizon=12,
        hidden_dim=64,
    ).to(device)

    batch_size = 4
    history = torch.randn(batch_size, 10, 4).to(device)
    action = torch.randn(batch_size, 2).to(device)

    pred = predictor(history, action)
    print(f"ReactiveTrajectoryPredictor output shape: {pred.shape}")
    assert pred.shape == (batch_size, 12, 2)

    # Test multiple actions
    actions = torch.randn(5, 2).to(device)
    preds = predictor.predict_multiple_actions(history, actions)
    print(f"Multiple actions prediction shape: {preds.shape}")
    assert preds.shape == (batch_size, 5, 12, 2)

    # Test SimpleTrajectoryPredictor
    simple_pred = SimpleTrajectoryPredictor(pred_horizon=12).to(device)

    ped_pos = torch.randn(batch_size, 3, 2).to(device)
    ped_vel = torch.randn(batch_size, 3, 2).to(device) * 0.5
    robot_action = torch.randn(batch_size, 2).to(device)
    robot_pos = torch.zeros(batch_size, 2).to(device)

    simple_preds = simple_pred(ped_pos, ped_vel, robot_action, robot_pos)
    print(f"SimpleTrajectoryPredictor output shape: {simple_preds.shape}")
    assert simple_preds.shape == (batch_size, 3, 12, 2)

    # Test collision probability
    robot_traj = torch.randn(batch_size, 12, 2).to(device)
    ped_trajs = torch.randn(batch_size, 3, 12, 2).to(device)
    coll_probs = compute_collision_probability(robot_traj, ped_trajs)
    print(f"Collision probabilities shape: {coll_probs.shape}")
    print(f"Collision probs range: [{coll_probs.min().item():.3f}, {coll_probs.max().item():.3f}]")

    print("\nAll tests passed!")
