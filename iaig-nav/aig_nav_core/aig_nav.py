"""
AIG-Nav: Action-conditioned Interaction Graph for Social Navigation

This module implements a SAC-based agent with Action-conditioned Interaction Graph encoding.

Key idea:
The policy learns to choose actions based on predicted future interactions,
not just current state. This enables proactive, anticipatory navigation.

Architecture:
    Observation -> AIG Encoder (action-conditioned) -> Context
    Context + Scan + Core -> Actor -> Action Distribution
    Context + Scan + Core + Action -> Critic -> Q-Value
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from statistics import mean
import threading

from .aig_encoder import AIGGraphEncoder, ActionSetAggregator, IterativeEquilibriumModule
from .noop_writer import NoOpSummaryWriter


# ============================================================================
# Distribution Classes
# ============================================================================

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.999, 0.999))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ============================================================================
# MLP Utilities
# ============================================================================

def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    """Create MLP with specified architecture."""
    if hidden_depth == 0:
        return nn.Linear(input_dim, output_dim)

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(hidden_depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))

    return nn.Sequential(*layers)


def weight_init(m):
    """Custom weight initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# ============================================================================
# AIG Actor
# ============================================================================

class AIGActor(nn.Module):
    """
    Actor network that uses AIG encoding for decision making.

    The actor conditions its action on predicted future interactions,
    enabling proactive navigation behavior.
    """

    def __init__(
        self,
        scan_dim: int,
        core_dim: int,
        action_dim: int,
        aig_output_dim: int = 128,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        log_std_bounds: tuple = (-2, 2),
    ):
        super().__init__()

        self.scan_dim = scan_dim
        self.core_dim = core_dim
        self.log_std_bounds = log_std_bounds

        # Input: scan + core + AIG context
        input_dim = scan_dim + core_dim + aig_output_dim

        # Actor trunk
        self.trunk = mlp(input_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.apply(weight_init)

    def forward(
        self,
        scan: torch.Tensor,
        core: torch.Tensor,
        action_context: torch.Tensor,
    ) -> pyd.Distribution:
        """
        Forward pass.

        Args:
            scan: [batch, scan_dim] LiDAR scan
            core: [batch, core_dim] core features (distance, cos, sin)
            action_context: [batch, aig_output_dim] action-set AIG encoding

        Returns:
            action distribution
        """
        # Concatenate all features
        combined = torch.cat([scan, core, action_context], dim=-1)

        # Actor MLP
        mu, log_std = self.trunk(combined).chunk(2, dim=-1)

        # Constrain log_std
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        return SquashedNormal(mu, std)


# ============================================================================
# AIG Critic
# ============================================================================

class AIGCritic(nn.Module):
    """
    Double Q-Critic that uses AIG encoding.
    """

    def __init__(
        self,
        scan_dim: int,
        core_dim: int,
        action_dim: int,
        aig_output_dim: int = 128,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
    ):
        super().__init__()

        # Input: scan + core + AIG context + action
        input_dim = scan_dim + core_dim + aig_output_dim + action_dim

        # Two Q-networks
        self.Q1 = mlp(input_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(input_dim, hidden_dim, 1, hidden_depth)

        self.apply(weight_init)

    def forward(
        self,
        scan: torch.Tensor,
        core: torch.Tensor,
        aig_context: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            scan: [batch, scan_dim]
            core: [batch, core_dim]
            aig_context: [batch, aig_output_dim]
            action: [batch, action_dim]

        Returns:
            q1, q2: Q-values from both networks
        """
        combined = torch.cat([scan, core, aig_context, action], dim=-1)

        q1 = self.Q1(combined)
        q2 = self.Q2(combined)

        return q1, q2


# ============================================================================
# AIG-Nav Agent
# ============================================================================

class AIGNav:
    """
    Action-conditioned Interaction Graph for Social Navigation.

    Key idea:
    Uses AIG encoding which predicts future interactions based on the
    chosen action. The policy learns to choose actions that lead to
    favorable predicted interactions.

    This enables PROACTIVE navigation: the robot anticipates and avoids
    future collisions before they become imminent.
    """

    def __init__(
        self,
        scan_dim: int,
        core_dim: int = 3,
        action_dim: int = 2,
        device: str = "cuda",
        max_action: float = 1.0,
        # SAC hyperparameters
        discount: float = 0.99,
        init_temperature: float = 0.2,
        alpha_lr: float = 3e-5,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        encoder_lr: float = 3e-4,
        critic_tau: float = 0.005,
        actor_update_frequency: int = 1,
        critic_target_update_frequency: int = 2,
        learnable_temperature: bool = True,
        # AIG encoder parameters
        aig_hidden_dim: int = 128,
        aig_output_dim: int = 128,
        pred_horizon: int = 12,
        max_pedestrians: int = 10,
        history_length: int = 10,
        max_static_obstacles: int = 8,
        goal_distance_scale: float = 6.0,
        num_candidate_actions: int = 9,
        aux_pred_loss_weight: float = 0.5,
        action_conditioned: bool = True,
        reactive_model: bool = True,
        pred_edges: bool = True,
        use_ped_ped_edges: bool = True,
        edge_distance_threshold: float = 2.0,
        edge_distance_temperature: float = 0.5,
        # MLP parameters
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        # Save/load
        save_every: int = 2000,
        save_directory: Path = Path("models/AIGNav"),
        model_name: str = "AIGNav_v1",
        # === IAIG parameters ===
        use_iterative_eq: bool = True,
        K: int = 3,
        intention_dim: int = 64,
        lambda_conv: float = 0.1,
        # Robot physical parameters
        robot_radius: float = 0.25,
    ):
        self.robot_radius = robot_radius
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.action_range = (-max_action, max_action)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.learnable_temperature = learnable_temperature
        self.save_every = save_every
        self.save_directory = Path(save_directory)
        self.model_name = model_name

        self.scan_dim = scan_dim
        self.core_dim = core_dim
        self.action_dim = action_dim
        self.aig_output_dim = aig_output_dim
        self.max_pedestrians = max_pedestrians

        self.aux_pred_loss_weight = aux_pred_loss_weight
        self.action_conditioned = action_conditioned
        self.reactive_model = reactive_model
        self.pred_edges = pred_edges
        self.use_ped_ped_edges = use_ped_ped_edges

        # === AIG Encoder ===
        self.aig_encoder = AIGGraphEncoder(
            node_feat_dim=8,
            hidden_dim=aig_hidden_dim,
            output_dim=aig_output_dim,
            pred_horizon=pred_horizon,
            history_length=history_length,
            goal_distance_scale=goal_distance_scale,
            robot_radius=self.robot_radius,
            max_pedestrians=max_pedestrians,
            max_static_obstacles=max_static_obstacles,
            use_ped_ped_edges=use_ped_ped_edges,
            action_conditioned=action_conditioned,
            reactive_model=reactive_model,
            pred_edges=pred_edges,
            edge_distance_threshold=edge_distance_threshold,
            edge_distance_temperature=edge_distance_temperature,
        ).to(self.device)

        # Target AIG encoder
        self.aig_encoder_target = copy.deepcopy(self.aig_encoder).to(self.device)

        # === Action-set encoder / Iterative Equilibrium Module for actor context ===
        self.use_iterative_eq = use_iterative_eq

        if self.use_iterative_eq:
            self.iter_eq_module = IterativeEquilibriumModule(
                core_dim=core_dim,
                hidden_dim=aig_hidden_dim,
                output_dim=aig_output_dim,
                intention_dim=intention_dim,
                action_dim=action_dim,
                K=K,
                pred_horizon=pred_horizon,
                dt=0.1,
                history_length=history_length,
                max_pedestrians=max_pedestrians,
                robot_radius=self.robot_radius,
                ped_radius=0.3,
                safety_margin=0.2,
                goal_distance_scale=goal_distance_scale,
                lambda_conv=lambda_conv,
            ).to(self.device)
            self.action_set_encoder = None
        else:
            # === AIG-Nav (ours, predecessor design with discrete ActionSet) ===
            self.action_set_encoder = ActionSetAggregator(
                output_dim=aig_output_dim,
                num_candidate_actions=num_candidate_actions,
                action_hidden_dim=max(32, aig_hidden_dim // 2),
            ).to(self.device)
            self.iter_eq_module = None

        # === Actor ===
        self.actor = AIGActor(
            scan_dim=scan_dim,
            core_dim=core_dim,
            action_dim=action_dim,
            aig_output_dim=aig_output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)

        # === Critic ===
        self.critic = AIGCritic(
            scan_dim=scan_dim,
            core_dim=core_dim,
            action_dim=action_dim,
            aig_output_dim=aig_output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)

        # === Critic Target ===
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # === Inference copies (for thread safety) ===
        self._infer_lock = threading.Lock()
        self.aig_encoder_infer = copy.deepcopy(self.aig_encoder).to(self.device)
        if self.use_iterative_eq:
            self.iter_eq_module_infer = copy.deepcopy(self.iter_eq_module).to(self.device)
            self.iter_eq_module_infer.eval()
            self.action_set_encoder_infer = None
        else:
            self.action_set_encoder_infer = copy.deepcopy(self.action_set_encoder).to(self.device)
            self.action_set_encoder_infer.eval()
            self.iter_eq_module_infer = None
        self.actor_infer = copy.deepcopy(self.actor).to(self.device)
        self.aig_encoder_infer.eval()
        self.actor_infer.eval()

        # === Temperature ===
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = 0
        self.alpha_min = 0.05
        self.alpha_max = 1.0

        # === Optimizers ===
        # Critic optimizer includes AIG encoder
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.aig_encoder.parameters()),
            lr=critic_lr
        )
        actor_params = list(self.actor.parameters())
        if self.use_iterative_eq:
            actor_params += list(self.iter_eq_module.parameters())
        else:
            actor_params += list(self.action_set_encoder.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # === AMP (mixed precision) for faster training ===
        self._use_amp = device.startswith("cuda") if isinstance(device, str) else (device.type == "cuda")
        self._amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        # === Training state ===
        self.step = 0
        self.writer = NoOpSummaryWriter()
        self.actor_delay_steps = 1000  # Let critic stabilize ~20 training calls before actor updates

        self.train_metrics = {
            "critic_loss": [],
            "actor_loss": [],
            "entropy": [],
            "alpha": [],
            "pred_loss": [],
            "conv_loss": [],
        }

        self.latest_critic_loss = 0.0
        self.latest_entropy = 0.0

        # For inference state
        self._last_action = np.zeros(action_dim, dtype=np.float32)

        method_tag = "IAIG-Nav" if self.use_iterative_eq else "AIG-Nav"
        print(f"[{method_tag}] Initialized on {self.device}")
        print(f"[{method_tag}] AIG Encoder params: {sum(p.numel() for p in self.aig_encoder.parameters()):,}")
        if self.use_iterative_eq:
            print(f"[{method_tag}] Iter-Eq module params: {sum(p.numel() for p in self.iter_eq_module.parameters()):,}")
            print(f"[{method_tag}] K={K}, intention_dim={intention_dim}, lambda_conv={lambda_conv}")
        else:
            print(f"[{method_tag}] Action-set encoder params: {sum(p.numel() for p in self.action_set_encoder.parameters()):,}")
        print(f"[{method_tag}] Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"[{method_tag}] Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(
            f"[{method_tag}] Ablation flags: "
            f"action_conditioned={self.action_conditioned} "
            f"reactive_model={self.reactive_model} "
            f"pred_edges={self.pred_edges} "
            f"use_ped_ped_edges={self.use_ped_ped_edges}"
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sync_inference(self):
        """Sync training weights to inference copies."""
        with self._infer_lock:
            self.aig_encoder_infer.load_state_dict(self.aig_encoder.state_dict())
            if self.use_iterative_eq:
                self.iter_eq_module_infer.load_state_dict(self.iter_eq_module.state_dict())
                self.iter_eq_module_infer.eval()
            else:
                self.action_set_encoder_infer.load_state_dict(self.action_set_encoder.state_dict())
                self.action_set_encoder_infer.eval()
            self.actor_infer.load_state_dict(self.actor.state_dict())
            self.aig_encoder_infer.eval()
            self.actor_infer.eval()

    def act(
        self,
        scan: np.ndarray,
        core: np.ndarray,
        ped_positions: np.ndarray,
        ped_velocities: np.ndarray,
        ped_histories: Optional[np.ndarray] = None,
        static_obstacles: Optional[np.ndarray] = None,
        sample: bool = True,
    ) -> np.ndarray:
        """
        Select action given observation.

        Args:
            scan: [scan_dim] LiDAR scan
            core: [core_dim] core features
            ped_positions: [num_peds, 2] pedestrian positions
            ped_velocities: [num_peds, 2] pedestrian velocities
            ped_histories: [num_peds, T, 4] optional pedestrian histories
            static_obstacles: [num_static, 3] optional static obstacles
            sample: if True, sample from distribution; else use mean

        Returns:
            action: [action_dim] selected action
        """
        with torch.no_grad():
            scan_t = torch.FloatTensor(scan).unsqueeze(0).to(self.device)
            core_t = torch.FloatTensor(core).unsqueeze(0).to(self.device)

            # Handle pedestrian data
            if ped_positions is not None and len(ped_positions) > 0:
                ped_pos_t = torch.as_tensor(ped_positions, dtype=torch.float32, device=self.device).unsqueeze(0)
                ped_vel_t = torch.as_tensor(ped_velocities, dtype=torch.float32, device=self.device).unsqueeze(0)
                ped_mask_t = torch.ones(1, ped_pos_t.size(1), dtype=torch.bool, device=self.device)
                if ped_histories is not None and len(ped_histories) > 0:
                    ped_hist_t = torch.as_tensor(ped_histories, dtype=torch.float32, device=self.device).unsqueeze(0)
                else:
                    ped_hist_t = None
            else:
                ped_pos_t = torch.zeros(1, 0, 2, device=self.device)
                ped_vel_t = torch.zeros(1, 0, 2, device=self.device)
                ped_hist_t = None
                ped_mask_t = torch.zeros(1, 0, dtype=torch.bool, device=self.device)

            if static_obstacles is not None and len(static_obstacles) > 0:
                static_obs_t = torch.as_tensor(static_obstacles, dtype=torch.float32, device=self.device).unsqueeze(0)
                static_mask_t = torch.ones(1, static_obs_t.size(1), dtype=torch.bool, device=self.device)
            else:
                static_obs_t = torch.zeros(1, 0, 3, device=self.device)
                static_mask_t = torch.zeros(1, 0, dtype=torch.bool, device=self.device)

            with self._infer_lock, torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                if self.use_iterative_eq:
                    action_context, _, _ = self.iter_eq_module_infer(
                        core_t,
                        ped_pos_t,
                        ped_vel_t,
                        ped_histories=ped_hist_t,
                        ped_mask=ped_mask_t,
                        static_obstacles=static_obs_t,
                        static_mask=static_mask_t,
                        detach_encoder=False,
                    )
                else:
                    action_context, _, _ = self.action_set_encoder_infer(
                        self.aig_encoder_infer,
                        core_t,
                        ped_pos_t,
                        ped_vel_t,
                        ped_histories=ped_hist_t,
                        ped_mask=ped_mask_t,
                        static_obstacles=static_obs_t,
                        static_mask=static_mask_t,
                        detach_encoder=False,
                    )

                dist = self.actor_infer(scan_t, core_t, action_context)
                action = dist.sample() if sample else dist.mean
                action = action.clamp(*self.action_range)

                action_np = action[0].float().cpu().numpy()
                self._last_action = action_np
                return action_np

    def reset_episode(self):
        """Reset episode state."""
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    def update(self, batch: Dict[str, Any]):
        """
        Update actor and critic from a batch of transitions.

        Args:
            batch: dictionary containing:
                - scan, core, action, reward, next_scan, next_core, done
                - ped_positions, ped_velocities, ped_histories, ped_masks
                - static_obstacles, static_masks
                - next_ped_positions, next_ped_velocities, next_ped_histories, next_ped_masks
                - next_static_obstacles, next_static_masks
        """
        self.step += 1

        scan = batch["scan"].to(self.device)
        core = batch["core"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_scan = batch["next_scan"].to(self.device)
        next_core = batch["next_core"].to(self.device)
        done = batch["done"].to(self.device)

        ped_positions = batch["ped_positions"].to(self.device)
        ped_velocities = batch["ped_velocities"].to(self.device)
        ped_histories = batch["ped_histories"].to(self.device)
        ped_masks = batch["ped_masks"].to(self.device)
        next_ped_positions = batch["next_ped_positions"].to(self.device)
        next_ped_velocities = batch["next_ped_velocities"].to(self.device)
        next_ped_histories = batch["next_ped_histories"].to(self.device)
        next_ped_masks = batch["next_ped_masks"].to(self.device)

        static_obstacles = batch["static_obstacles"].to(self.device)
        static_masks = batch["static_masks"].to(self.device)
        next_static_obstacles = batch["next_static_obstacles"].to(self.device)
        next_static_masks = batch["next_static_masks"].to(self.device)

        # World coordinates and robot pose for aux_pred_loss supervision
        ped_positions_world = batch["ped_positions_world"].to(self.device)
        next_ped_positions_world = batch["next_ped_positions_world"].to(self.device)
        robot_pose = batch["robot_pose"].to(self.device)  # [B, 3] (x, y, yaw)

        amp_ctx = torch.amp.autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype)

        # === Get AIG encodings ===
        with amp_ctx:
            # Current state encoding (conditioned on taken action)
            aig_context = self.aig_encoder(
                core,
                action,
                ped_positions,
                ped_velocities,
                ped_histories=ped_histories,
                ped_mask=ped_masks,
                static_obstacles=static_obstacles,
                static_mask=static_masks,
            )

        # Next state encoding (need to sample next action first)
        with torch.no_grad(), amp_ctx:
            if self.use_iterative_eq:
                next_action_context, _, _ = self.iter_eq_module(
                    next_core,
                    next_ped_positions,
                    next_ped_velocities,
                    ped_histories=next_ped_histories,
                    ped_mask=next_ped_masks,
                    static_obstacles=next_static_obstacles,
                    static_mask=next_static_masks,
                    detach_encoder=True,
                )
            else:
                next_action_context, _, _ = self.action_set_encoder(
                    self.aig_encoder,
                    next_core,
                    next_ped_positions,
                    next_ped_velocities,
                    ped_histories=next_ped_histories,
                    ped_mask=next_ped_masks,
                    static_obstacles=next_static_obstacles,
                    static_mask=next_static_masks,
                    detach_encoder=True,
                )

            # Get next action distribution
            next_dist = self.actor(next_scan, next_core, next_action_context)
            next_action = next_dist.rsample()
            log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

            next_aig_context_target = self.aig_encoder_target(
                next_core,
                next_action,
                next_ped_positions,
                next_ped_velocities,
                ped_histories=next_ped_histories,
                ped_mask=next_ped_masks,
                static_obstacles=next_static_obstacles,
                static_mask=next_static_masks,
            )

            # Target Q
            target_Q1, target_Q2 = self.critic_target(
                next_scan, next_core, next_aig_context_target, next_action
            )
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.discount * target_V
            target_Q = torch.clamp(target_Q, -100.0, 100.0)

        # === Update Critic ===
        with amp_ctx:
            current_Q1, current_Q2 = self.critic(scan, core, aig_context, action)
            critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + F.smooth_l1_loss(current_Q2, target_Q)

            pred_loss = None
            if self.aux_pred_loss_weight > 0 and ped_positions.size(1) > 0:
                # The predictor outputs future positions in robot frame at time t.
                # We need ground truth also in robot frame at time t.
                # Convert next_ped_positions_world to robot frame at time t using robot_pose.
                robot_x = robot_pose[:, 0:1].unsqueeze(1)   # [B, 1, 1]
                robot_y = robot_pose[:, 1:2].unsqueeze(1)   # [B, 1, 1]
                robot_yaw = robot_pose[:, 2]                 # [B]
                cos_neg_yaw = torch.cos(-robot_yaw).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                sin_neg_yaw = torch.sin(-robot_yaw).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

                # Transform next_ped_positions_world to robot frame at time t
                dx = next_ped_positions_world[:, :, 0:1] - robot_x  # [B, N, 1]
                dy = next_ped_positions_world[:, :, 1:2] - robot_y  # [B, N, 1]
                actual_next_in_t_x = cos_neg_yaw * dx - sin_neg_yaw * dy  # [B, N, 1]
                actual_next_in_t_y = sin_neg_yaw * dx + cos_neg_yaw * dy  # [B, N, 1]
                actual_next_in_t = torch.cat([actual_next_in_t_x, actual_next_in_t_y], dim=-1)  # [B, N, 2]

                pred_trajs = self.aig_encoder.predict_ped_trajectories(
                    ped_positions,
                    ped_velocities,
                    ped_histories,
                    action,
                    ped_mask=ped_masks,
                )
                if pred_trajs.numel() > 0:
                    pred_next_robot = pred_trajs[:, :, 0, :]  # [B, N, 2]

                    diff = pred_next_robot - actual_next_in_t
                    mask = ped_masks.unsqueeze(-1).float() if ped_masks is not None else None
                    if mask is not None and mask.numel() > 0:
                        diff = diff * mask
                        denom = mask.sum().clamp(min=1.0)
                        pred_loss = (diff ** 2).sum() / denom
                    else:
                        pred_loss = (diff ** 2).mean()

                    critic_loss = critic_loss + self.aux_pred_loss_weight * pred_loss

        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.aig_encoder.parameters()), 1.0
        )
        self.scaler.step(self.critic_optimizer)

        self.train_metrics["critic_loss"].append(critic_loss.item())
        self.latest_critic_loss = critic_loss.item()
        if pred_loss is not None:
            self.train_metrics["pred_loss"].append(pred_loss.item())

        # === Update Actor ===
        if self.step >= self.actor_delay_steps and self.step % self.actor_update_frequency == 0:
            with amp_ctx:
                if self.use_iterative_eq:
                    action_context, _, _ = self.iter_eq_module(
                        core,
                        ped_positions,
                        ped_velocities,
                        ped_histories=ped_histories,
                        ped_mask=ped_masks,
                        static_obstacles=static_obstacles,
                        static_mask=static_masks,
                        detach_encoder=True,
                    )
                else:
                    action_context, _, _ = self.action_set_encoder(
                        self.aig_encoder,
                        core,
                        ped_positions,
                        ped_velocities,
                        ped_histories=ped_histories,
                        ped_mask=ped_masks,
                        static_obstacles=static_obstacles,
                        static_mask=static_masks,
                        detach_encoder=True,
                    )

                dist = self.actor(scan, core, action_context)
                new_action = dist.rsample()
                log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

                aig_context_new = self.aig_encoder(
                    core,
                    new_action,
                    ped_positions,
                    ped_velocities,
                    ped_histories=ped_histories,
                    ped_mask=ped_masks,
                    static_obstacles=static_obstacles,
                    static_mask=static_masks,
                )
                Q1, Q2 = self.critic(scan, core, aig_context_new, new_action)
                Q = torch.min(Q1, Q2)

                actor_loss = (self.alpha.detach() * log_prob - Q).mean()

                if self.use_iterative_eq:
                    conv_loss = self.iter_eq_module.get_convergence_loss()
                    actor_loss = actor_loss + conv_loss

            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            if self.use_iterative_eq:
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.iter_eq_module.parameters()), 1.0
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.action_set_encoder.parameters()), 1.0
                )
            self.scaler.step(self.actor_optimizer)

            self.train_metrics["actor_loss"].append(actor_loss.item())
            if self.use_iterative_eq and conv_loss.item() > 0:
                self.train_metrics["conv_loss"].append(conv_loss.item())
            self.train_metrics["entropy"].append(-log_prob.mean().item())
            self.latest_entropy = -log_prob.mean().item()

            if self.latest_entropy < -0.5 and self.step % 5000 == 0:
                print(f"[WARN] Low entropy: {self.latest_entropy:.3f}, alpha: {self.alpha.item():.4f}")

            # === Update Temperature ===
            if self.learnable_temperature:
                alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()

                with torch.no_grad():
                    self.log_alpha.clamp_(
                        min=np.log(self.alpha_min),
                        max=np.log(self.alpha_max)
                    )

                self.train_metrics["alpha"].append(self.alpha.item())

        # Update AMP scaler (must be called once per update, after all optimizer steps)
        self.scaler.update()

        # === Update Target Networks ===
        if self.step % self.critic_target_update_frequency == 0:
            self._soft_update(self.critic, self.critic_target, self.critic_tau)
            self._soft_update(self.aig_encoder, self.aig_encoder_target, self.critic_tau)

    def _soft_update(self, source, target, tau):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, replay_buffer, iterations: int, batch_size: int):
        """Training loop."""
        for _ in range(iterations):
            batch = replay_buffer.sample(batch_size)
            if batch is not None:
                self.update(batch)

        self.sync_inference()

        # Log metrics
        for key, values in self.train_metrics.items():
            if values:
                avg = mean(values)
                self.writer.add_scalar(f"train/{key}", avg, self.step)
            self.train_metrics[key] = []

        # Save periodically
        if self.save_every > 0 and self.step % self.save_every == 0:
            self.save()

    def save(self, directory: Path = None):
        """Save model weights."""
        d = Path(directory) if directory is not None else self.save_directory
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), d / f"{self.model_name}_actor.pth")
        torch.save(self.critic.state_dict(), d / f"{self.model_name}_critic.pth")
        torch.save(self.aig_encoder.state_dict(), d / f"{self.model_name}_aig_encoder.pth")
        if self.use_iterative_eq:
            torch.save(self.iter_eq_module.state_dict(), d / f"{self.model_name}_iter_eq.pth")
        else:
            torch.save(self.action_set_encoder.state_dict(), d / f"{self.model_name}_action_set.pth")
        method_tag = "IAIG-Nav" if self.use_iterative_eq else "AIG-Nav"
        print(f"[{method_tag}] Saved to {d}")

    def load(self, directory: Path = None):
        """Load model weights."""
        if directory is None:
            directory = self.save_directory
        self.actor.load_state_dict(torch.load(directory / f"{self.model_name}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(directory / f"{self.model_name}_critic.pth", map_location=self.device))
        self.aig_encoder.load_state_dict(torch.load(directory / f"{self.model_name}_aig_encoder.pth", map_location=self.device))
        if self.use_iterative_eq:
            path = directory / f"{self.model_name}_iter_eq.pth"
            if path.exists():
                self.iter_eq_module.load_state_dict(torch.load(path, map_location=self.device))
        else:
            path = directory / f"{self.model_name}_action_set.pth"
            if path.exists():
                self.action_set_encoder.load_state_dict(torch.load(path, map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.aig_encoder_target.load_state_dict(self.aig_encoder.state_dict())
        self.sync_inference()
        method_tag = "IAIG-Nav" if self.use_iterative_eq else "AIG-Nav"
        print(f"[{method_tag}] Loaded from {directory}")


if __name__ == "__main__":
    # Test AIG-Nav
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing AIG-Nav on device: {device}")

    agent = AIGNav(
        scan_dim=32,
        core_dim=3,
        action_dim=2,
        device=device,
    )

    # Test act
    scan = np.random.randn(32).astype(np.float32)
    core = np.random.randn(3).astype(np.float32)
    ped_pos = np.random.randn(3, 2).astype(np.float32)
    ped_vel = np.random.randn(3, 2).astype(np.float32) * 0.5
    ped_hist = np.random.randn(3, 10, 4).astype(np.float32)
    static_obs = np.random.randn(2, 3).astype(np.float32)

    action = agent.act(scan, core, ped_pos, ped_vel, ped_histories=ped_hist, static_obstacles=static_obs, sample=True)
    print(f"Action shape: {action.shape}")
    assert action.shape == (2,)

    # Test with no pedestrians
    action = agent.act(scan, core, np.array([]), np.array([]), ped_histories=None, static_obstacles=None, sample=True)
    print(f"Action (no peds) shape: {action.shape}")

    print("\nAIG-Nav tests passed!")
