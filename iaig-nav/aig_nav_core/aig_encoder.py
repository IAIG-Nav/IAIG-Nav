"""
Action-conditioned Interaction Graph (AIG) Encoder.

This module implements:
1. Interaction-predictive edges based on future collision probability.
2. Action-conditioned graphs (G_a for each candidate action).
3. Reactive trajectory prediction (pedestrian response to robot action).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

from .trajectory_predictor import (
    MultiAgentTrajectoryPredictor,
    SimpleTrajectoryPredictor,
    compute_collision_probability,
    compute_pairwise_collision_probability,
    compute_static_collision_probability,
    predict_robot_trajectory,
)


class AIGGraphEncoder(nn.Module):
    """
    Action-conditioned predictive interaction graph encoder.

    Builds a predictive interaction graph G_a for a given action and encodes it
    with GATv2. Edge weights are future collision probabilities.
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        pred_horizon: int = 12,
        dt: float = 0.1,
        history_length: int = 10,
        goal_distance_scale: float = 6.0,
        robot_radius: float = 0.25,
        ped_radius: float = 0.3,
        safety_margin: float = 0.2,
        max_pedestrians: int = 10,
        max_static_obstacles: int = 8,
        use_ped_ped_edges: bool = True,
        action_conditioned: bool = True,
        reactive_model: bool = True,
        pred_edges: bool = True,
        edge_distance_threshold: float = 2.0,
        edge_distance_temperature: float = 0.5,
        use_simple_predictor: bool = False,
    ):
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pred_horizon = pred_horizon
        self.dt = dt
        self.history_length = history_length
        self.goal_distance_scale = goal_distance_scale
        self.robot_radius = robot_radius
        self.ped_radius = ped_radius
        self.safety_margin = safety_margin
        self.max_pedestrians = max_pedestrians
        self.max_static_obstacles = max_static_obstacles
        self.use_ped_ped_edges = use_ped_ped_edges
        self.action_conditioned = action_conditioned
        self.reactive_model = reactive_model
        self.pred_edges = pred_edges
        self.edge_distance_threshold = edge_distance_threshold
        self.edge_distance_temperature = edge_distance_temperature
        self.use_simple_predictor = use_simple_predictor

        if use_simple_predictor:
            self.traj_predictor = SimpleTrajectoryPredictor(
                pred_horizon=pred_horizon,
                hidden_dim=32,
                dt=dt,
            )
        else:
            self.traj_predictor = MultiAgentTrajectoryPredictor(
                history_len=history_length,
                pred_horizon=pred_horizon,
                hidden_dim=64,
            )

        # Node feature projection
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GATv2 layers
        self.gat1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=output_dim,
            heads=1,
            concat=True,
            dropout=dropout,
            edge_dim=1,
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _compute_goal_position(self, core: torch.Tensor) -> torch.Tensor:
        distance = core[:, 0] * self.goal_distance_scale
        goal_x = distance * core[:, 1]
        goal_y = distance * core[:, 2]
        return torch.stack([goal_x, goal_y], dim=-1)

    def _ensure_histories(
        self,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if ped_histories is None or ped_histories.numel() == 0:
            base = torch.cat([ped_positions, ped_velocities], dim=-1)  # [B, N, 4]
            return base.unsqueeze(2).repeat(1, 1, self.history_length, 1)

        if ped_histories.size(2) == self.history_length:
            return ped_histories

        if ped_histories.size(2) > self.history_length:
            return ped_histories[:, :, -self.history_length :, :]

        # Pad to history_length
        pad_len = self.history_length - ped_histories.size(2)
        pad_frame = ped_histories[:, :, :1, :].repeat(1, 1, pad_len, 1)
        return torch.cat([pad_frame, ped_histories], dim=2)

    def predict_ped_trajectories(
        self,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor],
        robot_action: torch.Tensor,
        ped_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ped_positions.numel() == 0:
            return torch.zeros(
                ped_positions.size(0), 0, self.pred_horizon, 2, device=ped_positions.device
            )

        if not self.reactive_model:
            robot_action = torch.zeros_like(robot_action)

        if self.use_simple_predictor:
            robot_pos = torch.zeros(ped_positions.size(0), 2, device=ped_positions.device)
            return self.traj_predictor(ped_positions, ped_velocities, robot_action, robot_pos)

        histories = self._ensure_histories(ped_positions, ped_velocities, ped_histories)
        return self.traj_predictor(histories, robot_action, ped_mask=ped_mask)

    def _distance_edge_weight(self, distance: torch.Tensor, threshold: float) -> torch.Tensor:
        return torch.sigmoid((threshold - distance) / self.edge_distance_temperature)

    def _build_graphs(
        self,
        core: torch.Tensor,
        robot_action: torch.Tensor,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor],
        ped_mask: Optional[torch.Tensor],
        static_obstacles: Optional[torch.Tensor],
        static_mask: Optional[torch.Tensor],
        goal_position: Optional[torch.Tensor],
    ) -> Batch:
        device = core.device
        batch_size = core.size(0)

        if not self.action_conditioned:
            robot_action = torch.zeros_like(robot_action)

        if goal_position is None:
            goal_position = self._compute_goal_position(core)

        # Normalize empty inputs
        if ped_positions is None or ped_positions.numel() == 0:
            ped_positions = torch.zeros(batch_size, 0, 2, device=device)
            ped_velocities = torch.zeros(batch_size, 0, 2, device=device)
            ped_histories = torch.zeros(batch_size, 0, self.history_length, 4, device=device)
            ped_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        if static_obstacles is None or static_obstacles.numel() == 0:
            static_obstacles = torch.zeros(batch_size, 0, 3, device=device)
            static_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

        # Predict robot trajectory (robot frame)
        robot_position = torch.zeros(batch_size, 2, device=device)
        robot_yaw = torch.zeros(batch_size, device=device)
        robot_traj = predict_robot_trajectory(
            robot_position,
            robot_yaw,
            robot_action,
            pred_horizon=self.pred_horizon,
            dt=self.dt,
        )

        # Predict pedestrian trajectories (reactive)
        ped_trajs = self.predict_ped_trajectories(
            ped_positions,
            ped_velocities,
            ped_histories,
            robot_action,
            ped_mask=ped_mask,
        )

        collision_probs = None
        static_probs = None
        ped_ped_probs = None
        if self.pred_edges:
            collision_probs = compute_collision_probability(
                robot_traj,
                ped_trajs,
                robot_radius=self.robot_radius,
                ped_radius=self.ped_radius,
                safety_margin=self.safety_margin,
                ped_mask=ped_mask,
            )

            static_probs = compute_static_collision_probability(
                robot_traj,
                static_obstacles,
                robot_radius=self.robot_radius,
                safety_margin=self.safety_margin,
                static_mask=static_mask,
            )

            if self.use_ped_ped_edges and ped_trajs.size(1) > 1:
                ped_ped_probs = compute_pairwise_collision_probability(
                    ped_trajs,
                    ped_radius=self.ped_radius,
                    safety_margin=self.safety_margin,
                    ped_mask=ped_mask,
                )

        # Fully vectorized graph construction — no Python loop over batch items.
        # Uses padded fixed-size nodes with only valid edges (via boolean masking).
        # Invalid nodes have zero features and NO edges, so GATv2 output is identical
        # to the original per-item construction.

        max_peds = ped_positions.size(1)
        max_static = static_obstacles.size(1)
        s_start = 2 + max_peds  # static node offset within each graph
        max_nodes = 2 + max_peds + max_static

        # Ensure boolean masks
        if ped_mask is None or ped_mask.numel() == 0:
            ped_mask = torch.ones(batch_size, max_peds, dtype=torch.bool, device=device) if max_peds > 0 else torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        if static_mask is None or static_mask.numel() == 0:
            static_mask = torch.ones(batch_size, max_static, dtype=torch.bool, device=device) if max_static > 0 else torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

        # === Build ALL node features as [B, max_nodes, 8] ===
        node_feats = torch.zeros(batch_size, max_nodes, self.node_feat_dim, device=device)

        # Robot node (index 0): [0, 0, pred_x, pred_y, vx, 0, type=0.0, conf=1.0]
        node_feats[:, 0, 2:4] = robot_traj[:, -1]
        node_feats[:, 0, 4] = robot_action[:, 0]
        node_feats[:, 0, 7] = 1.0

        # Goal node (index 1): [gx, gy, gx, gy, 0, 0, type=1.0, conf=1.0]
        node_feats[:, 1, :2] = goal_position
        node_feats[:, 1, 2:4] = goal_position
        node_feats[:, 1, 6] = 1.0
        node_feats[:, 1, 7] = 1.0

        # Pedestrian nodes (indices 2..2+max_peds-1), masked
        if max_peds > 0:
            pmf = ped_mask.unsqueeze(-1).float()  # [B, N, 1]
            node_feats[:, 2:2+max_peds, :2] = ped_positions * pmf
            node_feats[:, 2:2+max_peds, 2:4] = ped_trajs[:, :, -1] * pmf
            node_feats[:, 2:2+max_peds, 4:6] = ped_velocities * pmf
            node_feats[:, 2:2+max_peds, 6] = 0.5 * ped_mask.float()
            node_feats[:, 2:2+max_peds, 7] = ped_mask.float()

        # Static obstacle nodes (indices s_start..s_start+max_static-1), masked
        if max_static > 0:
            smf = static_mask.unsqueeze(-1).float()  # [B, M, 1]
            sp = static_obstacles[:, :, :2]
            node_feats[:, s_start:s_start+max_static, :2] = sp * smf
            node_feats[:, s_start:s_start+max_static, 2:4] = sp * smf
            node_feats[:, s_start:s_start+max_static, 6] = 0.25 * static_mask.float()
            node_feats[:, s_start:s_start+max_static, 7] = static_mask.float()

        # Flatten: [B * max_nodes, 8]
        x = node_feats.reshape(-1, self.node_feat_dim)

        # Batch assignment vector
        batch_vec = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_nodes).reshape(-1)

        # === Build ALL edges via vectorized boolean masking ===
        # Node offset for each graph in the flattened batch
        offsets = (torch.arange(batch_size, device=device) * max_nodes).unsqueeze(1)  # [B, 1]

        all_src = []
        all_dst = []
        all_w = []

        # 1) Robot-Goal edges (always present, all batch items)
        rg_robot = offsets.squeeze(1)      # [B]  node 0
        rg_goal = offsets.squeeze(1) + 1   # [B]  node 1
        all_src.append(torch.cat([rg_robot, rg_goal]))
        all_dst.append(torch.cat([rg_goal, rg_robot]))
        all_w.append(torch.ones(2 * batch_size, device=device))

        # 2) Robot-Pedestrian edges (only valid peds)
        if max_peds > 0 and ped_mask.any():
            rp_robot_all = offsets.expand(-1, max_peds)            # [B, N]
            rp_ped_all = offsets + 2 + torch.arange(max_peds, device=device)  # [B, N]

            rp_valid = ped_mask  # [B, N]
            rp_r = rp_robot_all[rp_valid]
            rp_p = rp_ped_all[rp_valid]

            if self.pred_edges and collision_probs is not None:
                rp_w = collision_probs[rp_valid]
            else:
                ped_dists = torch.norm(ped_positions, p=2, dim=-1)  # [B, N]
                rp_w = self._distance_edge_weight(ped_dists, self.edge_distance_threshold)[rp_valid]

            all_src.append(torch.cat([rp_r, rp_p]))
            all_dst.append(torch.cat([rp_p, rp_r]))
            all_w.append(rp_w.repeat(2))

        # 3) Robot-Static edges (only valid statics)
        if max_static > 0 and static_mask.any():
            rs_robot_all = offsets.expand(-1, max_static)                              # [B, M]
            rs_static_all = offsets + s_start + torch.arange(max_static, device=device)  # [B, M]

            rs_valid = static_mask  # [B, M]
            rs_r = rs_robot_all[rs_valid]
            rs_s = rs_static_all[rs_valid]

            if self.pred_edges and static_probs is not None:
                rs_w = static_probs[rs_valid]
            else:
                s_dists = torch.norm(static_obstacles[:, :, :2], p=2, dim=-1)  # [B, M]
                s_thresh = self.edge_distance_threshold + static_obstacles[:, :, 2]
                rs_w = self._distance_edge_weight(s_dists, s_thresh)[rs_valid]

            all_src.append(torch.cat([rs_r, rs_s]))
            all_dst.append(torch.cat([rs_s, rs_r]))
            all_w.append(rs_w.repeat(2))

        # 4) Ped-Ped edges (only valid pairs)
        if self.use_ped_ped_edges and max_peds > 1:
            idx_i, idx_j = torch.triu_indices(max_peds, max_peds, offset=1, device=device)
            pair_valid = ped_mask[:, idx_i] & ped_mask[:, idx_j]  # [B, n_pairs]

            if pair_valid.any():
                pp_src_all = offsets + 2 + idx_i  # [B, n_pairs]
                pp_dst_all = offsets + 2 + idx_j  # [B, n_pairs]

                pp_s = pp_src_all[pair_valid]
                pp_d = pp_dst_all[pair_valid]

                if self.pred_edges and ped_ped_probs is not None:
                    pp_w = ped_ped_probs[:, idx_i, idx_j][pair_valid]
                else:
                    pair_dists = torch.norm(
                        ped_positions[:, idx_i] - ped_positions[:, idx_j], p=2, dim=-1
                    )  # [B, n_pairs]
                    pp_w = self._distance_edge_weight(pair_dists, self.edge_distance_threshold)[pair_valid]

                all_src.append(torch.cat([pp_s, pp_d]))
                all_dst.append(torch.cat([pp_d, pp_s]))
                all_w.append(pp_w.repeat(2))

        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
        edge_attr = torch.cat(all_w).unsqueeze(-1)

        # Construct Batch directly (avoids Data object + from_data_list overhead)
        result = Batch()
        result.x = x
        result.edge_index = edge_index
        result.edge_attr = edge_attr
        result.batch = batch_vec
        return result

    def forward(
        self,
        core: torch.Tensor,
        robot_action: torch.Tensor,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor] = None,
        ped_mask: Optional[torch.Tensor] = None,
        static_obstacles: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        goal_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = self._build_graphs(
            core,
            robot_action,
            ped_positions,
            ped_velocities,
            ped_histories,
            ped_mask,
            static_obstacles,
            static_mask,
            goal_position,
        )

        # Force float32 for GATv2 attention to avoid float16 overflow in softmax,
        # especially with large batches from ActionSetAggregator (batch * 9 actions).
        device_type = "cuda" if batch.x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            h = self.node_proj(batch.x.float())
            edge_attr = batch.edge_attr.float()
            h = self.gat1(h, batch.edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = self.gat2(h, batch.edge_index, edge_attr=edge_attr)

        # Robot node is first in each graph
        B = int(batch.batch.max().item()) + 1
        counts = torch.bincount(batch.batch, minlength=B)
        offsets = torch.cumsum(counts, dim=0) - counts
        robot_enc = h[offsets]

        context = self.layer_norm(robot_enc)
        return self.output_proj(context)

    def encode_actions(
        self,
        core: torch.Tensor,
        candidate_actions: torch.Tensor,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor] = None,
        ped_mask: Optional[torch.Tensor] = None,
        static_obstacles: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        goal_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode predictive graphs for all candidate actions.

        Returns:
            action_encodings: [B, A, output_dim]
        """
        device = core.device
        batch_size = core.size(0)
        num_actions = candidate_actions.size(0)

        actions_exp = candidate_actions.unsqueeze(0).expand(batch_size, -1, -1)
        actions_flat = actions_exp.reshape(batch_size * num_actions, -1)

        core_flat = core.unsqueeze(1).expand(batch_size, num_actions, -1).reshape(batch_size * num_actions, -1)
        ped_positions_flat = ped_positions.unsqueeze(1).expand(batch_size, num_actions, -1, -1).reshape(
            batch_size * num_actions, ped_positions.size(1), ped_positions.size(2)
        )
        ped_velocities_flat = ped_velocities.unsqueeze(1).expand(batch_size, num_actions, -1, -1).reshape(
            batch_size * num_actions, ped_velocities.size(1), ped_velocities.size(2)
        )

        if ped_histories is not None:
            ped_histories_flat = ped_histories.unsqueeze(1).expand(
                batch_size, num_actions, -1, -1, -1
            ).reshape(
                batch_size * num_actions,
                ped_histories.size(1),
                ped_histories.size(2),
                ped_histories.size(3),
            )
        else:
            ped_histories_flat = None

        if ped_mask is not None:
            ped_mask_flat = ped_mask.unsqueeze(1).expand(batch_size, num_actions, -1).reshape(
                batch_size * num_actions, -1
            )
        else:
            ped_mask_flat = None

        if static_obstacles is not None:
            static_obstacles_flat = static_obstacles.unsqueeze(1).expand(
                batch_size, num_actions, -1, -1
            ).reshape(batch_size * num_actions, static_obstacles.size(1), static_obstacles.size(2))
        else:
            static_obstacles_flat = None

        if static_mask is not None:
            static_mask_flat = static_mask.unsqueeze(1).expand(batch_size, num_actions, -1).reshape(
                batch_size * num_actions, -1
            )
        else:
            static_mask_flat = None

        if goal_position is not None:
            goal_pos_flat = goal_position.unsqueeze(1).expand(batch_size, num_actions, -1).reshape(
                batch_size * num_actions, -1
            )
        else:
            goal_pos_flat = None

        encodings_flat = self.forward(
            core_flat,
            actions_flat,
            ped_positions_flat,
            ped_velocities_flat,
            ped_histories=ped_histories_flat,
            ped_mask=ped_mask_flat,
            static_obstacles=static_obstacles_flat,
            static_mask=static_mask_flat,
            goal_position=goal_pos_flat,
        )

        return encodings_flat.reshape(batch_size, num_actions, -1)


class ActionSetAggregator(nn.Module):
    """
    Aggregate action-conditioned graph encodings into a single context.
    """

    def __init__(
        self,
        output_dim: int = 128,
        num_candidate_actions: int = 9,
        action_hidden_dim: int = 64,
    ):
        super().__init__()
        self.output_dim = output_dim

        lin_vels = torch.tensor([-0.2, 0.0, 0.3, 0.6])
        ang_vels = torch.tensor([-0.5, 0.0, 0.5])
        actions = []
        for lv in lin_vels:
            for av in ang_vels:
                actions.append([lv.item(), av.item()])

        self.register_buffer(
            "candidate_actions",
            torch.tensor(actions[:num_candidate_actions], dtype=torch.float32),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(2, action_hidden_dim),
            nn.ReLU(),
            nn.Linear(action_hidden_dim, output_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.score_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
        )

    def forward(
        self,
        aig_encoder: AIGGraphEncoder,
        core: torch.Tensor,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor] = None,
        ped_mask: Optional[torch.Tensor] = None,
        static_obstacles: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        goal_position: Optional[torch.Tensor] = None,
        detach_encoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_actions = self.candidate_actions
        action_encodings = aig_encoder.encode_actions(
            core,
            candidate_actions,
            ped_positions,
            ped_velocities,
            ped_histories=ped_histories,
            ped_mask=ped_mask,
            static_obstacles=static_obstacles,
            static_mask=static_mask,
            goal_position=goal_position,
        )

        if detach_encoder:
            action_encodings = action_encodings.detach()

        action_embed = self.action_encoder(candidate_actions)  # [A, D]
        action_embed = action_embed.unsqueeze(0).expand(action_encodings.size(0), -1, -1)

        fused = self.fusion(torch.cat([action_encodings, action_embed], dim=-1))

        scores = self.score_head(fused).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * fused, dim=1)

        return context, weights, candidate_actions


class IterativeEquilibriumModule(nn.Module):
    """Iterative equilibrium reasoning module for the actor path.

    Runs K rounds of (intention -> predict pedestrian response -> refine intention).
    Each round uses lightweight trajectory rollout to estimate collision probability.
    When K=0 the refinement loop is skipped (z_fused -> intention_init -> output_proj),
    but the IEM keeps its own LSTM + attention scene encoder. This is architecturally
    distinct from ActionSetAggregator (the predecessor AIG-Nav design, which uses
    GATv2 graph encoding with softmax aggregation over discrete candidate actions);
    the two are independent designs, not mathematical limits of one another.
    """

    def __init__(
        self,
        core_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        intention_dim: int = 64,
        action_dim: int = 2,
        K: int = 1,
        pred_horizon: int = 12,
        dt: float = 0.1,
        history_length: int = 10,
        max_pedestrians: int = 10,
        robot_radius: float = 0.25,
        ped_radius: float = 0.3,
        safety_margin: float = 0.2,
        goal_distance_scale: float = 6.0,
        lambda_conv: float = 0.1,
    ):
        super().__init__()

        self.K = K
        self.intention_dim = intention_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.dt = dt
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.history_length = history_length
        self.max_pedestrians = max_pedestrians
        self.robot_radius = robot_radius
        self.ped_radius = ped_radius
        self.safety_margin = safety_margin
        self.goal_distance_scale = goal_distance_scale
        self.lambda_conv = lambda_conv

        self.ped_traj_encoder = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.social_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        self.core_encoder = nn.Sequential(
            nn.Linear(core_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # (x, y, radius)
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # core + social + static
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Iteration block (weights shared across K rounds)
        self.intention_init = nn.Sequential(
            nn.Linear(hidden_dim, intention_dim),
            nn.ReLU(),
            nn.Linear(intention_dim, intention_dim),
            nn.Tanh(),
        )

        self.action_decoder = nn.Sequential(
            nn.Linear(intention_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

        self.ped_response_predictor = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        self.response_scale = nn.Parameter(torch.tensor(0.3))

        self.response_per_ped = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.response_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        # Intention refinement: (z_fused, z_response, intention, goal_progress) -> intention
        self.f_refine = nn.Sequential(
            nn.Linear(hidden_dim * 2 + intention_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, intention_dim),
            nn.Tanh(),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + intention_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self._all_intentions = []

    def _predict_ped_response(self, ped_positions, ped_velocities, robot_action, ped_mask=None):
        """Lightweight pedestrian response: velocity correction + constant-velocity rollout -> collision probabilities."""
        B, N, _ = ped_positions.shape
        device = ped_positions.device

        action_exp = robot_action.unsqueeze(1).expand(-1, N, -1)
        ped_input = torch.cat([ped_positions, ped_velocities, action_exp], dim=-1)

        vel_correction = self.ped_response_predictor(
            ped_input.reshape(B * N, 6)
        ).reshape(B, N, 2)

        dist_to_origin = torch.norm(ped_positions, dim=-1, keepdim=True).clamp(min=0.1)
        influence = torch.exp(-dist_to_origin / 2.0)
        corrected_vel = ped_velocities + self.response_scale * influence * vel_correction

        ped_trajs = []
        current = ped_positions.clone()
        for t in range(self.pred_horizon):
            current = current + corrected_vel * self.dt
            ped_trajs.append(current.clone())
        ped_trajs = torch.stack(ped_trajs, dim=2)  # [B, N, T, 2]

        robot_pos = torch.zeros(B, 2, device=device)
        robot_yaw = torch.zeros(B, device=device)
        robot_traj = predict_robot_trajectory(
            robot_pos, robot_yaw, robot_action,
            pred_horizon=self.pred_horizon, dt=self.dt
        )

        collision_probs = compute_collision_probability(
            robot_traj, ped_trajs,
            robot_radius=self.robot_radius, ped_radius=self.ped_radius,
            safety_margin=self.safety_margin, ped_mask=ped_mask,
        )

        delta_positions = ped_trajs[:, :, -1, :] - ped_positions
        return collision_probs, delta_positions

    def _scale_action(self, raw_action):
        """Map Tanh outputs in [-1, 1] to physical action ranges."""
        lin = raw_action[:, 0:1] * 0.4 + 0.2   # [-1, 1] -> [-0.2, 0.6] m/s
        ang = raw_action[:, 1:2] * 0.5          # [-1, 1] -> [-0.5, 0.5] rad/s
        return torch.cat([lin, ang], dim=-1)

    def _compute_goal_progress(self, core, robot_action):
        """Goal-progress signal in [-1, 1]: positive = approaches goal, negative = retreats."""
        B = core.size(0)
        device = core.device

        goal_dist = core[:, 0] * self.goal_distance_scale
        goal_x = goal_dist * core[:, 1]
        goal_y = goal_dist * core[:, 2]
        goal_pos = torch.stack([goal_x, goal_y], dim=-1)  # [B, 2]

        robot_pos = torch.zeros(B, 2, device=device)
        robot_yaw = torch.zeros(B, device=device)
        robot_traj = predict_robot_trajectory(
            robot_pos, robot_yaw, robot_action,
            pred_horizon=self.pred_horizon, dt=self.dt
        )
        final_pos = robot_traj[:, -1, :]  # [B, 2]

        initial_dist = torch.norm(goal_pos, dim=-1)
        final_dist = torch.norm(goal_pos - final_pos, dim=-1)
        goal_progress = (initial_dist - final_dist) / self.goal_distance_scale

        return goal_progress  # [B]

    def forward(self, core, ped_positions, ped_velocities,
                ped_histories=None, ped_mask=None,
                static_obstacles=None, static_mask=None,
                detach_encoder=False):
        """
        Returns:
            action_context: [B, output_dim]
            final_candidate_action: [B, 2]
            all_intentions: List[Tensor]
        """
        device = core.device
        B = core.size(0)
        N = ped_positions.size(1) if ped_positions.dim() > 1 else 0

        # Step 1: encode scene
        core_enc = self.core_encoder(core)

        if N > 0:
            if ped_histories is not None and ped_histories.numel() > 0:
                hist = ped_histories
                if hist.size(2) < self.history_length:
                    pad_len = self.history_length - hist.size(2)
                    pad = hist[:, :, :1, :].expand(-1, -1, pad_len, -1)
                    hist = torch.cat([pad, hist], dim=2)
                elif hist.size(2) > self.history_length:
                    hist = hist[:, :, -self.history_length:, :]
            else:
                state = torch.cat([ped_positions, ped_velocities], dim=-1)
                hist = state.unsqueeze(2).expand(-1, -1, self.history_length, -1)

            hist_flat = hist.reshape(B * N, self.history_length, 4)
            _, (h_n, _) = self.ped_traj_encoder(hist_flat)
            e_peds = h_n[-1].reshape(B, N, self.hidden_dim)

            key_padding_mask = ~ped_mask if ped_mask is not None else None
            z_social, _ = self.social_attention(
                e_peds, e_peds, e_peds,
                key_padding_mask=key_padding_mask,
            )

            if ped_mask is not None:
                mask_f = ped_mask.unsqueeze(-1).float()
                z_social_pooled = (z_social * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            else:
                z_social_pooled = z_social.mean(dim=1)
        else:
            e_peds = torch.zeros(B, 0, self.hidden_dim, device=device)
            z_social_pooled = torch.zeros(B, self.hidden_dim, device=device)

        if (static_obstacles is not None and static_obstacles.numel() > 0
                and static_obstacles.size(1) > 0):
            s_enc = self.static_encoder(static_obstacles)  # [B, M, hidden_dim]
            if static_mask is not None:
                s_mask_f = static_mask.unsqueeze(-1).float()
                z_static_pooled = (
                    (s_enc * s_mask_f).sum(dim=1)
                    / s_mask_f.sum(dim=1).clamp(min=1.0)
                )
            else:
                z_static_pooled = s_enc.mean(dim=1)
        else:
            z_static_pooled = torch.zeros(B, self.hidden_dim, device=device)

        z_fused = self.fusion(torch.cat([core_enc, z_social_pooled, z_static_pooled], dim=-1))

        if detach_encoder:
            z_fused = z_fused.detach()
            e_peds = e_peds.detach() if e_peds.numel() > 0 else e_peds

        # Step 2: iterative equilibrium reasoning
        intention = self.intention_init(z_fused)
        all_intentions = [intention]

        if self.K > 0 and N > 0:
            for k in range(self.K):
                raw_action = self.action_decoder(intention)
                candidate_action = self._scale_action(raw_action)
                collision_probs, delta_pos = self._predict_ped_response(
                    ped_positions, ped_velocities, candidate_action, ped_mask
                )

                goal_progress = self._compute_goal_progress(core, candidate_action)

                resp_input = torch.cat([
                    collision_probs.unsqueeze(-1),
                    delta_pos,
                    e_peds,
                ], dim=-1)

                resp_features = self.response_per_ped(
                    resp_input.reshape(B * N, -1)
                ).reshape(B, N, self.hidden_dim)

                attn_scores = self.response_attention(resp_features).squeeze(-1)
                if ped_mask is not None:
                    attn_scores = attn_scores.masked_fill(~ped_mask, float('-inf'))
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = torch.where(
                    torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
                )
                z_response = torch.einsum('bn,bnd->bd', attn_weights, resp_features)

                refine_input = torch.cat([
                    z_fused, z_response, intention, goal_progress.unsqueeze(-1)
                ], dim=-1)
                intention = self.f_refine(refine_input)
                all_intentions.append(intention)

        # Step 3: output projection
        action_context = self.output_proj(torch.cat([z_fused, intention], dim=-1))
        action_context = self.layer_norm(action_context)

        final_candidate = self._scale_action(self.action_decoder(intention))
        self._all_intentions = all_intentions

        return action_context, final_candidate, all_intentions

    def get_convergence_loss(self):
        """Convergence regularizer: encourages consecutive iterations' intentions to stay close."""
        if len(self._all_intentions) < 2:
            return torch.tensor(0.0, device=self._all_intentions[0].device)

        conv_loss = torch.tensor(0.0, device=self._all_intentions[0].device)
        for i in range(1, len(self._all_intentions)):
            conv_loss = conv_loss + F.mse_loss(
                self._all_intentions[i],
                self._all_intentions[i - 1].detach()
            )
        return self.lambda_conv * conv_loss / (len(self._all_intentions) - 1)


class ActionConditionedAIGEncoder(nn.Module):
    """
    Standalone action-conditioned encoder for ablations or discrete policies.
    """

    def __init__(
        self,
        node_feat_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        num_candidate_actions: int = 9,
        pred_horizon: int = 12,
        history_length: int = 10,
    ):
        super().__init__()
        self.aig_encoder = AIGGraphEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            pred_horizon=pred_horizon,
            history_length=history_length,
        )
        self.aggregator = ActionSetAggregator(
            output_dim=output_dim,
            num_candidate_actions=num_candidate_actions,
        )

    def forward(
        self,
        core: torch.Tensor,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_histories: Optional[torch.Tensor] = None,
        ped_mask: Optional[torch.Tensor] = None,
        static_obstacles: Optional[torch.Tensor] = None,
        static_mask: Optional[torch.Tensor] = None,
        goal_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context, weights, candidate_actions = self.aggregator(
            self.aig_encoder,
            core,
            ped_positions,
            ped_velocities,
            ped_histories=ped_histories,
            ped_mask=ped_mask,
            static_obstacles=static_obstacles,
            static_mask=static_mask,
            goal_position=goal_position,
        )
        return context, candidate_actions


class SimpleAIGEncoder(nn.Module):
    """
    Simplified AIG encoder for faster experiments.
    """

    def __init__(
        self,
        scan_dim: int = 32,
        core_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        pred_horizon: int = 12,
        history_length: int = 10,
    ):
        super().__init__()
        self.scan_dim = scan_dim
        self.core_dim = core_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pred_horizon = pred_horizon
        self.history_length = history_length

        self.traj_predictor = SimpleTrajectoryPredictor(pred_horizon=pred_horizon, hidden_dim=32)

        self.scan_encoder = nn.Sequential(
            nn.Linear(scan_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.ped_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        self.ped_attention = nn.Sequential(nn.Linear(hidden_dim // 2, 1))

        self.core_encoder = nn.Sequential(
            nn.Linear(core_dim, hidden_dim // 4),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
        )

        fusion_dim = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        scan: torch.Tensor,
        core: torch.Tensor,
        robot_action: torch.Tensor,
        ped_positions: torch.Tensor,
        ped_velocities: torch.Tensor,
        ped_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = scan.size(0)
        device = scan.device

        scan_enc = self.scan_encoder(scan)
        core_enc = self.core_encoder(core)
        action_enc = self.action_encoder(robot_action)

        num_peds = ped_positions.size(1) if ped_positions.dim() > 1 else 0
        if num_peds > 0:
            robot_pos = torch.zeros(batch_size, 2, device=device)
            ped_trajs = self.traj_predictor(ped_positions, ped_velocities, robot_action, robot_pos)
            robot_traj = predict_robot_trajectory(robot_pos, torch.zeros(batch_size, device=device), robot_action, self.pred_horizon, 0.1)
            collision_probs = compute_collision_probability(robot_traj, ped_trajs, ped_mask=ped_mask)

            ped_features = torch.cat(
                [
                    ped_positions,
                    ped_velocities,
                    collision_probs.unsqueeze(-1),
                ],
                dim=-1,
            )
            ped_enc = self.ped_encoder(ped_features)
            attn_scores = self.ped_attention(ped_enc).squeeze(-1)

            if ped_mask is not None:
                attn_scores = attn_scores.masked_fill(~ped_mask, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
            ped_context = torch.einsum("bn,bnd->bd", attn_weights, ped_enc)
        else:
            ped_context = torch.zeros(batch_size, self.hidden_dim // 2, device=device)

        fused = torch.cat([scan_enc, ped_context, core_enc, action_enc], dim=-1)
        output = self.fusion(fused)
        return self.layer_norm(output)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing AIG encoder on device: {device}")

    batch_size = 2
    num_peds = 3
    history_len = 10

    core = torch.randn(batch_size, 3, device=device)
    action = torch.randn(batch_size, 2, device=device)
    ped_pos = torch.randn(batch_size, num_peds, 2, device=device)
    ped_vel = torch.randn(batch_size, num_peds, 2, device=device) * 0.5
    ped_hist = torch.randn(batch_size, num_peds, history_len, 4, device=device)
    ped_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool, device=device)
    static_obs = torch.randn(batch_size, 2, 3, device=device)
    static_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool, device=device)

    encoder = AIGGraphEncoder(history_length=history_len).to(device)
    context = encoder(core, action, ped_pos, ped_vel, ped_histories=ped_hist, ped_mask=ped_mask, static_obstacles=static_obs, static_mask=static_mask)
    print(f"AIGGraphEncoder output shape: {context.shape}")
    assert context.shape == (batch_size, encoder.output_dim)

    aggregator = ActionSetAggregator(output_dim=encoder.output_dim).to(device)
    ctx, weights, cand = aggregator(
        encoder,
        core,
        ped_pos,
        ped_vel,
        ped_histories=ped_hist,
        ped_mask=ped_mask,
        static_obstacles=static_obs,
        static_mask=static_mask,
        detach_encoder=True,
    )
    print(f"ActionSetAggregator context shape: {ctx.shape}")
    print(f"ActionSetAggregator weights shape: {weights.shape}")
    assert ctx.shape == (batch_size, encoder.output_dim)

    print("All AIG encoder tests passed!")

# Backward-compatible alias
AIGEncoder = AIGGraphEncoder
