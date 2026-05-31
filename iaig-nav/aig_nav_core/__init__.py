"""
AIG-Nav core package.

Provides the environment, models, and utilities for action-conditioned
interaction graph navigation.
"""

from .pygame_env_aig_nav import PygameNavEnvAIGNav
from .trajectory_predictor import (
    SimpleTrajectoryPredictor,
    ReactiveTrajectoryPredictor,
    compute_collision_probability,
    compute_pairwise_collision_probability,
    compute_static_collision_probability,
    predict_robot_trajectory,
)
from .aig_encoder import (
    SimpleAIGEncoder,
    AIGGraphEncoder,
    ActionConditionedAIGEncoder,
    ActionSetAggregator,
    IterativeEquilibriumModule,
)
from .aig_nav import AIGNav
from .aig_nav_td3 import AIGNavTD3
from .aig_nav_replay_buffer import AIGNavReplayBuffer

__version__ = "1.0.0"
__author__ = "Research Implementation"

__all__ = [
    "PygameNavEnvAIGNav",
    "SimpleTrajectoryPredictor",
    "ReactiveTrajectoryPredictor",
    "compute_collision_probability",
    "compute_pairwise_collision_probability",
    "compute_static_collision_probability",
    "predict_robot_trajectory",
    "SimpleAIGEncoder",
    "AIGGraphEncoder",
    "ActionConditionedAIGEncoder",
    "ActionSetAggregator",
    "IterativeEquilibriumModule",
    "AIGNav",
    "AIGNavTD3",
    "AIGNavReplayBuffer",
]
