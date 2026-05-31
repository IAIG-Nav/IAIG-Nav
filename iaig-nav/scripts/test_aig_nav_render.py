"""
Render test script for IAIG-Nav (Iterative Action-conditioned Interaction Graph Navigation).
python3 scripts/test_aig_nav_render.py --model-dir experiments/models/aig_nav_td3/<run_name> \
  --model-name AIGNavTD3_v1 --episodes 50 --max-steps 500


Usage:
  python scripts/test_aig_nav_render.py \
      --model-dir experiments/models/aig_nav_td3/<run_name> \
      --model-name AIGNavTD3_v1 \
      --episodes 5 \
      --max-steps 500
"""

import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from aig_nav_core import AIGNavTD3, PygameNavEnvAIGNav
from aig_nav_core.eval_scenarios import EVAL_SCENARIOS
from scripts.config_utils import load_config, cfg_get


def set_seed(seed: int, deterministic: bool = False):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def map_action_to_cmd(action):
    """Map neural network action to velocity command."""
    if action[0] >= 0:
        lin_vel = action[0]
    else:
        lin_vel = action[0] * 0.3
    ang_vel = action[1]
    return float(lin_vel), float(ang_vel)


def run_episode(model, env, scenario, max_steps):
    obs = env.reset(scenario=scenario) if scenario is not None else env.reset()
    model.reset_episode()

    episode_reward = 0.0
    steps = 0

    while steps < max_steps:
        if obs["collision"] or obs["goal_reached"]:
            break

        ped_states = obs["pedestrian_states"]
        ped_pos = ped_states["positions"]
        ped_vel = ped_states["velocities"]
        ped_hist = ped_states["histories"]
        static_obs = obs["static_obstacles"]

        scan = obs["scan"]
        distance = obs["distance"]
        cos_v = obs["cos_v"]
        sin_v = obs["sin_v"]

        norm_distance = min(distance / 6.0, 1.0)
        core = np.array([norm_distance, cos_v, sin_v], dtype=np.float32)

        action = model.act(
            scan,
            core,
            ped_pos,
            ped_vel,
            ped_histories=ped_hist,
            static_obstacles=static_obs,
            sample=False,
        )

        lin_vel, ang_vel = map_action_to_cmd(action)
        obs = env.step(lin_vel, ang_vel)

        episode_reward += obs["reward"]
        steps += 1

    return {
        "reward": episode_reward,
        "steps": steps,
        "collision": int(obs["collision"]),
        "goal": int(obs["goal_reached"]),
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="experiments/configs/defaults.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Render test for AIG-Nav")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--model-dir", default="experiments/models/aig_nav_td3")
    parser.add_argument("--model-name", default="AIGNavTD3_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=cfg_get(cfg, "render_test", "episodes", 5))
    parser.add_argument("--max-steps", type=int, default=cfg_get(cfg, "render_test", "max_steps", 500))
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--num-pedestrians", type=int, default=cfg_get(cfg, "env", "num_pedestrians", 4))
    parser.add_argument("--pedestrian-reactivity", type=float, default=cfg_get(cfg, "env", "pedestrian_reactivity", 0.0))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out-dir", default="experiments/results/aig_nav_render")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--use-eval-scenarios", action="store_true")
    parser.add_argument("--use-iterative-eq", type=int, default=cfg_get(cfg, "iaig", "use_iterative_eq", 1))
    parser.add_argument("--K", type=int, default=cfg_get(cfg, "iaig", "K", 3))
    parser.add_argument("--intention-dim", type=int, default=cfg_get(cfg, "iaig", "intention_dim", 64))
    parser.add_argument("--lambda-conv", type=float, default=cfg_get(cfg, "iaig", "lambda_conv", 0.1))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = "cpu"

    set_seed(args.seed, deterministic=args.deterministic)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    env = PygameNavEnvAIGNav(
        render=True,
        render_fps=args.render_fps,
        num_pedestrians=args.num_pedestrians,
        pedestrian_reactivity=args.pedestrian_reactivity,
        seed=args.seed,
    )

    obs = env.reset()
    scan_dim = len(obs["scan"])
    core_dim = 3
    action_dim = 2

    model = AIGNavTD3(
        scan_dim=scan_dim,
        core_dim=core_dim,
        action_dim=action_dim,
        device=args.device,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        aig_hidden_dim=128,
        aig_output_dim=128,
        pred_horizon=12,
        max_pedestrians=args.num_pedestrians + 2,
        history_length=env.history_length,
        max_static_obstacles=len(env.static_obstacles),
        goal_distance_scale=getattr(env, "max_goal_dist", 6.0),
        num_candidate_actions=9,
        aux_pred_loss_weight=0.0,
        hidden_dim=256,
        hidden_depth=2,
        save_every=0,
        save_directory=model_dir,
        model_name=args.model_name,
        # === IAIG parameters ===
        use_iterative_eq=bool(args.use_iterative_eq),
        K=args.K,
        intention_dim=args.intention_dim,
        lambda_conv=args.lambda_conv,
    )

    model.load(model_dir)
    model.sync_inference()

    print("[INFO] AIG-Nav render test started")

    total_reward = 0.0
    total_goal = 0
    total_collision = 0

    for ep in range(args.episodes):
        if args.use_eval_scenarios:
            scenario = EVAL_SCENARIOS[ep % len(EVAL_SCENARIOS)]
        else:
            scenario = None

        metrics = run_episode(model, env, scenario, args.max_steps)
        total_reward += metrics["reward"]
        total_goal += metrics["goal"]
        total_collision += metrics["collision"]

        print(
            f"[EP {ep + 1}/{args.episodes}] reward={metrics['reward']:.2f} "
            f"steps={metrics['steps']} goal={metrics['goal']} collision={metrics['collision']}"
        )

    avg_reward = total_reward / max(args.episodes, 1)
    goal_rate = total_goal / max(args.episodes, 1)
    collision_rate = total_collision / max(args.episodes, 1)

    print(
        f"[RESULT] avg_reward={avg_reward:.2f} "
        f"goal_rate={goal_rate:.2f} collision_rate={collision_rate:.2f}"
    )

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    if args.seed is not None and f"seed{args.seed}" not in run_name:
        run_name = f"{run_name}_seed{args.seed}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "num_pedestrians": args.num_pedestrians,
                "pedestrian_reactivity": args.pedestrian_reactivity,
                "seed": args.seed,
                "avg_reward": avg_reward,
                "goal_rate": goal_rate,
                "collision_rate": collision_rate,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Metrics saved: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
