"""Train an AIG-Nav ablation variant (TD3 version)."""

import argparse
import math
import os
import sys
import time
import random
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from ablation_configs import get_variant

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Torch backward compatibility (system Python 3.10 has torch 2.1)
if not hasattr(torch.amp, 'GradScaler'):
    class _GradScalerCompat(torch.cuda.amp.GradScaler):
        def __init__(self, device=None, **kwargs):
            super().__init__(**kwargs)
    torch.amp.GradScaler = _GradScalerCompat
if not hasattr(torch, 'autocast'):
    torch.autocast = torch.cuda.amp.autocast

from aig_nav_core import AIGNavTD3, AIGNavReplayBuffer, PygameNavEnvAIGNav
from scripts.config_utils import load_config, cfg_get
from scripts.eval_utils import (
    DEFAULT_EVAL_SCENARIO_SEED,
    generate_eval_scenarios,
    set_global_seeds,
)


def map_action_to_cmd(action):
    if action[0] >= 0:
        lin_vel = action[0]
    else:
        lin_vel = action[0] * 0.3
    ang_vel = action[1]
    return float(lin_vel), float(ang_vel)


def format_progress_bar(current: int, total: int, width: int = 20):
    if total <= 0:
        return "[--------------------]", 0.0
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    return bar, ratio


def make_progress_bar(total: int, desc: str):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)


def evaluate(model, eval_env, scenarios, log_step: int, max_steps, seed_base: int, eval_count: int = 0):
    model.sync_inference()
    total_reward = 0.0
    total_col = 0
    total_goal = 0
    total_steps = 0

    rng_state_np = np.random.get_state()
    rng_state_py = random.getstate()

    for idx, scenario in enumerate(scenarios):
        seed = seed_base + idx
        set_global_seeds(seed)
        obs = eval_env.reset(scenario=scenario, eval_seed=seed)
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
                add_noise=False,
            )

            lin_vel, ang_vel = map_action_to_cmd(action)
            obs = eval_env.step(lin_vel, ang_vel)

            episode_reward += obs["reward"]
            steps += 1

        total_reward += episode_reward
        total_col += int(obs["collision"])
        total_goal += int(obs["goal_reached"])
        total_steps += steps

    n = len(scenarios)
    avg_reward = total_reward / max(n, 1)
    avg_col = total_col / max(n, 1)
    avg_goal = total_goal / max(n, 1)
    avg_steps = total_steps / max(n, 1)

    model.writer.add_scalar("eval/avg_reward", avg_reward, log_step)
    model.writer.add_scalar("eval/avg_col", avg_col, log_step)
    model.writer.add_scalar("eval/avg_goal", avg_goal, log_step)
    model.writer.add_scalar("eval/avg_steps", avg_steps, log_step)

    print(
        f"[EVAL #{eval_count}][step {log_step}] avg_reward={avg_reward:.2f} "
        f"collision_rate={avg_col:.2f} goal_rate={avg_goal:.2f} "
        f"avg_steps={avg_steps:.1f}"
    )

    np.random.set_state(rng_state_np)
    random.setstate(rng_state_py)

    return avg_reward, avg_col, avg_goal, avg_steps


def is_better(curr: dict, best: dict | None) -> bool:
    if best is None:
        return True
    if curr["avg_goal"] != best["avg_goal"]:
        return curr["avg_goal"] > best["avg_goal"]
    if curr["avg_col"] != best["avg_col"]:
        return curr["avg_col"] < best["avg_col"]
    if curr["avg_steps"] != best["avg_steps"]:
        return curr["avg_steps"] < best["avg_steps"]
    return False


def set_seed(seed: int, deterministic: bool = False):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="experiments/configs/defaults.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train AIG-Nav-TD3 ablation")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--runs-root", default="experiments/runs/ablations")
    parser.add_argument("--models-root", default="experiments/models/ablations")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-epochs", type=int, default=cfg_get(cfg, "train", "max_epochs", 300))
    parser.add_argument("--episodes-per-epoch", type=int, default=cfg_get(cfg, "train", "episodes_per_epoch", 50))
    parser.add_argument("--max-steps", type=int, default=cfg_get(cfg, "train", "max_steps", 500))
    parser.add_argument("--total-steps", type=int, default=cfg_get(cfg, "train", "total_steps", None))
    parser.add_argument("--eval-every-steps", type=int, default=cfg_get(cfg, "train", "eval_every_steps", None))
    parser.add_argument("--eval-scenario-seed", type=int, default=cfg_get(cfg, "train", "eval_scenario_seed", DEFAULT_EVAL_SCENARIO_SEED))
    parser.add_argument("--train-every-n", type=int, default=cfg_get(cfg, "train", "train_every_n", 2))
    parser.add_argument("--training-iterations", type=int, default=cfg_get(cfg, "train", "training_iterations", 50))
    parser.add_argument("--batch-size", type=int, default=cfg_get(cfg, "train", "batch_size", 128))
    parser.add_argument("--random-steps", type=int, default=cfg_get(cfg, "train", "random_steps", 5000))
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--num-pedestrians", type=int, default=cfg_get(cfg, "env", "num_pedestrians", 4))
    parser.add_argument("--pedestrian-reactivity", type=float, default=cfg_get(cfg, "env", "pedestrian_reactivity", 0.0))
    parser.add_argument("--pred-horizon", type=int, default=cfg_get(cfg, "env", "pred_horizon", 12))
    parser.add_argument("--history-length", type=int, default=cfg_get(cfg, "env", "history_length", 10))
    parser.add_argument("--num-candidate-actions", type=int, default=cfg_get(cfg, "model", "num_candidate_actions", 9))
    parser.add_argument("--aux-pred-loss-weight", type=float, default=cfg_get(cfg, "model", "aux_pred_loss_weight", 0.5))
    parser.add_argument("--aig-hidden-dim", type=int, default=cfg_get(cfg, "model", "aig_hidden_dim", 128))
    parser.add_argument("--aig-output-dim", type=int, default=cfg_get(cfg, "model", "aig_output_dim", 128))
    parser.add_argument("--use-iterative-eq", type=int, default=cfg_get(cfg, "iaig", "use_iterative_eq", 1),
                        help="1=IAIG-Nav (iterative eq, our primary method), 0=AIG-Nav (our predecessor design with discrete ActionSet)")
    parser.add_argument("--K", type=int, default=cfg_get(cfg, "iaig", "K", 1),
                        help="Number of equilibrium iterations")
    parser.add_argument("--intention-dim", type=int, default=cfg_get(cfg, "iaig", "intention_dim", 64))
    parser.add_argument("--lambda-conv", type=float, default=cfg_get(cfg, "iaig", "lambda_conv", 0.1))
    # TD3-specific parameters
    parser.add_argument("--expl-noise", type=float, default=0.2,
                        help="Exploration noise std")
    parser.add_argument("--policy-noise", type=float, default=0.2,
                        help="Target policy smoothing noise")
    parser.add_argument("--noise-clip", type=float, default=0.5,
                        help="Noise clip range")
    parser.add_argument("--policy-freq", type=int, default=2,
                        help="Delayed policy update frequency")
    # Sim-to-real parameters
    parser.add_argument("--robot-radius", type=float, default=cfg_get(cfg, "env", "robot_radius", 0.25))
    parser.add_argument("--scan-fov", type=float, default=cfg_get(cfg, "env", "scan_fov", math.pi / 2))
    parser.add_argument("--scan-noise-std", type=float, default=cfg_get(cfg, "env", "scan_noise_std", 0.005))
    # Domain randomization parameters
    parser.add_argument("--randomize-obstacles", type=int, default=cfg_get(cfg, "train", "randomize_obstacles", 1))
    parser.add_argument("--randomize-num-peds", type=int, default=cfg_get(cfg, "train", "randomize_num_pedestrians", 1))
    parser.add_argument("--ped-speed-min", type=float, default=cfg_get(cfg, "train", "ped_speed_min", 0.5))
    parser.add_argument("--ped-speed-max", type=float, default=cfg_get(cfg, "train", "ped_speed_max", 1.0))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = "cpu"

    if args.total_steps is None:
        args.total_steps = args.max_epochs * args.episodes_per_epoch * args.max_steps
    if args.eval_every_steps is None:
        args.eval_every_steps = args.episodes_per_epoch * args.max_steps
    if args.eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive")
    if args.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if args.total_steps < args.eval_every_steps:
        print("[WARN] total_steps < eval_every_steps; only one eval will run.")
    elif args.total_steps % args.eval_every_steps != 0:
        print("[WARN] total_steps is not divisible by eval_every_steps; last eval may be skipped.")

    set_seed(args.seed, deterministic=args.deterministic)

    variant = get_variant(args.variant)
    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    if args.seed is not None and f"seed{args.seed}" not in run_name:
        run_name = f"{run_name}_seed{args.seed}"

    env = PygameNavEnvAIGNav(
        render=args.render,
        render_fps=args.render_fps,
        num_pedestrians=args.num_pedestrians,
        pedestrian_reactivity=args.pedestrian_reactivity,
        seed=args.seed,
        history_length=args.history_length,
        prediction_horizon=args.pred_horizon,
        robot_radius=args.robot_radius,
        scan_fov=args.scan_fov,
        scan_noise_std=args.scan_noise_std,
        randomize_obstacles=bool(args.randomize_obstacles),
        randomize_num_pedestrians=bool(args.randomize_num_peds),
        ped_speed_range=(args.ped_speed_min, args.ped_speed_max),
    )
    eval_env = PygameNavEnvAIGNav(
        render=False,
        render_fps=args.render_fps,
        num_pedestrians=args.num_pedestrians,
        pedestrian_reactivity=args.pedestrian_reactivity,
        seed=args.seed,
        history_length=args.history_length,
        prediction_horizon=args.pred_horizon,
        robot_radius=args.robot_radius,
        scan_fov=args.scan_fov,
        scan_noise_std=args.scan_noise_std,
        randomize_obstacles=False,
        randomize_num_pedestrians=False,
        ped_speed_range=(args.ped_speed_min, args.ped_speed_max),
    )

    obs = env.reset()
    scan_dim = len(obs["scan"])
    core_dim = 3
    action_dim = 2

    # Compute max capacities for variable-size inputs
    if args.randomize_num_peds:
        max_peds = max(args.num_pedestrians, env.num_pedestrians_range[1]) + 2
    else:
        max_peds = args.num_pedestrians + 2
    if args.randomize_obstacles:
        max_static = max(len(env.static_obstacles), env.num_obstacles_range[1])
    else:
        max_static = len(env.static_obstacles)

    save_dir = Path(args.models_root) / args.variant / run_name
    os.makedirs(save_dir, exist_ok=True)

    model = AIGNavTD3(
        scan_dim=scan_dim,
        core_dim=core_dim,
        action_dim=action_dim,
        device=args.device,
        max_action=1.0,
        # TD3 hyperparameters
        discount=0.99,
        tau=0.005,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        actor_lr=3e-4,
        critic_lr=3e-4,
        expl_noise=args.expl_noise,
        # AIG encoder parameters
        aig_hidden_dim=args.aig_hidden_dim,
        aig_output_dim=args.aig_output_dim,
        pred_horizon=args.pred_horizon,
        max_pedestrians=max_peds,
        history_length=args.history_length,
        max_static_obstacles=max_static,
        goal_distance_scale=getattr(env, "max_goal_dist", 6.0),
        num_candidate_actions=args.num_candidate_actions,
        aux_pred_loss_weight=args.aux_pred_loss_weight,
        hidden_dim=256,
        hidden_depth=2,
        save_every=0,
        save_directory=save_dir,
        model_name=f"AIGNavTD3_{args.variant}",
        # === IAIG parameters ===
        use_iterative_eq=bool(args.use_iterative_eq),
        K=args.K,
        intention_dim=args.intention_dim,
        lambda_conv=args.lambda_conv,
        robot_radius=args.robot_radius,
        **variant["params"],
    )

    log_dir = Path(args.runs_root) / args.variant / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model.writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[INFO] TensorBoard logs: {log_dir}")

    config_path = log_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "variant": args.variant,
                "variant_params": variant["params"],
            },
            f,
            indent=2,
        )

    best_metrics = None
    best_dir = save_dir / "best"
    best_metrics_path = best_dir / "best_eval.json"

    replay_buffer = AIGNavReplayBuffer(
        buffer_size=args.buffer_size,
        scan_dim=scan_dim,
        core_dim=core_dim,
        action_dim=action_dim,
        max_pedestrians=max_peds,
        history_length=args.history_length,
        max_static_obstacles=max_static,
        device=args.device,
    )

    train_eval_episodes = cfg_get(cfg, "eval", "episodes", 100)
    eval_scenarios = generate_eval_scenarios(
        eval_env, int(train_eval_episodes), seed=args.eval_scenario_seed
    )

    epoch = 0
    episode = 0
    steps = 0
    total_steps = 0
    next_eval_step = args.eval_every_steps
    eval_count = 0
    progress_width = 20
    progress_interval = max(int(args.eval_every_steps // progress_width), 1)
    progress_update = max(int(args.eval_every_steps // 200), 1)
    last_eval_step = 0
    next_progress = progress_interval
    progress_bar = None
    progress_count = 0

    ped_states = obs["pedestrian_states"]
    ped_pos = ped_states["positions"]
    ped_vel = ped_states["velocities"]
    ped_hist = ped_states["histories"]
    ped_pos_world = ped_states["positions_world"]
    static_obs = obs["static_obstacles"]
    robot_pose = obs["robot_pose"]
    scan = obs["scan"]
    distance = obs["distance"]
    cos_v = obs["cos_v"]
    sin_v = obs["sin_v"]

    print("=" * 60)
    _method_tag = "IAIG-Nav-TD3" if args.use_iterative_eq else "AIG-Nav-TD3"
    print(f"{_method_tag} Ablation Training ({args.variant})")
    print(f"run={run_name} seed={args.seed} device={args.device}")
    print(f"variant: {variant['description']}")
    print(
        f"total_steps={args.total_steps} eval_every_steps={args.eval_every_steps} "
        f"eval_scenario_seed={args.eval_scenario_seed}"
    )
    print(
        f"env: pedestrians={args.num_pedestrians} reactivity={args.pedestrian_reactivity} "
        f"pred_horizon={args.pred_horizon} history={args.history_length}"
    )
    print("=" * 60)
    progress_bar = make_progress_bar(args.eval_every_steps, "train")

    while total_steps < args.total_steps:
        norm_distance = min(distance / 6.0, 1.0)
        core = np.array([norm_distance, cos_v, sin_v], dtype=np.float32)

        if total_steps < args.random_steps:
            action = np.array([
                np.random.uniform(-0.3, 1.0),
                np.random.uniform(-1.0, 1.0),
            ], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
        else:
            action = model.act(
                scan,
                core,
                ped_pos,
                ped_vel,
                ped_histories=ped_hist,
                static_obstacles=static_obs,
                add_noise=True,
            )

        lin_vel, ang_vel = map_action_to_cmd(action)
        next_obs = env.step(lin_vel, ang_vel)

        next_ped_states = next_obs["pedestrian_states"]
        next_ped_pos = next_ped_states["positions"]
        next_ped_vel = next_ped_states["velocities"]
        next_ped_hist = next_ped_states["histories"]
        next_ped_pos_world = next_ped_states["positions_world"]
        next_static_obs = next_obs["static_obstacles"]
        next_scan = next_obs["scan"]
        next_distance = next_obs["distance"]
        next_cos = next_obs["cos_v"]
        next_sin = next_obs["sin_v"]

        next_norm_distance = min(next_distance / 6.0, 1.0)
        next_core = np.array([next_norm_distance, next_cos, next_sin], dtype=np.float32)

        reward = next_obs["reward"]
        collision = next_obs["collision"]
        goal = next_obs["goal_reached"]
        episode_done = collision or goal or (steps + 1) >= args.max_steps

        replay_buffer.add(
            scan=scan,
            core=core,
            action=action,
            reward=reward,
            next_scan=next_scan,
            next_core=next_core,
            done=episode_done,
            ped_positions=ped_pos,
            ped_velocities=ped_vel,
            ped_histories=ped_hist,
            next_ped_positions=next_ped_pos,
            next_ped_velocities=next_ped_vel,
            next_ped_histories=next_ped_hist,
            static_obstacles=static_obs,
            next_static_obstacles=next_static_obs,
            ped_positions_world=ped_pos_world,
            next_ped_positions_world=next_ped_pos_world,
            robot_pose=robot_pose,
        )

        scan = next_scan
        distance = next_distance
        cos_v = next_cos
        sin_v = next_sin
        ped_pos = next_ped_pos
        ped_vel = next_ped_vel
        ped_hist = next_ped_hist
        ped_pos_world = next_ped_pos_world
        robot_pose = next_obs["robot_pose"]
        static_obs = next_static_obs

        steps += 1
        total_steps += 1

        progress_since_eval = total_steps - last_eval_step
        if progress_bar is not None:
            progress_target = min(progress_since_eval, args.eval_every_steps)
            delta = progress_target - progress_count
            if delta >= progress_update or progress_target == args.eval_every_steps:
                progress_bar.update(delta)
                progress_count += delta
        elif progress_since_eval >= next_progress:
            bar, ratio = format_progress_bar(progress_since_eval, args.eval_every_steps, progress_width)
            print(
                f"[PROGRESS] {progress_since_eval}/{args.eval_every_steps} {bar} "
                f"{ratio * 100:5.1f}% step={total_steps}"
            )
            next_progress += progress_interval

        if total_steps >= next_eval_step:
            while total_steps >= next_eval_step:
                eval_count += 1
                avg_reward, avg_col, avg_goal, avg_steps = evaluate(
                    model,
                    eval_env,
                    eval_scenarios,
                    next_eval_step,
                    args.max_steps,
                    seed_base=args.eval_scenario_seed,
                    eval_count=eval_count,
                )
                current = {
                    "step": next_eval_step,
                    "eval_count": eval_count,
                    "avg_reward": avg_reward,
                    "avg_col": avg_col,
                    "avg_goal": avg_goal,
                    "avg_steps": avg_steps,
                    "criteria": "goal_rate desc, collision_rate asc, avg_steps asc",
                }
                if is_better(current, best_metrics):
                    best_metrics = current
                    model.save(best_dir)
                    with best_metrics_path.open("w", encoding="utf-8") as f:
                        json.dump(best_metrics, f, indent=2)
                    print(
                        f"[BEST][step {next_eval_step}] goal_rate={avg_goal:.2f} "
                        f"collision_rate={avg_col:.2f} avg_steps={avg_steps:.1f}"
                    )
                if progress_bar is not None:
                    remaining = args.eval_every_steps - progress_count
                    if remaining > 0:
                        progress_bar.update(remaining)
                    progress_bar.close()
                    progress_bar = make_progress_bar(args.eval_every_steps, "train")
                    progress_count = 0
                else:
                    next_progress = progress_interval
                last_eval_step = next_eval_step
                next_eval_step += args.eval_every_steps

        if episode_done:
            episode += 1

            if episode % args.train_every_n == 0 and len(replay_buffer) > args.batch_size:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=args.training_iterations,
                    batch_size=args.batch_size,
                )

            if episode % args.episodes_per_epoch == 0:
                epoch += 1
                episode = 0

            obs = env.reset()
            model.reset_episode()

            ped_states = obs["pedestrian_states"]
            ped_pos = ped_states["positions"]
            ped_vel = ped_states["velocities"]
            ped_hist = ped_states["histories"]
            ped_pos_world = ped_states["positions_world"]
            static_obs = obs["static_obstacles"]
            robot_pose = obs["robot_pose"]
            scan = obs["scan"]
            distance = obs["distance"]
            cos_v = obs["cos_v"]
            sin_v = obs["sin_v"]
            steps = 0

    if progress_bar is not None:
        progress_bar.close()

    # Save final model and run final evaluation
    final_dir = save_dir / "final"
    model.save(final_dir)
    eval_count += 1
    avg_reward, avg_col, avg_goal, avg_steps = evaluate(
        model, eval_env, eval_scenarios, total_steps,
        args.max_steps, seed_base=args.eval_scenario_seed,
        eval_count=eval_count,
    )
    final_metrics = {
        "step": total_steps,
        "eval_count": eval_count,
        "avg_reward": avg_reward,
        "avg_col": avg_col,
        "avg_goal": avg_goal,
        "avg_steps": avg_steps,
    }
    final_metrics_path = final_dir / "final_eval.json"
    with final_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(
        f"[FINAL][step {total_steps}] goal_rate={avg_goal:.2f} "
        f"collision_rate={avg_col:.2f} avg_steps={avg_steps:.1f}"
    )

    env.close()
    eval_env.close()
    print(f"[INFO] Training complete! (variant={args.variant})")


if __name__ == "__main__":
    main()
