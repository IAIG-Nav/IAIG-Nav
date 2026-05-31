"""Evaluate an AIG-Nav ablation variant (TD3 version)."""

import argparse
import json
import math
import time
import random
from pathlib import Path

import numpy as np
import torch

# torch 2.1 compat: torch.amp.GradScaler doesn't exist yet
if not hasattr(torch.amp, 'GradScaler'):
    class _GradScalerCompat(torch.cuda.amp.GradScaler):
        def __init__(self, device=None, **kwargs):
            super().__init__(**kwargs)
    torch.amp.GradScaler = _GradScalerCompat

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    # tensorboard not installed — fall back to a no-op writer. All the
    # eval metrics are persisted via the metrics.json write at the end of
    # main(); the SummaryWriter is only used to tee per-noise-level
    # scalars into a tensorboard run for inspection. On machines where
    # we just want the JSON dump, skipping tensorboard is fine.
    # NoOpSummaryWriter has no __init__; wrap to swallow the log_dir kwarg.
    from aig_nav_core.noop_writer import NoOpSummaryWriter as _NoOpBase

    class SummaryWriter(_NoOpBase):
        def __init__(self, *args, **kwargs):
            pass

from ablation_configs import get_variant

import sys
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from aig_nav_core import AIGNavTD3, PygameNavEnvAIGNav
from scripts.evaluate_aig_nav import (
    apply_prediction_noise,
    restore_prediction_noise,
    evaluate_noise_level,
    plot_ade_success_curve,
    parse_noise_list,
)
from scripts.config_utils import load_config, cfg_get
from scripts.eval_utils import DEFAULT_EVAL_SCENARIO_SEED, generate_eval_scenarios


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

    parser = argparse.ArgumentParser(description="Evaluate AIG-Nav-TD3 ablation variant")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--models-root", default="experiments/models/ablations")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--runs-root", default="experiments/results/ablations")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--episodes", type=int, default=cfg_get(cfg, "eval", "episodes", 20))
    parser.add_argument("--max-steps", type=int, default=cfg_get(cfg, "eval", "max_steps", 500))
    parser.add_argument("--pred-horizon", type=int, default=cfg_get(cfg, "env", "pred_horizon", 12))
    parser.add_argument("--history-length", type=int, default=cfg_get(cfg, "env", "history_length", 10))
    parser.add_argument("--num-pedestrians", type=int, default=cfg_get(cfg, "env", "num_pedestrians", 4))
    parser.add_argument("--pedestrian-reactivity", type=float, default=cfg_get(cfg, "env", "pedestrian_reactivity", 0.0))
    parser.add_argument("--num-candidate-actions", type=int, default=cfg_get(cfg, "model", "num_candidate_actions", 9))
    parser.add_argument("--aig-hidden-dim", type=int, default=cfg_get(cfg, "model", "aig_hidden_dim", 128))
    parser.add_argument("--aig-output-dim", type=int, default=cfg_get(cfg, "model", "aig_output_dim", 128))
    parser.add_argument("--use-iterative-eq", type=int, default=cfg_get(cfg, "iaig", "use_iterative_eq", 1))
    parser.add_argument("--K", type=int, default=cfg_get(cfg, "iaig", "K", 1))
    parser.add_argument("--intention-dim", type=int, default=cfg_get(cfg, "iaig", "intention_dim", 64))
    parser.add_argument("--lambda-conv", type=float, default=cfg_get(cfg, "iaig", "lambda_conv", 0.1))
    parser.add_argument("--noise-stds", default=cfg_get(cfg, "eval", "noise_stds", "0.0,0.05,0.1,0.2"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--near-miss-margin", type=float, default=cfg_get(cfg, "eval", "near_miss_margin", 0.3))
    parser.add_argument("--eval-scenario-seed", type=int, default=cfg_get(cfg, "eval", "eval_scenario_seed", DEFAULT_EVAL_SCENARIO_SEED))
    # Sim-to-real parameters
    parser.add_argument("--robot-radius", type=float, default=cfg_get(cfg, "env", "robot_radius", 0.25))
    parser.add_argument("--scan-fov", type=float, default=cfg_get(cfg, "env", "scan_fov", math.pi / 2))
    parser.add_argument("--scan-noise-std", type=float, default=cfg_get(cfg, "env", "scan_noise_std", 0.005))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = "cpu"

    set_seed(args.seed, deterministic=args.deterministic)

    variant = get_variant(args.variant)

    model_dir = Path(args.model_dir) if args.model_dir else Path(args.models_root) / args.variant / args.run_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    # Auto-detect best/ subdirectory
    if (model_dir / "best").exists():
        model_dir = model_dir / "best"

    model_name = args.model_name.strip() or f"AIGNavTD3_{args.variant}"

    env = PygameNavEnvAIGNav(
        render=args.render,
        num_pedestrians=args.num_pedestrians,
        pedestrian_reactivity=args.pedestrian_reactivity,
        seed=args.seed,
        history_length=args.history_length,
        prediction_horizon=args.pred_horizon,
        robot_radius=args.robot_radius,
        scan_fov=args.scan_fov,
        scan_noise_std=args.scan_noise_std,
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
        expl_noise=0.2,
        aig_hidden_dim=args.aig_hidden_dim,
        aig_output_dim=args.aig_output_dim,
        pred_horizon=args.pred_horizon,
        max_pedestrians=args.num_pedestrians + 2,
        history_length=args.history_length,
        max_static_obstacles=len(env.static_obstacles),
        goal_distance_scale=getattr(env, "max_goal_dist", 6.0),
        num_candidate_actions=args.num_candidate_actions,
        aux_pred_loss_weight=0.0,
        hidden_dim=256,
        hidden_depth=2,
        save_every=0,
        save_directory=model_dir,
        model_name=model_name,
        use_iterative_eq=bool(args.use_iterative_eq),
        K=args.K,
        intention_dim=args.intention_dim,
        lambda_conv=args.lambda_conv,
        robot_radius=args.robot_radius,
        **variant["params"],
    )

    model.load(model_dir)
    model.sync_inference()

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    if args.seed is not None and f"seed{args.seed}" not in run_name:
        run_name = f"{run_name}_seed{args.seed}"
    log_dir = Path(args.runs_root) / args.variant / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    noise_stds = parse_noise_list(args.noise_stds)
    eval_scenarios = generate_eval_scenarios(env, args.episodes, seed=args.eval_scenario_seed)

    ade_values = []
    success_values = []
    metrics_map = {}

    collision_dist = env.robot_radius + env.pedestrian_radius
    near_miss_dist = collision_dist + args.near_miss_margin

    base_seed = args.seed
    for idx, noise_std in enumerate(noise_stds):
        print(f"[EVAL] noise_std={noise_std:.3f}")

        set_seed(base_seed, deterministic=args.deterministic)
        noise_patches = apply_prediction_noise(model, noise_std, seed=idx)

        metrics = evaluate_noise_level(
            env,
            model,
            episodes=args.episodes,
            max_steps=args.max_steps,
            pred_horizon=args.pred_horizon,
            noise_std=noise_std,
            collision_dist=collision_dist,
            near_miss_dist=near_miss_dist,
            dt=env.dt,
            scenarios=eval_scenarios,
            seed_base=args.eval_scenario_seed,
        )

        restore_prediction_noise(noise_patches)

        step = int(noise_std * 1000)
        writer.add_scalar("eval_noise/ade", metrics["ade"], step)
        writer.add_scalar("eval_noise/fde", metrics["fde"], step)
        writer.add_scalar("eval_noise/brier", metrics["brier"], step)
        writer.add_scalar("eval_noise/ece", metrics["ece"], step)
        writer.add_scalar("eval_noise/success_rate", metrics["success_rate"], step)
        writer.add_scalar("eval_noise/collision_rate", metrics["collision_rate"], step)
        writer.add_scalar("eval_noise/avg_ttc", metrics["avg_ttc"], step)
        writer.add_scalar("eval_noise/min_separation", metrics["min_separation"], step)
        writer.add_scalar("eval_noise/near_miss", metrics["near_miss"], step)

        ade_values.append(metrics["ade"])
        success_values.append(metrics["success_rate"])
        metrics_map[str(noise_std)] = metrics

        print(
            f"  ADE={metrics['ade']:.3f} FDE={metrics['fde']:.3f} "
            f"Brier={metrics['brier']:.3f} ECE={metrics['ece']:.3f} "
            f"Success={metrics['success_rate']:.2f} Collision={metrics['collision_rate']:.2f} "
            f"TTC={metrics['avg_ttc']:.2f} MinSep={metrics['min_separation']:.2f} "
            f"NearMiss={metrics['near_miss']:.2f}"
        )

    fig = plot_ade_success_curve(ade_values, success_values)
    if fig is not None:
        writer.add_figure("correlation/ade_vs_success", fig, global_step=0)
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        writer.add_text(
            "correlation/ade_vs_success",
            "matplotlib not available; cannot render ADE vs Success plot",
            global_step=0,
        )

    metrics_path = log_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": args.variant,
                "model_dir": str(model_dir),
                "model_name": model_name,
                "seed": args.seed,
                "num_pedestrians": args.num_pedestrians,
                "pedestrian_reactivity": args.pedestrian_reactivity,
                "pred_horizon": args.pred_horizon,
                "history_length": args.history_length,
                "num_candidate_actions": args.num_candidate_actions,
                "aig_hidden_dim": args.aig_hidden_dim,
                "aig_output_dim": args.aig_output_dim,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "noise_stds": noise_stds,
                "metrics": metrics_map,
            },
            f,
            indent=2,
        )

    writer.flush()
    writer.close()
    env.close()
    print(f"[INFO] TensorBoard logs: {log_dir}")
    print(f"[INFO] Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
