"""
Evaluation script for AIG-Nav.

Computes:
- Prediction metrics: ADE/FDE, Brier, ECE
- Safety metrics: TTC, minimum separation, near-miss count
- Correlation curve: ADE vs success rate

Logs metrics to TensorBoard with labeled axes for plots.
"""

import argparse
import math
import time
import random
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    # tensorboard not installed — metrics are persisted via metrics.json;
    # the writer is only used for scalar tees.
    from aig_nav_core.noop_writer import NoOpSummaryWriter as _NoOpBase

    class SummaryWriter(_NoOpBase):
        def __init__(self, *args, **kwargs):
            pass

import sys as _sys
_repo_root = Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_repo_root))

from aig_nav_core import AIGNav, PygameNavEnvAIGNav
from aig_nav_core.trajectory_predictor import compute_collision_probability, predict_robot_trajectory
from scripts.config_utils import load_config, cfg_get
from scripts.eval_utils import DEFAULT_EVAL_SCENARIO_SEED, generate_eval_scenarios, set_global_seeds

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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


def world_to_robot_at(pose: Tuple[float, float, float], points: np.ndarray) -> np.ndarray:
    """Transform world points to robot frame at the given pose."""
    x, y, yaw = pose
    dx = points[:, 0] - x
    dy = points[:, 1] - y
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rx = c * dx - s * dy
    ry = s * dx + c * dy
    return np.stack([rx, ry], axis=-1)


def compute_ece(probs: List[float], labels: List[float], num_bins: int = 10) -> float:
    """Expected calibration error for binary events."""
    if not probs:
        return 0.0
    probs_np = np.asarray(probs, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.float32)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        if i == num_bins - 1:
            mask = (probs_np >= bins[i]) & (probs_np <= bins[i + 1])
        else:
            mask = (probs_np >= bins[i]) & (probs_np < bins[i + 1])
        if not np.any(mask):
            continue
        avg_conf = float(probs_np[mask].mean())
        avg_acc = float(labels_np[mask].mean())
        ece += (mask.mean()) * abs(avg_conf - avg_acc)
    return float(ece)


def _make_noise_like(noise_std: float, seed: int):
    generators: Dict[str, torch.Generator] = {}

    def sample_like(ref: torch.Tensor) -> torch.Tensor:
        if noise_std <= 0.0 or ref.numel() == 0:
            return torch.zeros_like(ref)

        key = str(ref.device)
        gen = generators.get(key)
        if gen is None:
            gen = torch.Generator(device=ref.device) if ref.device.type == "cuda" else torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            generators[key] = gen

        return torch.randn(ref.shape, dtype=ref.dtype, device=ref.device, generator=gen) * noise_std

    return sample_like


def apply_prediction_noise(model: AIGNav, noise_std: float, seed: int):
    """
    Inject Gaussian noise into prediction path for robustness evaluation.
    For IAIG mode, also injects noise in iter-eq response prediction so noise
    affects the actor decision path.
    """
    patches: List[Tuple[Any, str, Any]] = []
    sample_like = _make_noise_like(noise_std, seed)

    # Always perturb encoder trajectory predictions (used by ADE/FDE/Brier metrics).
    encoder = model.aig_encoder_infer
    original_predict = encoder.predict_ped_trajectories

    def noisy_predict(*args, **kwargs):
        preds = original_predict(*args, **kwargs)
        return preds + sample_like(preds)

    encoder.predict_ped_trajectories = noisy_predict
    patches.append((encoder, "predict_ped_trajectories", original_predict))

    # In IAIG mode, actor decisions do not use encoder.predict_ped_trajectories.
    # Patch iter_eq_module_infer response prediction to affect decision robustness.
    if getattr(model, "use_iterative_eq", False) and getattr(model, "iter_eq_module_infer", None) is not None:
        iter_eq_module = model.iter_eq_module_infer
        original_response = iter_eq_module._predict_ped_response

        def noisy_predict_response(*args, **kwargs):
            collision_probs, delta_positions = original_response(*args, **kwargs)
            noisy_collision = torch.clamp(collision_probs + sample_like(collision_probs), 0.0, 1.0)
            noisy_delta = delta_positions + sample_like(delta_positions)
            return noisy_collision, noisy_delta

        iter_eq_module._predict_ped_response = noisy_predict_response
        patches.append((iter_eq_module, "_predict_ped_response", original_response))

    return patches


def restore_prediction_noise(patches):
    for obj, attr_name, original_fn in patches:
        setattr(obj, attr_name, original_fn)


def get_world_state(env: PygameNavEnvAIGNav) -> Dict:
    ped_world = np.array([[p.x, p.y] for p in env.pedestrians], dtype=np.float32)
    return {
        "robot_pose": (env.x, env.y, env.yaw),
        "robot_pos": np.array([env.x, env.y], dtype=np.float32),
        "ped_world": ped_world,
    }


def finalize_prediction(
    pred_item: Dict,
    state_history: List[Dict],
    pred_horizon: int,
    collision_dist: float,
) -> Tuple[float, int, float, int, float, int, List[float], List[float]]:
    start = pred_item["step"]
    pred_trajs = pred_item["pred_trajs"]  # [N, H, 2]
    pred_coll = pred_item["pred_coll"]  # [N]
    robot_pose = pred_item["robot_pose"]

    if pred_trajs.size == 0:
        return 0.0, 0, 0.0, 0, 0.0, 0, [], []

    actual_list = []
    robot_positions = []
    ped_positions = []

    for k in range(1, pred_horizon + 1):
        state = state_history[start + k]
        ped_world = state["ped_world"]
        actual_list.append(world_to_robot_at(robot_pose, ped_world))
        robot_positions.append(state["robot_pos"])
        ped_positions.append(ped_world)

    actual_trajs = np.stack(actual_list, axis=1)  # [N, H, 2]
    diff = pred_trajs - actual_trajs

    ade_sum = float(np.linalg.norm(diff, axis=-1).sum())
    ade_count = int(diff.shape[0] * diff.shape[1])
    fde_sum = float(np.linalg.norm(diff[:, -1, :], axis=-1).sum())
    fde_count = int(diff.shape[0])

    robot_positions = np.stack(robot_positions, axis=0)  # [H, 2]
    ped_positions = np.stack(ped_positions, axis=0)  # [H, N, 2]
    distances = np.linalg.norm(ped_positions - robot_positions[:, None, :], axis=-1)  # [H, N]
    labels = (distances < collision_dist).any(axis=0).astype(np.float32)  # [N]

    brier_sum = float(((pred_coll - labels) ** 2).sum())
    brier_count = int(labels.shape[0])

    return ade_sum, ade_count, fde_sum, fde_count, brier_sum, brier_count, pred_coll.tolist(), labels.tolist()


def compute_safety_metrics(
    state_history: List[Dict],
    collision_dist: float,
    near_miss_dist: float,
    pred_horizon: int,
    dt: float,
) -> Tuple[float, float, int]:
    min_sep = float("inf")
    near_miss_count = 0
    ttc_values = []

    total_steps = len(state_history)
    for t in range(total_steps):
        robot_pos = state_history[t]["robot_pos"]
        ped_world = state_history[t]["ped_world"]
        if ped_world.size == 0:
            continue
        distances = np.linalg.norm(ped_world - robot_pos[None, :], axis=-1)
        step_min = float(distances.min())
        min_sep = min(min_sep, step_min)
        if step_min < near_miss_dist and step_min >= collision_dist:
            near_miss_count += 1

        ttc = pred_horizon * dt
        max_k = min(pred_horizon, total_steps - t - 1)
        for k in range(1, max_k + 1):
            future_robot = state_history[t + k]["robot_pos"]
            future_peds = state_history[t + k]["ped_world"]
            if future_peds.size == 0:
                continue
            future_dist = np.linalg.norm(future_peds - future_robot[None, :], axis=-1).min()
            if future_dist < collision_dist:
                ttc = k * dt
                break
        ttc_values.append(ttc)

    avg_ttc = float(np.mean(ttc_values)) if ttc_values else 0.0
    min_sep = float(min_sep) if min_sep != float("inf") else 0.0
    return avg_ttc, min_sep, near_miss_count


def evaluate_noise_level(
    env: PygameNavEnvAIGNav,
    model: AIGNav,
    episodes: int,
    max_steps: int,
    pred_horizon: int,
    noise_std: float,
    collision_dist: float,
    near_miss_dist: float,
    dt: float,
    scenarios: List[Tuple[float, float, float, float, float]],
    seed_base: int,
) -> Dict:
    ade_sum = 0.0
    ade_count = 0
    fde_sum = 0.0
    fde_count = 0
    brier_sum = 0.0
    brier_count = 0
    prob_list = []
    label_list = []

    success_count = 0
    collision_count = 0
    ttc_values = []
    min_sep_values = []
    near_miss_values = []

    for ep in range(episodes):
        seed = seed_base + ep
        set_global_seeds(seed)
        obs = env.reset(scenario=scenarios[ep], eval_seed=seed)
        model.reset_episode()

        step = 0
        done = False
        state_history: List[Dict] = []
        pending = deque()

        while step < max_steps and not done:
            state_history.append(get_world_state(env))

            while pending and (pending[0]["step"] + pred_horizon) < len(state_history):
                pred_item = pending.popleft()
                result = finalize_prediction(pred_item, state_history, pred_horizon, collision_dist)
                ade_sum += result[0]
                ade_count += result[1]
                fde_sum += result[2]
                fde_count += result[3]
                brier_sum += result[4]
                brier_count += result[5]
                prob_list.extend(result[6])
                label_list.extend(result[7])

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

            if len(ped_pos) > 0:
                ped_pos_t = torch.as_tensor(ped_pos, dtype=torch.float32, device=model.device).unsqueeze(0)
                ped_vel_t = torch.as_tensor(ped_vel, dtype=torch.float32, device=model.device).unsqueeze(0)
                ped_hist_t = torch.as_tensor(ped_hist, dtype=torch.float32, device=model.device).unsqueeze(0)
                ped_mask_t = torch.ones(1, ped_pos_t.size(1), dtype=torch.bool, device=model.device)
            else:
                ped_pos_t = torch.zeros(1, 0, 2, device=model.device)
                ped_vel_t = torch.zeros(1, 0, 2, device=model.device)
                ped_hist_t = torch.zeros(1, 0, model.aig_encoder.history_length, 4, device=model.device)
                ped_mask_t = torch.zeros(1, 0, dtype=torch.bool, device=model.device)

            action_t = torch.as_tensor(action, dtype=torch.float32, device=model.device).unsqueeze(0)

            pred_trajs = model.aig_encoder_infer.predict_ped_trajectories(
                ped_pos_t,
                ped_vel_t,
                ped_hist_t,
                action_t,
                ped_mask=ped_mask_t,
            )

            robot_traj = predict_robot_trajectory(
                torch.zeros(1, 2, device=model.device),
                torch.zeros(1, device=model.device),
                action_t,
                pred_horizon=pred_horizon,
                dt=dt,
            )

            pred_coll = compute_collision_probability(
                robot_traj,
                pred_trajs,
                robot_radius=env.robot_radius,
                ped_radius=env.pedestrian_radius,
                safety_margin=env.pedestrian_radius * 0.0 + 0.2,
                ped_mask=ped_mask_t,
            )

            pending.append(
                {
                    "step": step,
                    "pred_trajs": pred_trajs.squeeze(0).detach().cpu().numpy(),
                    "pred_coll": pred_coll.squeeze(0).detach().cpu().numpy(),
                    "robot_pose": state_history[-1]["robot_pose"],
                }
            )

            lin_vel = float(action[0]) if action[0] >= 0 else float(action[0] * 0.3)
            ang_vel = float(action[1])
            obs = env.step(lin_vel, ang_vel)

            step += 1
            done = bool(obs["collision"] or obs["goal_reached"])

        state_history.append(get_world_state(env))
        while pending and (pending[0]["step"] + pred_horizon) < len(state_history):
            pred_item = pending.popleft()
            result = finalize_prediction(pred_item, state_history, pred_horizon, collision_dist)
            ade_sum += result[0]
            ade_count += result[1]
            fde_sum += result[2]
            fde_count += result[3]
            brier_sum += result[4]
            brier_count += result[5]
            prob_list.extend(result[6])
            label_list.extend(result[7])

        success = bool(obs["goal_reached"])
        collision = bool(obs["collision"])
        success_count += int(success)
        collision_count += int(collision)

        avg_ttc, min_sep, near_miss = compute_safety_metrics(
            state_history,
            collision_dist=collision_dist,
            near_miss_dist=near_miss_dist,
            pred_horizon=pred_horizon,
            dt=dt,
        )
        ttc_values.append(avg_ttc)
        min_sep_values.append(min_sep)
        near_miss_values.append(near_miss)

    ade = ade_sum / max(ade_count, 1)
    fde = fde_sum / max(fde_count, 1)
    brier = brier_sum / max(brier_count, 1)
    ece = compute_ece(prob_list, label_list, num_bins=10)

    metrics = {
        "ade": ade,
        "fde": fde,
        "brier": brier,
        "ece": ece,
        "success_rate": success_count / max(episodes, 1),
        "collision_rate": collision_count / max(episodes, 1),
        "avg_ttc": float(np.mean(ttc_values)) if ttc_values else 0.0,
        "min_separation": float(np.mean(min_sep_values)) if min_sep_values else 0.0,
        "near_miss": float(np.mean(near_miss_values)) if near_miss_values else 0.0,
    }
    return metrics


def plot_ade_success_curve(ade_values: List[float], success_values: List[float]):
    if plt is None:
        return None
    pairs = sorted(zip(ade_values, success_values), key=lambda x: x[0])
    ade_sorted = [p[0] for p in pairs]
    success_sorted = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ade_sorted, success_sorted, marker="o", linewidth=2)
    ax.set_xlabel("ADE (m)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Prediction Accuracy vs Navigation Success")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def parse_noise_list(noise_arg: str) -> List[float]:
    return [float(x.strip()) for x in noise_arg.split(",") if x.strip()]


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="experiments/configs/defaults.json")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Evaluate AIG-Nav with predictive metrics")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--model-dir", default="experiments/models/aig_nav_sac")
    parser.add_argument("--model-name", default="AIGNav_v1")
    parser.add_argument("--log-dir", default="experiments/results/aig_nav_sac_eval")
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
    parser.add_argument("--K", type=int, default=cfg_get(cfg, "iaig", "K", 3))
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

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    if args.seed is not None and f"seed{args.seed}" not in run_name:
        run_name = f"{run_name}_seed{args.seed}"

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

    model = AIGNav(
        scan_dim=scan_dim,
        core_dim=core_dim,
        action_dim=action_dim,
        device=args.device,
        max_action=1.0,
        discount=0.99,
        init_temperature=0.2,
        alpha_lr=3e-5,
        actor_lr=3e-4,
        critic_lr=3e-4,
        critic_tau=0.005,
        actor_update_frequency=1,
        critic_target_update_frequency=2,
        learnable_temperature=True,
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
        save_directory=Path(args.model_dir),
        model_name=args.model_name,
        # === IAIG parameters ===
        use_iterative_eq=bool(args.use_iterative_eq),
        K=args.K,
        intention_dim=args.intention_dim,
        lambda_conv=args.lambda_conv,
        robot_radius=args.robot_radius,
    )

    model_dir = Path(args.model_dir)
    candidate = model_dir / run_name
    if (candidate / f"{args.model_name}_actor.pth").exists():
        model_dir = candidate
    # Auto-detect best/ subdirectory
    if (model_dir / "best" / f"{args.model_name}_actor.pth").exists():
        model_dir = model_dir / "best"
    if not (model_dir / f"{args.model_name}_actor.pth").exists():
        raise FileNotFoundError(
            f"Model weights not found in {model_dir}. "
            "Please pass --model-dir to a specific run folder."
        )
    model.load(model_dir)
    model.sync_inference()

    log_dir = Path(args.log_dir)
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "model_dir": str(model_dir)}, f, indent=2)

    noise_stds = parse_noise_list(args.noise_stds)
    eval_scenarios = generate_eval_scenarios(env, args.episodes, seed=args.eval_scenario_seed)

    ade_values = []
    success_values = []
    metrics_map = {}

    collision_dist = env.robot_radius + env.pedestrian_radius
    near_miss_dist = collision_dist + args.near_miss_margin

    for idx, noise_std in enumerate(noise_stds):
        print(f"[EVAL] noise_std={noise_std:.3f}")

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
        plt.close(fig)
    else:
        writer.add_text(
            "correlation/ade_vs_success",
            "matplotlib not available; cannot render ADE vs Success plot",
            global_step=0,
        )

    writer.flush()
    writer.close()
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": "aig_nav_sac",
                "model_dir": str(model_dir),
                "model_name": args.model_name,
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
    env.close()
    print(f"[INFO] TensorBoard logs: {run_dir}")
    print(f"[INFO] Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
