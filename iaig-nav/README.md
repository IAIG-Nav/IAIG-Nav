# IAIG-Nav

**Iterative Action-conditioned Interaction Graph Navigation** — deep reinforcement learning for social robot navigation with K-step iterative equilibrium reasoning.

## Highlights

- **Iterative Equilibrium Module (IEM):** the actor refines its action context through K rounds of self-prediction, letting the policy "think before acting" about how pedestrians will react.
- **Action-conditioned Interaction Graph (AIG):** robot–pedestrian interactions are encoded as a graph whose edges depend on the planned action, captured with GATv2.
- **TD3 backbone (main method)** with an SAC variant kept as an appendix implementation.
- **Pygame simulator with reactive pedestrians** — no ROS / Gazebo required for training.

For double-anonymous review, author-identifying information has been removed.

Supplementary video: https://youtu.be/jjJO04RPb8k

## Demo

![IAIG-Nav demo](assets/demo.gif)

## Installation

```bash
cd iaig-nav
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, CUDA 11.8.

## Quickstart

Visualize the simulator (no checkpoint required):

```bash
python -m scripts.test_aig_nav_render
```

Train the main method:

```bash
python -m scripts.train_aig_nav_td3 \
  --config experiments/configs/iaig_nav_td3.json \
  --seed 0
```

Evaluate a trained checkpoint:

```bash
python -m scripts.evaluate_aig_nav_td3 \
  --model-dir <path-to-checkpoint>
```

Run ablations:

```bash
# K-step depth (K0 = no refinement, K1 = main, K3, K5)
python -m ablations.train_ablation_td3 --variant K0 --seed 0
python -m ablations.train_ablation_td3 --variant K3 --seed 0
python -m ablations.train_ablation_td3 --variant K5 --seed 0

# Component ablations
python -m ablations.train_ablation_td3 --variant no_conv_loss --seed 0
python -m ablations.train_ablation_td3 --variant no_aux_pred --seed 0
python -m ablations.train_ablation_td3 --variant fixed_graph --seed 0
python -m ablations.train_ablation_td3 --variant no_reactive --seed 0
python -m ablations.train_ablation_td3 --variant no_cond_graph --seed 0
python -m ablations.train_ablation_td3 --variant no_pred_edge --seed 0
```

See [`ablations/README.md`](ablations/README.md) for the full ablation matrix.

## Repository layout

```
aig_nav_core/         core algorithm (encoder, IEM, TD3, SAC, pygame env, predictors)
scripts/              training / evaluation entry points
ablations/            K-value and component ablations
experiments/configs/  training configurations (JSON)
```


## License

Released under the [MIT License](LICENSE).
