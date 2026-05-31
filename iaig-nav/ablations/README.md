# Ablations — IAIG-Nav

Ablation studies for IAIG-Nav. All variants share the **TD3 backbone** with the
main experiment; each variant flips a single design switch.

---

## Variants

### (a) IEM iteration depth (K-sweep)

| Variant   | Description                            | Difference vs. main method |
| --------- | -------------------------------------- | -------------------------- |
| `K0`      | IEM with refinement loop disabled      | `K=0`                      |
| *(K=1)*   | **Main method** (reference)            | —                          |
| `K3`      | IEM runs 3 rounds                      | `K=3`                      |
| `K5`      | IEM runs 5 rounds                      | `K=5`                      |

> The main method uses K=1 (single IEM refinement step). K=3 and K=5 probe
> whether deeper iteration helps; K=0 skips the refinement loop entirely (the
> IEM keeps its LSTM + attention scene encoder, only the iteration block is
> bypassed).
>
> Note: K=0 is **architecturally distinct from AIG-Nav** — the predecessor
> design uses an entirely different aggregator (ActionSetAggregator + GATv2),
> not an IEM with K=0.

### (b) Auxiliary training losses

| Variant         | Description                                   | Difference vs. full model |
| --------------- | --------------------------------------------- | ------------------------- |
| `no_conv_loss`  | Disable convergence regularizer               | `λ_conv = 0`              |
| `no_aux_pred`   | Disable auxiliary trajectory prediction loss  | `λ_pred = 0`              |

### (c) Graph structure

| Variant         | Description                                                  | Difference vs. full model    |
| --------------- | ------------------------------------------------------------ | ---------------------------- |
| `fixed_graph`   | Remove pedestrian–pedestrian edges                           | `use_ped_ped_edges=False`    |
| `no_reactive`   | Pedestrian prediction ignores robot action                   | `reactive_model=False`       |
| `no_cond_graph` | Graph structure does not depend on robot action              | `action_conditioned=False`   |
| `no_pred_edge`  | Edge weights use current distance, not predicted collision   | `pred_edges=False`           |

---

## Scripts

| Script                              | Purpose                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| `ablation_configs.py`               | Variant registry (boolean kwargs → `AIGNavTD3`)          |
| `train_ablation_td3.py`             | Unified TD3 training script (`--variant` selects)        |
| `evaluate_ablation_td3.py`          | Unified TD3 evaluation script                            |
| `aggregate_ablation_results.py`     | Aggregate multi-seed results, print table + LaTeX       |

---

## Usage

Run from the repository root:

```bash
# Train one variant / one seed
python -m ablations.train_ablation_td3 --variant K0 --seed 0
python -m ablations.train_ablation_td3 --variant K3 --seed 1
python -m ablations.train_ablation_td3 --variant K5 --seed 2
python -m ablations.train_ablation_td3 --variant no_aux_pred --seed 0

# Evaluate
python -m ablations.evaluate_ablation_td3 --variant K5 --seed 0

# Sweep all variants × seeds with a shell loop
for v in K0 K3 K5 no_conv_loss no_aux_pred fixed_graph no_reactive no_cond_graph no_pred_edge; do
  for s in 0 1 2; do
    python -m ablations.train_ablation_td3 --variant "$v" --seed "$s"
  done
done
```

---

## Output paths

| Type                          | Path                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| Best model (saved during run) | `experiments/models/iaig_nav_td3_ablation/<variant>/seed<N>/best/`                   |
| Final model                   | `experiments/models/iaig_nav_td3_ablation/<variant>/seed<N>/final/`                  |
| TensorBoard logs              | `experiments/runs/iaig_nav_td3_ablation/<variant>/seed<N>/`                          |
| Evaluation metrics            | `experiments/results/iaig_nav_td3_ablation/evals/<variant>/seed<N>_*/metrics.json`   |

---

## Aggregating results

After training and evaluation:

```bash
python -m ablations.aggregate_ablation_results \
  --results-root experiments/results/iaig_nav_td3_ablation/evals
```

Outputs a console summary table (Variant / SR mean ± std / Δpp) plus a LaTeX
snippet ready to paste into a paper.

---

## Hyperparameters

All variants share the same hyperparameters as the main experiment; only one
switch differs per variant.

| Parameter                        | Value      |
| -------------------------------- | ---------- |
| `total_steps`                    | 1,000,000  |
| `num_pedestrians`                | 4          |
| `pedestrian_reactivity`          | 0.3        |
| `K` (full model)                 | 1          |
| `lambda_conv`                    | 0.1        |
| `aux_pred_loss_weight`           | 0.5        |
| seeds                            | [0, 1, 2]  |
| `eval_scenario_seed` (training)  | 12345      |
| `eval_scenario_seed` (final)     | 54321      |
| episodes (final eval)            | 100        |
