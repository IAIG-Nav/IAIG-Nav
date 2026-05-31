#!/usr/bin/env python3
"""
Aggregate ablation results and print a LaTeX-ready summary table.

Usage (from repo root):
    /usr/bin/python3 ablations/aggregate_ablation_results.py
    /usr/bin/python3 ablations/aggregate_ablation_results.py \
        --results-root experiments/results/ablations \
        --baseline iaig_nav_td3
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


# ── Ordered display groups ────────────────────────────────────────────────────
GROUPS = [
    {
        "label": "(a) IEM refinement depth ($K$ sweep)",
        "rows": [
            ("K0",           "$K{=}0$ (AIG-Nav, discrete ActionSet)"),
            ("K1",           "$K{=}1$ (one refinement step)"),
            ("iaig_nav_td3", r"\method{} ($K{=}3$, full)"),
            ("K5",           "$K{=}5$ (five refinement steps)"),
        ],
    },
    {
        "label": "(b) Auxiliary training losses",
        "rows": [
            ("no_conv_loss", r"w/o $\mathcal{L}_{\text{conv}}$ ($\lambda_c{=}0$, No-Conv-Loss)"),
            ("no_aux_pred",  r"w/o $\mathcal{L}_{\text{pred}}$ ($\lambda_p{=}0$, No-Aux-Pred)"),
        ],
    },
    {
        "label": "(c) Graph structure",
        "rows": [
            ("fixed_graph",   r"w/o ped--ped edges (Fixed-Graph)"),
            ("no_reactive",   r"w/o reactive response model (No-Reactive)"),
            ("no_cond_graph", r"w/o action-conditioned graph (No-Cond-Graph)"),
            ("no_pred_edge",  r"w/o predicted-collision edges (No-Pred-Edge)"),
        ],
    },
]


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None
    mu = sum(values) / n
    if n < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / n   # population std (ddof=0)
    return mu, math.sqrt(var)


def load_variant_metrics(results_root: Path, variant: str, noise_std: float = 0.0):
    """
    Collect SR values for a given variant across all seed sub-directories.
    Returns a list of per-seed SR floats.
    """
    sr_values = []
    pattern = str(noise_std)
    variant_dir = results_root / variant
    if not variant_dir.is_dir():
        return sr_values
    for metrics_path in sorted(variant_dir.rglob("metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        metrics_map = data.get("metrics", {})
        for key, m in metrics_map.items():
            if abs(float(key) - noise_std) < 1e-6:
                sr = m.get("success_rate")
                if sr is not None:
                    sr_values.append(float(sr))
                break
    return sr_values


def fmt_sr(mu, std, bold=False):
    if mu is None:
        return "---"
    s = f"{mu:.3f}$\\pm${std:.3f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def fmt_delta(mu, ref_mu):
    if mu is None or ref_mu is None:
        return "---"
    delta_pp = (mu - ref_mu) * 100
    sign = "+" if delta_pp >= 0 else "\u2212"
    return f"${sign}{abs(delta_pp):.1f}\\,\\text{{pp}}$"


def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--results-root",
                        default="experiments/results/iaig_nav_td3_ablation/evals",
                        help="Root directory containing per-variant result dirs")
    parser.add_argument("--baseline",
                        default="iaig_nav_td3",
                        help="Variant name to use as the reference method")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Noise level to report (default: 0.0)")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"[WARN] Results root not found: {results_root}")

    # ── Load all variants ─────────────────────────────────────────────────────
    all_variants = set()
    for g in GROUPS:
        for vkey, _ in g["rows"]:
            all_variants.add(vkey)

    data = {}   # variant -> (mu, std, n, raw_list)
    for vkey in all_variants:
        sr_list = load_variant_metrics(results_root, vkey, args.noise_std)
        if sr_list:
            mu, std = mean_std(sr_list)
            data[vkey] = (mu, std, len(sr_list), sr_list)
        else:
            data[vkey] = (None, None, 0, [])

    ref_mu = data.get(args.baseline, (None,))[0]

    # ── Console table ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  IAIG-Nav TD3 Ablation Results  "
          f"(noise_std={args.noise_std}, baseline={args.baseline})")
    print("=" * 72)
    print(f"  {'Variant':<52} {'SR (mean±std)':<18} {'Δ (pp)':<10} N")
    print("-" * 72)

    for group in GROUPS:
        print(f"\n  {group['label']}")
        for vkey, label in group["rows"]:
            mu, std, n, _ = data.get(vkey, (None, None, 0, []))
            if mu is not None:
                sr_str = f"{mu:.3f}±{std:.3f}"
                if ref_mu is not None:
                    delta = (mu - ref_mu) * 100
                    dstr = f"{delta:+.1f}"
                else:
                    dstr = "---"
            else:
                sr_str = "pending"
                dstr = "---"
            # Strip LaTeX from label for console
            clean = label.replace("\\method{}", "IAIG-Nav").replace("$", "").replace("\\", "")
            print(f"  {clean:<52} {sr_str:<18} {dstr:<10} {n}")

    print()
    print("=" * 72)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("\n% ── LaTeX table (paste into paper) ──────────────────────────────")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Variant & SR & $\Delta$ \\")
    print(r"\midrule")

    best_mu = max((d[0] for d in data.values() if d[0] is not None), default=None)

    for group in GROUPS:
        print(r"\multicolumn{3}{l}{\textit{" + group["label"] + r"}} \\")
        for vkey, label in group["rows"]:
            mu, std, n, _ = data.get(vkey, (None, None, 0, []))
            is_bold = (mu is not None and best_mu is not None
                       and abs(mu - best_mu) < 1e-9)
            sr_col = fmt_sr(mu, std, bold=is_bold)
            delta_col = fmt_delta(mu, ref_mu) if vkey != args.baseline else "---"
            print(f"{label} & {sr_col} & {delta_col} \\\\")
        print(r"\midrule")

    # Remove trailing \midrule, replace with \bottomrule
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()

    # ── Pending summary ───────────────────────────────────────────────────────
    pending = [vkey for g in GROUPS for vkey, _ in g["rows"] if data.get(vkey, (None,))[0] is None]
    if pending:
        print(f"[INFO] Pending variants ({len(pending)}):", ", ".join(pending))
    else:
        print("[INFO] All variants complete — update paper table above.")


if __name__ == "__main__":
    main()
