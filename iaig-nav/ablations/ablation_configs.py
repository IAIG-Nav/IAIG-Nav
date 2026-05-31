"""
Ablation configuration registry for AIG-Nav.
"""

VARIANTS = {
    "iaig_nav": {
        "description": "IAIG-Nav: iterative equilibrium + full AIG features",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "iaig_nav_td3": {
        "description": "IAIG-Nav-TD3: iterative equilibrium + full AIG features (TD3 backbone)",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "aig_nav_td3": {
        "description": "AIG-Nav-TD3: original action-set aggregator (TD3 backbone)",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "aig_nav": {
        "description": "AIG-Nav (SAC): original action-set aggregator",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "full": {
        "description": "Alias for iaig_nav",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "no_cond_graph": {
        "description": "No-Cond-Graph: graph does not depend on robot action",
        "params": {
            "action_conditioned": False,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "no_reactive": {
        "description": "No-Reactive: pedestrian prediction ignores robot action",
        "params": {
            "action_conditioned": True,
            "reactive_model": False,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "no_pred_edge": {
        "description": "No-Pred-Edge: edges based on current distance, not predicted collisions",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": False,
            "use_ped_ped_edges": True,
        },
    },
    "fixed_graph": {
        "description": "Fixed-Graph: remove pedestrian-pedestrian edges",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": False,
        },
    },
    "no_aux_pred": {
        "description": "No-Aux-Pred: disable auxiliary trajectory prediction loss (lambda_pred=0)",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    # ── K-sweep variants ─────────────────────────────────────────────────
    "K0": {
        "description": "K=0: IEM with zero refinement steps (skip refinement loop; scene encoder + intention_init + output_proj only). NOTE: structurally distinct from AIG-Nav (which uses ActionSetAggregator+GATv2); this variant retains the IEM's own LSTM+MultiHeadAttention scene encoder.",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "K1": {
        "description": "K=1: single IEM refinement step (K=1, everything else full)",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "K3": {
        "description": "K=3: three IEM refinement steps",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    "K5": {
        "description": "K=5: five IEM refinement steps (K=5, everything else full)",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
    # ── Loss ablation ────────────────────────────────────────────────────
    "no_conv_loss": {
        "description": "No-Conv-Loss: disable convergence regularization (lambda_conv=0)",
        "params": {
            "action_conditioned": True,
            "reactive_model": True,
            "pred_edges": True,
            "use_ped_ped_edges": True,
        },
    },
}


def list_variants():
    return list(VARIANTS.keys())


def get_variant(name: str) -> dict:
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Available: {list_variants()}")
    return VARIANTS[name]
