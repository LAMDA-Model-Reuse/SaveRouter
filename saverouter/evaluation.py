"""ORBIT-compatible evaluation over the paper's complete policy family."""

from __future__ import annotations

import numpy as np

from .metrics import best_single, summarize_frontier, supervision_amortized_metrics


def evaluate_predictions(
    predicted_quality,
    predicted_cost,
    true_quality,
    true_cost,
    *,
    supervision_cost,
    global_predicted_cost=None,
    quality_target_multiplier=1.0,
    horizon=1_000_000,
    lambda_points=200,
    cost_threshold_points=64,
    gate_points=100,
):
    predicted_quality = np.asarray(predicted_quality, dtype=np.float64)
    predicted_cost = np.asarray(predicted_cost, dtype=np.float64)
    # ORBIT evaluates selected outcomes in the benchmark's float32 matrices,
    # while the best-single target is computed from float64 pandas means.
    # Preserve the input dtype here; ``best_single`` performs its own float64
    # conversion. This distinction matters for exact target-boundary ties.
    true_quality = np.asarray(true_quality)
    true_cost = np.asarray(true_cost)
    if not (
        predicted_quality.shape
        == predicted_cost.shape
        == true_quality.shape
        == true_cost.shape
    ):
        raise ValueError("All prediction and outcome matrices must have equal shape.")
    available = np.isfinite(true_quality) & np.isfinite(true_cost)
    predicted_available = available & np.isfinite(predicted_quality) & np.isfinite(predicted_cost)
    if np.any(~predicted_available.any(axis=1)):
        raise ValueError("Every test query needs at least one usable candidate.")

    if min(int(lambda_points), int(cost_threshold_points), int(gate_points)) <= 0:
        raise ValueError("Policy grid sizes must be positive.")
    if global_predicted_cost is None:
        global_cost = np.nanmean(
            np.where(np.isfinite(predicted_cost), predicted_cost, np.nan), axis=0
        )
    else:
        global_cost = np.asarray(global_predicted_cost, dtype=np.float64)
    if global_cost.shape != (predicted_quality.shape[1],):
        raise ValueError("global_predicted_cost must contain one cost per model.")
    if not np.all(np.isfinite(global_cost)) or np.any(global_cost < 0):
        raise ValueError("global_predicted_cost must contain finite non-negative costs.")

    rows = np.arange(len(true_quality))
    points = []

    def add_policy(choices):
        points.append({
            "cost": float(np.mean(true_cost[rows, choices])),
            "quality": float(np.mean(true_quality[rows, choices])),
        })

    # These three policy families and their grids match the main-table ORBIT
    # implementation. Omitting the lambda or incremental-gate sweep can miss
    # the published peak and target-quality points.
    scores = np.where(predicted_available, predicted_quality, -np.inf)
    costs = np.where(predicted_available, predicted_cost, np.inf)
    cost_scale = max(float(np.max(global_cost)), 1e-12)
    normalized_cost = costs / cost_scale
    lambdas = np.r_[0.0, np.logspace(-4.0, 3.0, int(lambda_points))]
    for penalty in lambdas:
        utility = scores if penalty == 0.0 else scores - penalty * normalized_cost
        add_policy(np.argmax(utility, axis=1))

    high_choices = np.argmax(scores, axis=1)
    thresholds = np.sort(np.unique(global_cost))
    if len(thresholds) > int(cost_threshold_points):
        thresholds = np.unique(
            np.quantile(
                thresholds,
                np.linspace(0.0, 1.0, int(cost_threshold_points)),
            )
        )
    for threshold in thresholds:
        allowed = global_cost <= threshold
        fallback = np.argmax(
            np.where(allowed[None, :], scores, -np.inf), axis=1
        )
        add_policy(fallback)
        benefit = scores[rows, high_choices] - scores[rows, fallback]
        incremental_cost = np.maximum(
            costs[rows, high_choices] - costs[rows, fallback], 1e-8
        )
        gain_per_cost = benefit / incremental_cost
        gates = np.unique(
            np.quantile(gain_per_cost, np.linspace(0.0, 1.0, int(gate_points)))
        )
        for gate in gates:
            choices = fallback.copy()
            choices[gain_per_cost >= gate] = high_choices[gain_per_cost >= gate]
            add_policy(choices)
    add_policy(high_choices)

    baseline = best_single(true_quality, true_cost)
    target_quality = float(quality_target_multiplier) * baseline["quality"]
    summary = summarize_frontier(
        points,
        target_quality=target_quality,
        baseline_cost=baseline["cost"],
    )
    route_cost = None if summary["target_point"] is None else summary["target_point"]["cost"]
    economics = supervision_amortized_metrics(
        target_achieved=summary["target_achieved"],
        baseline_cost_per_request=baseline["cost"],
        routed_cost_per_request=route_cost,
        supervision_cost=supervision_cost,
        horizon=horizon,
    )
    return {
        "best_single": baseline,
        "quality_target": target_quality,
        "maximum_quality": summary["maximum_quality"],
        "target_achieved": summary["target_achieved"],
        "target_cost_ratio": summary["target_cost_ratio"],
        "SA-BEP": economics["SA-BEP"],
        f"SA-CR@{horizon}": economics[f"SA-CR@{horizon}"],
        "TC-BEP": economics["TC-BEP"],
        f"TCO-CR@{horizon}": economics[f"TCO-CR@{horizon}"],
        "supervision_cost": float(supervision_cost),
        "evaluation_protocol": {
            "name": "paper-full-frontier-v1",
            "lambda_points": int(lambda_points),
            "cost_threshold_points": int(cost_threshold_points),
            "gate_points": int(gate_points),
            "candidate_policies": len(points),
        },
        "frontier": summary["frontier"],
    }
