"""Pareto and supervision-amortized deployment metrics."""

from __future__ import annotations

from math import ceil, inf

import numpy as np


def pareto_front(points):
    """Return strictly improving physical cost-quality operating points."""
    ordered = sorted(
        ({"cost": float(p["cost"]), "quality": float(p.get("quality", p.get("performance")))} for p in points),
        key=lambda p: p["cost"],
    )
    result = []
    best = -inf
    for point in ordered:
        if point["quality"] > best:
            result.append(point)
            best = point["quality"]
    return result


def best_single(performance, costs):
    quality = np.asarray(performance, dtype=np.float64)
    price = np.asarray(costs, dtype=np.float64)
    if quality.ndim != 2 or price.shape != quality.shape:
        raise ValueError("performance and costs must be same-shaped matrices.")
    mean_quality = np.nanmean(quality, axis=0)
    model = int(np.nanargmax(mean_quality))
    return {
        "model_index": model,
        "quality": float(mean_quality[model]),
        "cost": float(np.nanmean(price[:, model])),
    }


def summarize_frontier(points, *, target_quality, baseline_cost):
    frontier = pareto_front(points)
    eligible = [p for p in frontier if p["quality"] >= float(target_quality)]
    target = min(eligible, key=lambda p: p["cost"]) if eligible else None
    return {
        "frontier": frontier,
        "maximum_quality": max(p["quality"] for p in frontier),
        "target_achieved": target is not None,
        "target_point": target,
        "target_cost_ratio": None if target is None else target["cost"] / float(baseline_cost),
    }


def supervision_amortized_metrics(
    *,
    target_achieved,
    baseline_cost_per_request,
    routed_cost_per_request,
    supervision_cost,
    horizon=1_000_000,
    judge_cost=0.0,
    training_cost=0.0,
    fixed_cost=0.0,
    router_overhead_per_request=0.0,
):
    """Compute SA-BEP and SA-CR@H.

    SA-BEP is finite only when the target is met and serving saves money.
    SA-CR amortizes all upfront costs over ``horizon`` requests.
    """
    baseline = float(baseline_cost_per_request)
    route = None if routed_cost_per_request is None else float(routed_cost_per_request)
    upfront = sum(map(float, (supervision_cost, judge_cost, training_cost, fixed_cost)))
    overhead = float(router_overhead_per_request)
    horizon = int(horizon)
    if baseline <= 0 or horizon <= 0 or upfront < 0 or overhead < 0:
        raise ValueError("Costs and horizon must define a positive deployment baseline.")
    if route is None or not np.isfinite(route):
        return {
            "SA-BEP": inf,
            f"SA-CR@{horizon}": None,
            "TC-BEP": inf,
            f"TCO-CR@{horizon}": None,
            "upfront_cost": upfront,
            "net_savings_per_request": None,
            "economically_eligible": False,
        }
    saving = baseline - route - overhead
    eligible = bool(target_achieved and saving > 0)
    bep = int(ceil(upfront / saving)) if eligible else inf
    ratio = (upfront + horizon * (route + overhead)) / (horizon * baseline)
    return {
        "SA-BEP": bep,
        f"SA-CR@{horizon}": ratio if target_achieved else None,
        "upfront_cost": upfront,
        "net_savings_per_request": saving,
        "economically_eligible": eligible,
        # Backward-compatible names used in the research workspace.
        "TC-BEP": bep,
        f"TCO-CR@{horizon}": ratio if target_achieved else None,
    }
