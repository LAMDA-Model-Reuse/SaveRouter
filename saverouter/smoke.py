"""Small deterministic end-to-end check requiring no downloads."""

from __future__ import annotations

import numpy as np

from .evaluation import evaluate_predictions
from .router import FixedKSparseRouter
from .supervision import simulate_fixed_k_supervision


def run_smoke_test():
    rng = np.random.default_rng(42)
    n_train, n_test, n_models = 120, 60, 6
    train_x = rng.normal(size=(n_train, 12))
    test_x = rng.normal(size=(n_test, 12))
    train_groups = (train_x[:, 0] > 0).astype(int)
    test_groups = (test_x[:, 0] > 0).astype(int)
    ability = np.array([[0.82, 0.40, 0.60, 0.55, 0.48, 0.35], [0.38, 0.84, 0.62, 0.52, 0.45, 0.30]])
    train_quality = np.clip(ability[train_groups] + 0.04 * train_x[:, 1, None], 0, 1)
    test_quality = np.clip(ability[test_groups] + 0.04 * test_x[:, 1, None], 0, 1)
    prices = np.array([0.8, 0.9, 0.5, 0.35, 0.2, 0.1])
    train_cost = np.broadcast_to(prices, train_quality.shape).copy()
    test_cost = np.broadcast_to(prices, test_quality.shape).copy()
    supervision = simulate_fixed_k_supervision(
        train_quality, train_groups, costs=train_cost, k=2, seed=42
    )
    router = FixedKSparseRouter(min_model_observations=4).fit(train_x, supervision)
    result = evaluate_predictions(
        router.predict_scores(test_x, group_ids=test_groups),
        router.predict_costs(test_x, group_ids=test_groups),
        test_quality,
        test_cost,
        supervision_cost=float(supervision.costs.sum()),
        global_predicted_cost=router.global_costs_,
    )
    if supervision.n_observations != n_train * 2:
        raise AssertionError("Fixed-K acquisition count is incorrect.")
    return {
        "status": "PASS",
        "observations": supervision.n_observations,
        "density": supervision.density,
        "maximum_quality": result["maximum_quality"],
        "target_cost_ratio": result["target_cost_ratio"],
    }
