import math

import numpy as np

from saverouter.evaluation import evaluate_predictions
from saverouter.metrics import supervision_amortized_metrics


def test_sa_metrics_match_definition():
    result = supervision_amortized_metrics(
        target_achieved=True,
        baseline_cost_per_request=1.0,
        routed_cost_per_request=0.6,
        supervision_cost=100.0,
        horizon=1000,
    )
    assert result["SA-BEP"] == 250
    assert result["SA-CR@1000"] == 0.7
    assert result["TC-BEP"] == result["SA-BEP"]


def test_sa_bep_is_infinite_when_target_is_missed():
    result = supervision_amortized_metrics(
        target_achieved=False,
        baseline_cost_per_request=1.0,
        routed_cost_per_request=0.1,
        supervision_cost=10.0,
    )
    assert math.isinf(result["SA-BEP"])
    assert result["SA-CR@1000000"] is None


def test_evaluation_uses_complete_paper_policy_family():
    predicted_quality = np.array([[0.7, 0.9], [0.8, 0.6]])
    predicted_cost = np.array([[0.1, 1.0], [0.1, 1.0]])
    result = evaluate_predictions(
        predicted_quality,
        predicted_cost,
        predicted_quality,
        predicted_cost,
        supervision_cost=1.0,
        global_predicted_cost=np.array([0.1, 1.0]),
        lambda_points=5,
        cost_threshold_points=2,
        gate_points=5,
    )
    protocol = result["evaluation_protocol"]
    assert protocol["name"] == "paper-full-frontier-v1"
    assert protocol["candidate_policies"] > protocol["lambda_points"]


def test_best_single_target_keeps_orbit_float_boundary():
    values = np.array(
        [0.04097352549433708, 0.016527635976672173], dtype=np.float32
    )
    quality = np.column_stack([values, np.zeros(2, dtype=np.float32)])
    costs = np.ones_like(quality)
    result = evaluate_predictions(
        quality,
        costs,
        quality,
        costs,
        supervision_cost=0.0,
        global_predicted_cost=np.ones(2),
        lambda_points=1,
        cost_threshold_points=1,
        gate_points=1,
    )
    assert result["quality_target"] > float(np.mean(values))
