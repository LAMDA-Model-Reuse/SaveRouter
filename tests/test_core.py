import inspect

import numpy as np

from saverouter import FixedKSparseRouter, simulate_fixed_k_supervision


def test_training_api_accepts_sparse_supervision_only():
    parameters = list(inspect.signature(FixedKSparseRouter.fit).parameters)
    assert parameters == ["self", "X", "supervision"]


def test_unobserved_rewards_do_not_affect_acquisition():
    rng = np.random.default_rng(11)
    rewards = rng.random((40, 7))
    groups = np.repeat([0, 1, 2, 3], 10)
    first = simulate_fixed_k_supervision(rewards, groups, k=2, seed=19)
    changed = rewards.copy()
    changed[~first.observation_mask()] = 10_000.0
    second = simulate_fixed_k_supervision(changed, groups, k=2, seed=19)
    np.testing.assert_array_equal(first.query_indices, second.query_indices)
    np.testing.assert_array_equal(first.model_indices, second.model_indices)
    np.testing.assert_allclose(first.rewards, second.rewards)


def test_unobserved_costs_do_not_affect_router():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(40, 6))
    rewards = rng.random((40, 5))
    costs = rng.random((40, 5))
    groups = np.repeat([0, 1, 2, 3], 10)
    first = simulate_fixed_k_supervision(
        rewards, groups, costs=costs, k=2, seed=29
    )
    changed_costs = costs.copy()
    changed_costs[~first.observation_mask()] = 10_000.0
    second = simulate_fixed_k_supervision(
        rewards, groups, costs=changed_costs, k=2, seed=29
    )
    first_router = FixedKSparseRouter(min_model_observations=2).fit(features, first)
    second_router = FixedKSparseRouter(min_model_observations=2).fit(features, second)
    np.testing.assert_allclose(first_router.global_costs_, second_router.global_costs_)
    np.testing.assert_allclose(first_router.group_costs_, second_router.group_costs_)
