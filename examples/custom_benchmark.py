"""Minimal custom-benchmark use without ORBIT or external downloads."""

import numpy as np

from saverouter import FixedKSparseRouter, simulate_fixed_k_supervision

rng = np.random.default_rng(42)
queries = rng.normal(size=(200, 16))
groups = (queries[:, 0] > 0).astype(int)
rewards = rng.random((200, 8))
costs = np.broadcast_to(np.linspace(0.1, 1.0, 8), rewards.shape).copy()

feedback = simulate_fixed_k_supervision(
    rewards, groups, costs=costs, k=4, seed=42
)
router = FixedKSparseRouter().fit(queries, feedback)
print(router.route(queries[:5], max_cost=0.5, group_ids=groups[:5]))
