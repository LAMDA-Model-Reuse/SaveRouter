"""Leakage-resistant fixed-K feedback acquisition."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from .data import SparseSupervision


Feedback = Union[float, Tuple[float, float]]
RevealCallback = Callable[[int, int], Feedback]


def collect_fixed_k_supervision(
    *,
    n_queries: int,
    n_models: int,
    group_ids: Sequence[int],
    reveal: RevealCallback,
    k: int,
    available: Optional[np.ndarray] = None,
    beta: float = 0.35,
    prior_strength: float = 10.0,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    seed: int = 42,
    seed_stride: int = 1009,
    reset_each_pass: bool = True,
    query_ids: Optional[Sequence[str]] = None,
    model_ids: Optional[Sequence[str]] = None,
) -> SparseSupervision:
    """Choose exactly ``k`` distinct models per query and reveal only those.

    The callback is invoked *after* selection and exactly ``n_queries * k``
    times. It may return ``reward`` or ``(reward, cost)``.  This API can call
    live models/evaluators; an unselected query/model pair is never requested.

    ``reset_each_pass=True`` reproduces the original ORBIT multi-pass policy.
    Setting it to false carries evidence across passes, which can be useful in
    streaming deployments but is a distinct ablation.
    """
    n_queries = int(n_queries)
    n_models = int(n_models)
    k = int(k)
    groups = np.asarray(group_ids, dtype=np.int64)
    if n_queries <= 0 or n_models <= 0:
        raise ValueError("n_queries and n_models must be positive.")
    if groups.shape != (n_queries,) or np.any(groups < 0):
        raise ValueError("group_ids must be non-negative with one value per query.")
    if k <= 0 or k > n_models:
        raise ValueError("k must be in [1, n_models].")
    if beta < 0 or min(prior_strength, prior_alpha, prior_beta) <= 0:
        raise ValueError("beta must be non-negative and prior parameters positive.")
    if available is None:
        availability = np.ones((n_queries, n_models), dtype=bool)
    else:
        availability = np.asarray(available, dtype=bool)
        if availability.shape != (n_queries, n_models):
            raise ValueError("available must have shape (n_queries, n_models).")
    if np.any(availability.sum(axis=1) < k):
        raise ValueError("Every query must have at least k available models.")

    n_groups = int(groups.max()) + 1
    selected = np.zeros((n_queries, n_models), dtype=bool)
    query_log = []
    model_log = []
    reward_log = []
    cost_log = []
    returns_cost = None

    persistent = None
    for pass_index in range(k):
        rng = np.random.default_rng(seed + seed_stride * pass_index)
        if persistent is None or reset_each_pass:
            group_sum = np.zeros((n_groups, n_models), dtype=np.float64)
            group_count = np.zeros((n_groups, n_models), dtype=np.int64)
            global_sum = np.zeros(n_models, dtype=np.float64)
            global_count = np.zeros(n_models, dtype=np.int64)
        else:
            group_sum, group_count, global_sum, global_count = persistent

        for query_index in rng.permutation(n_queries):
            group_index = int(groups[query_index])
            candidates = availability[query_index] & ~selected[query_index]
            unseen = np.flatnonzero(candidates & (group_count[group_index] == 0))
            if unseen.size:
                model_index = int(rng.choice(unseen))
            else:
                global_scores = (global_sum + prior_alpha) / (
                    global_count + prior_alpha + prior_beta
                )
                posterior = (
                    group_sum[group_index] + prior_strength * global_scores
                ) / (group_count[group_index] + prior_strength)
                bonus = beta * np.sqrt(
                    np.log(float(group_count[group_index].sum()) + 1.0)
                    / np.maximum(group_count[group_index], 1)
                )
                scores = np.where(candidates, posterior + bonus, -np.inf)
                model_index = int(np.argmax(scores + rng.random(n_models) * 1e-12))

            feedback = reveal(int(query_index), model_index)
            if isinstance(feedback, tuple):
                if len(feedback) != 2:
                    raise ValueError("reveal tuples must contain exactly (reward, cost).")
                reward, cost = map(float, feedback)
                has_cost = True
            else:
                reward = float(feedback)
                cost = np.nan
                has_cost = False
            if not np.isfinite(reward) or (has_cost and not np.isfinite(cost)):
                raise ValueError("The reveal callback must return finite feedback.")
            if returns_cost is None:
                returns_cost = has_cost
            elif returns_cost != has_cost:
                raise ValueError("reveal must consistently return rewards with or without costs.")

            selected[query_index, model_index] = True
            query_log.append(int(query_index))
            model_log.append(model_index)
            reward_log.append(reward)
            if has_cost:
                cost_log.append(cost)
            group_sum[group_index, model_index] += reward
            group_count[group_index, model_index] += 1
            global_sum[model_index] += reward
            global_count[model_index] += 1
        persistent = (group_sum, group_count, global_sum, global_count)

    return SparseSupervision(
        n_queries=n_queries,
        n_models=n_models,
        query_indices=np.asarray(query_log),
        model_indices=np.asarray(model_log),
        rewards=np.asarray(reward_log),
        costs=np.asarray(cost_log) if returns_cost else None,
        group_ids=groups,
        k=k,
        query_ids=query_ids,
        model_ids=model_ids,
        metadata={
            "collector": "grouped_empirical_bayes_ucb",
            "beta": float(beta),
            "prior_strength": float(prior_strength),
            "prior_alpha": float(prior_alpha),
            "prior_beta": float(prior_beta),
            "seed": int(seed),
            "seed_stride": int(seed_stride),
            "reset_each_pass": bool(reset_each_pass),
        },
    )


def simulate_fixed_k_supervision(
    rewards: np.ndarray,
    group_ids: Sequence[int],
    *,
    k: int,
    costs: Optional[np.ndarray] = None,
    query_ids: Optional[Sequence[str]] = None,
    model_ids: Optional[Sequence[str]] = None,
    **collector_kwargs,
) -> SparseSupervision:
    """Offline benchmark adapter; dense arrays are scoped to this function.

    Use ``collect_fixed_k_supervision`` directly in a real deployment.  The
    returned object contains selected entries only and is the sole input to
    router training.
    """
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rewards must be a two-dimensional array.")
    cost_values = None if costs is None else np.asarray(costs, dtype=np.float64)
    if cost_values is not None and cost_values.shape != values.shape:
        raise ValueError("costs must have the same shape as rewards.")
    available = np.isfinite(values)
    if cost_values is not None:
        available &= np.isfinite(cost_values)

    def reveal(query_index: int, model_index: int):
        reward = float(values[query_index, model_index])
        if cost_values is None:
            return reward
        return reward, float(cost_values[query_index, model_index])

    supervision = collect_fixed_k_supervision(
        n_queries=values.shape[0],
        n_models=values.shape[1],
        group_ids=group_ids,
        reveal=reveal,
        k=k,
        available=available,
        query_ids=query_ids,
        model_ids=model_ids,
        **collector_kwargs,
    )
    supervision.metadata["available_pairs"] = int(available.sum())
    return supervision
