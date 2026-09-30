"""Router training that consumes sparse feedback and never a dense label matrix."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.linear_model import Ridge

from .data import SparseSupervision
from .grouping import GroupAssigner


class FixedKSparseRouter:
    """Group prior plus per-model contextual residual regressors.

    Parameters mirror the fixed-K ORBIT implementation.  ``fit`` intentionally
    has no dense reward-matrix parameter: its labels come exclusively from a
    :class:`SparseSupervision` artifact.
    """

    def __init__(
        self,
        *,
        group_assigner: Optional[GroupAssigner] = None,
        prior_strength: float = 40.0,
        prior_ridge_alpha: float = 10.0,
        residual_ridge_alpha: float = 100.0,
        residual_gamma: float = 2.0,
        min_model_observations: int = 8,
        score_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ) -> None:
        if min(prior_strength, prior_ridge_alpha, residual_ridge_alpha) <= 0:
            raise ValueError("Ridge and shrinkage strengths must be positive.")
        if min_model_observations <= 0:
            raise ValueError("min_model_observations must be positive.")
        if score_range is not None and score_range[0] >= score_range[1]:
            raise ValueError("score_range must be an increasing pair or None.")
        self.group_assigner = group_assigner
        self.prior_strength = float(prior_strength)
        self.prior_ridge_alpha = float(prior_ridge_alpha)
        self.residual_ridge_alpha = float(residual_ridge_alpha)
        self.residual_gamma = float(residual_gamma)
        self.min_model_observations = int(min_model_observations)
        self.score_range = score_range
        self.residual_models_ = {}

    @staticmethod
    def _check_features(X, n_rows: Optional[int] = None):
        if not hasattr(X, "shape") or len(X.shape) != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if n_rows is not None and int(X.shape[0]) != int(n_rows):
            raise ValueError("X and supervision contain different query counts.")
        if not issparse(X):
            X = np.asarray(X, dtype=np.float64)
            if not np.all(np.isfinite(X)):
                raise ValueError("X must contain finite features.")
        return X

    def _clip(self, values: np.ndarray) -> np.ndarray:
        if self.score_range is None:
            return values
        return np.clip(values, self.score_range[0], self.score_range[1])

    def fit(
        self,
        X,
        supervision: SparseSupervision,
    ) -> "FixedKSparseRouter":
        X = self._check_features(X, supervision.n_queries)
        supervision.validate()
        # ORBIT materializes the sparse mask with ``np.where``, which is
        # row-major. Preserve that order for numerically identical Ridge fits.
        order = np.lexsort(
            (supervision.model_indices, supervision.query_indices)
        )
        q = supervision.query_indices[order]
        m = supervision.model_indices[order]
        y = supervision.rewards[order]
        groups = supervision.group_ids
        n_obs = supervision.n_observations
        n_groups = int(groups.max()) + 1
        n_models = supervision.n_models
        observed_groups = groups[q]

        rows = np.arange(n_obs)
        design = np.zeros((n_obs, n_groups + n_models), dtype=np.float64)
        design[rows, observed_groups] = 1.0
        design[rows, n_groups + m] = 1.0
        # Match the paper ORBIT implementation exactly.
        prior_model = Ridge(
            alpha=self.prior_ridge_alpha,
            fit_intercept=True,
        ).fit(design, y)
        grid_groups = np.repeat(np.arange(n_groups), n_models)
        grid_models = np.tile(np.arange(n_models), n_groups)
        grid = np.zeros(
            (n_groups * n_models, n_groups + n_models), dtype=np.float64
        )
        grid_rows = np.arange(n_groups * n_models)
        grid[grid_rows, grid_groups] = 1.0
        grid[grid_rows, n_groups + grid_models] = 1.0
        additive = prior_model.predict(grid).reshape(n_groups, n_models)
        self.additive_prior_ = self._clip(additive)

        self.group_sums_ = np.zeros((n_groups, n_models), dtype=np.float64)
        self.group_counts_ = np.zeros((n_groups, n_models), dtype=np.int64)
        np.add.at(self.group_sums_, (observed_groups, m), y)
        np.add.at(self.group_counts_, (observed_groups, m), 1)
        self.group_scores_ = (
            self.group_sums_ + self.prior_strength * self.additive_prior_
        ) / (self.group_counts_ + self.prior_strength)

        self.residual_models_ = {}
        for model_index in range(n_models):
            positions = np.flatnonzero(m == model_index)
            positions = positions[np.argsort(q[positions], kind="stable")]
            if positions.size < self.min_model_observations:
                continue
            query_indices = q[positions]
            group_indices = groups[query_indices]
            observed_rewards = y[positions]
            leave_one_out = (
                self.group_sums_[group_indices, model_index]
                - observed_rewards
                + self.prior_strength
                * self.additive_prior_[group_indices, model_index]
            ) / (
                self.group_counts_[group_indices, model_index]
                - 1
                + self.prior_strength
            )
            residual = observed_rewards - leave_one_out
            self.residual_models_[model_index] = Ridge(
                alpha=self.residual_ridge_alpha,
                solver="lsqr",
            ).fit(X[query_indices], residual)

        self.n_models_ = n_models
        self.n_groups_ = n_groups
        self.n_features_in_ = int(X.shape[1])
        self.model_ids_ = supervision.model_ids
        self._fit_costs(supervision)
        self.training_summary_ = {
            "queries": supervision.n_queries,
            "models": n_models,
            "groups": n_groups,
            "k": supervision.k,
            "observations": n_obs,
            "density": supervision.density,
            "residual_models": len(self.residual_models_),
        }
        return self

    def _fit_costs(
        self,
        supervision: SparseSupervision,
    ) -> None:
        if supervision.costs is None:
            self.global_costs_ = None
            self.group_costs_ = None
            return

        m = supervision.model_indices
        observed_groups = supervision.group_ids[supervision.query_indices]
        n_groups, n_models = self.group_scores_.shape
        cost_sum = np.zeros(n_models, dtype=np.float64)
        cost_count = np.zeros(n_models, dtype=np.int64)
        np.add.at(cost_sum, m, supervision.costs)
        np.add.at(cost_count, m, 1)
        observed_mean = np.divide(
            cost_sum,
            cost_count,
            out=np.full(n_models, np.nan),
            where=cost_count > 0,
        )
        known = observed_mean[np.isfinite(observed_mean)]
        if known.size == 0:
            raise ValueError("At least one observed cost is required.")
        fallback = np.full(n_models, float(known.max()) * 1.05)
        self.global_costs_ = np.where(np.isfinite(observed_mean), observed_mean, fallback)

        group_sum = np.zeros((n_groups, n_models), dtype=np.float64)
        group_count = np.zeros((n_groups, n_models), dtype=np.int64)
        np.add.at(group_sum, (observed_groups, m), supervision.costs)
        np.add.at(group_count, (observed_groups, m), 1)
        self.group_costs_ = np.divide(
            group_sum,
            group_count,
            out=np.broadcast_to(self.global_costs_, (n_groups, n_models)).copy(),
            where=group_count > 0,
        )

    def _resolve_groups(self, X, group_ids) -> np.ndarray:
        if group_ids is None:
            if self.group_assigner is None:
                raise ValueError("Pass group_ids or construct the router with GroupAssigner.")
            groups = self.group_assigner.predict(X)
        else:
            groups = np.asarray(group_ids, dtype=np.int64)
        if groups.shape != (int(X.shape[0]),):
            raise ValueError("group_ids must contain one value per query.")
        if np.any((groups < 0) | (groups >= self.n_groups_)):
            raise ValueError("Predicted group is outside the training group range.")
        return groups

    def predict_scores(self, X, *, group_ids=None) -> np.ndarray:
        X = self._check_features(X)
        if int(X.shape[1]) != self.n_features_in_:
            raise ValueError("X has a different feature dimension than training data.")
        groups = self._resolve_groups(X, group_ids)
        scores = self.group_scores_[groups].copy()
        for model_index, model in self.residual_models_.items():
            scores[:, model_index] += self.residual_gamma * model.predict(X)
        # The structured prior is clipped during fitting; the paper's final
        # equation does not clip after adding the query-level residual.
        return scores

    def predict_costs(self, X, *, group_ids=None) -> np.ndarray:
        if self.group_costs_ is None:
            raise RuntimeError("No costs were supplied during training.")
        X = self._check_features(X)
        groups = self._resolve_groups(X, group_ids)
        return self.group_costs_[groups].copy()

    def route(
        self,
        X,
        *,
        max_cost: float,
        group_ids=None,
        available: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Choose the highest predicted reward under a per-query cost cap."""
        scores = self.predict_scores(X, group_ids=group_ids)
        costs = self.predict_costs(X, group_ids=group_ids)
        usable = np.ones_like(scores, dtype=bool)
        if available is not None:
            available_mask = np.asarray(available, dtype=bool)
            if available_mask.shape != scores.shape:
                raise ValueError("available must match the prediction matrix shape.")
            usable &= available_mask
        max_cost = float(max_cost)
        if not np.isfinite(max_cost) or max_cost < 0:
            raise ValueError("max_cost must be a finite non-negative value.")
        if np.any(~usable.any(axis=1)):
            raise ValueError("Every query must have at least one available model.")
        feasible = usable & (costs <= max_cost)
        empty = ~feasible.any(axis=1)
        cheapest = np.argmin(np.where(usable, costs, np.inf), axis=1)
        feasible[np.arange(len(feasible))[empty], cheapest[empty]] = True
        return np.argmax(np.where(feasible, scores, -np.inf), axis=1)
