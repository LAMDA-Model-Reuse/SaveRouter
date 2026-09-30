"""Validated, portable representation of fixed-K sparse supervision."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SparseSupervision:
    """COO-form observations collected from a query-by-model reward matrix.

    ``query_indices[t]``, ``model_indices[t]`` and ``rewards[t]`` describe one
    revealed feedback item.  No placeholder exists for unrevealed rewards, so
    a downstream trainer cannot accidentally consume them.
    """

    n_queries: int
    n_models: int
    query_indices: np.ndarray
    model_indices: np.ndarray
    rewards: np.ndarray
    group_ids: np.ndarray
    k: int
    costs: Optional[np.ndarray] = None
    query_ids: Optional[Sequence[str]] = None
    model_ids: Optional[Sequence[str]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_queries = int(self.n_queries)
        self.n_models = int(self.n_models)
        self.k = int(self.k)
        self.query_indices = np.asarray(self.query_indices, dtype=np.int64)
        self.model_indices = np.asarray(self.model_indices, dtype=np.int64)
        self.rewards = np.asarray(self.rewards, dtype=np.float64)
        self.group_ids = np.asarray(self.group_ids, dtype=np.int64)
        if self.costs is not None:
            self.costs = np.asarray(self.costs, dtype=np.float64)
        if self.query_ids is not None:
            self.query_ids = tuple(str(value) for value in self.query_ids)
        if self.model_ids is not None:
            self.model_ids = tuple(str(value) for value in self.model_ids)
        self.metadata = dict(self.metadata)
        self.validate()

    @property
    def n_observations(self) -> int:
        return int(self.rewards.size)

    @property
    def density(self) -> float:
        available_pairs = int(
            self.metadata.get("available_pairs", self.n_queries * self.n_models)
        )
        return self.n_observations / float(available_pairs)

    @property
    def observation_counts(self) -> np.ndarray:
        return np.bincount(self.query_indices, minlength=self.n_queries)

    def validate(self) -> None:
        if self.n_queries <= 0 or self.n_models <= 0:
            raise ValueError("n_queries and n_models must be positive.")
        if self.k <= 0 or self.k > self.n_models:
            raise ValueError("k must be in [1, n_models].")
        arrays = (self.query_indices, self.model_indices, self.rewards)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Observation arrays must be one-dimensional.")
        if not (
            len(self.query_indices) == len(self.model_indices) == len(self.rewards)
        ):
            raise ValueError("Observation arrays must have equal length.")
        if self.group_ids.shape != (self.n_queries,):
            raise ValueError("group_ids must contain one value per query.")
        if self.group_ids.size == 0 or np.any(self.group_ids < 0):
            raise ValueError("group_ids must be non-negative integers.")
        if np.any((self.query_indices < 0) | (self.query_indices >= self.n_queries)):
            raise ValueError("query_indices contains an out-of-range value.")
        if np.any((self.model_indices < 0) | (self.model_indices >= self.n_models)):
            raise ValueError("model_indices contains an out-of-range value.")
        if not np.all(np.isfinite(self.rewards)):
            raise ValueError("Every revealed reward must be finite.")
        if self.costs is not None:
            if self.costs.shape != self.rewards.shape:
                raise ValueError("costs must align with rewards.")
            if not np.all(np.isfinite(self.costs)):
                raise ValueError("Every revealed cost must be finite.")
            if np.any(self.costs < 0):
                raise ValueError("Revealed costs must be non-negative.")
        pairs = self.query_indices * self.n_models + self.model_indices
        if len(np.unique(pairs)) != len(pairs):
            raise ValueError("A query/model pair may be observed at most once.")
        expected = np.full(self.n_queries, self.k, dtype=np.int64)
        if not np.array_equal(self.observation_counts, expected):
            raise ValueError("Fixed-K supervision must contain exactly k observations per query.")
        if self.query_ids is not None and len(self.query_ids) != self.n_queries:
            raise ValueError("query_ids must contain one ID per query.")
        if self.model_ids is not None and len(self.model_ids) != self.n_models:
            raise ValueError("model_ids must contain one ID per model.")

    def observation_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_queries, self.n_models), dtype=bool)
        mask[self.query_indices, self.model_indices] = True
        return mask

    def save(self, path: str | Path) -> None:
        """Save without pickle; the resulting NPZ is safe to load as plain data."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": np.asarray("fixed-k-supervision-v1"),
            "n_queries": np.asarray(self.n_queries, dtype=np.int64),
            "n_models": np.asarray(self.n_models, dtype=np.int64),
            "k": np.asarray(self.k, dtype=np.int64),
            "query_indices": self.query_indices,
            "model_indices": self.model_indices,
            "rewards": self.rewards,
            "group_ids": self.group_ids,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        if self.costs is not None:
            payload["costs"] = self.costs
        if self.query_ids is not None:
            payload["query_ids"] = np.asarray(self.query_ids, dtype=np.str_)
        if self.model_ids is not None:
            payload["model_ids"] = np.asarray(self.model_ids, dtype=np.str_)
        np.savez_compressed(destination, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "SparseSupervision":
        with np.load(Path(path), allow_pickle=False) as archive:
            version = str(archive["format_version"].item())
            if version != "fixed-k-supervision-v1":
                raise ValueError(f"Unsupported supervision format: {version}")
            names = set(archive.files)
            return cls(
                n_queries=int(archive["n_queries"].item()),
                n_models=int(archive["n_models"].item()),
                k=int(archive["k"].item()),
                query_indices=archive["query_indices"],
                model_indices=archive["model_indices"],
                rewards=archive["rewards"],
                group_ids=archive["group_ids"],
                costs=archive["costs"] if "costs" in names else None,
                query_ids=archive["query_ids"].tolist() if "query_ids" in names else None,
                model_ids=archive["model_ids"].tolist() if "model_ids" in names else None,
                metadata=json.loads(str(archive["metadata_json"].item())),
            )
