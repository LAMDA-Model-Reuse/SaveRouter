"""Benchmark-independent construction of query groups."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression


class GroupAssigner:
    """Create acquisition groups and predict them for unseen queries.

    Label groups are useful for in-domain benchmarks when a task/category label
    is available at serving time or predictable from the query.  Latent groups
    are the general fallback and need only query features.
    """

    def __init__(
        self,
        *,
        strategy: str = "auto",
        n_groups: Optional[int] = None,
        max_label_groups: int = 128,
        min_label_fit: float = 0.8,
        classifier_c: float = 10.0,
        seed: int = 42,
    ) -> None:
        if strategy not in {"auto", "labels", "latent"}:
            raise ValueError("strategy must be 'auto', 'labels', or 'latent'.")
        self.strategy = strategy
        self.n_groups = n_groups
        self.max_label_groups = int(max_label_groups)
        self.min_label_fit = float(min_label_fit)
        self.classifier_c = float(classifier_c)
        self.seed = int(seed)
        self.model_ = None
        self.strategy_ = None
        self.group_names_ = None
        self.constant_group_ = None
        self.training_accuracy_ = None

    def fit_predict(self, X, labels: Optional[Sequence[object]] = None) -> np.ndarray:
        n_samples = int(X.shape[0])
        if n_samples == 0:
            raise ValueError("X must contain at least one query.")
        use_labels = self.strategy == "labels" or (
            self.strategy == "auto" and labels is not None
        )
        if use_labels:
            if labels is None or len(labels) != n_samples:
                raise ValueError("Label grouping requires one label per query.")
            names, ids = np.unique(np.asarray(labels).astype(str), return_inverse=True)
            accuracy = 1.0
            classifier = None
            if len(names) > 1:
                classifier = LogisticRegression(
                    C=self.classifier_c,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=self.seed,
                ).fit(X, ids)
                accuracy = float(classifier.score(X, ids))
            acceptable = len(names) <= self.max_label_groups and accuracy >= self.min_label_fit
            if self.strategy == "labels" or acceptable:
                self.model_ = classifier
                self.constant_group_ = 0 if len(names) == 1 else None
                self.strategy_ = "labels"
                self.group_names_ = names.tolist()
                self.training_accuracy_ = accuracy
                return ids.astype(np.int64)

        requested = self.n_groups
        if requested is None:
            requested = int(np.clip(round(np.sqrt(n_samples) / 2), 4, 32))
        n_groups = max(1, min(int(requested), n_samples))
        if n_groups == 1:
            self.model_ = None
            self.constant_group_ = 0
            ids = np.zeros(n_samples, dtype=np.int64)
        else:
            self.model_ = MiniBatchKMeans(
                n_clusters=n_groups,
                random_state=self.seed,
                n_init=3,
                batch_size=min(2048, n_samples),
            ).fit(X)
            self.constant_group_ = None
            ids = self.model_.labels_.astype(np.int64)
        self.strategy_ = "latent"
        self.group_names_ = [f"latent_{idx}" for idx in range(n_groups)]
        self.training_accuracy_ = 1.0
        return ids

    def predict(self, X) -> np.ndarray:
        if self.strategy_ is None:
            raise RuntimeError("GroupAssigner must be fitted before predict().")
        if self.constant_group_ is not None:
            return np.full(int(X.shape[0]), self.constant_group_, dtype=np.int64)
        return np.asarray(self.model_.predict(X), dtype=np.int64)
