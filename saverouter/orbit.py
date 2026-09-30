"""Thin adapter for every ORBIT benchmark's standardized dataframe schema."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .supervision import simulate_fixed_k_supervision


def supervision_from_orbit_frame(
    frame,
    *,
    group_ids: Sequence[int],
    k: int,
    model_list: Optional[Sequence[str]] = None,
    query_id_column: Optional[str] = None,
    **collector_kwargs,
):
    """Convert any ORBIT benchmark's training split to sparse supervision.

    ORBIT loaders normalize all five built-in benchmarks to
    ``model_{i}_performance`` and ``model_{i}_cost`` columns.  This function is
    therefore benchmark-name agnostic and works for custom loaders that obey
    the same schema.
    """
    if model_list is None:
        performance_columns = sorted(
            (column for column in frame.columns if column.startswith("model_") and column.endswith("_performance")),
            key=lambda name: int(name.split("_")[1]),
        )
        n_models = len(performance_columns)
        model_ids = [str(index) for index in range(n_models)]
    else:
        n_models = len(model_list)
        model_ids = [str(model) for model in model_list]
        performance_columns = [f"model_{index}_performance" for index in range(n_models)]
    cost_columns = [f"model_{index}_cost" for index in range(n_models)]
    missing = [column for column in performance_columns if column not in frame]
    if missing:
        raise ValueError(f"Missing ORBIT performance columns: {missing[:3]}")
    rewards = frame[performance_columns].to_numpy(dtype=np.float64)
    costs = (
        frame[cost_columns].to_numpy(dtype=np.float64)
        if all(column in frame for column in cost_columns)
        else None
    )
    query_ids = None
    if query_id_column is not None:
        if query_id_column not in frame:
            raise ValueError(f"Missing query ID column: {query_id_column}")
        query_ids = frame[query_id_column].astype(str).tolist()
    return simulate_fixed_k_supervision(
        rewards,
        group_ids,
        k=k,
        costs=costs,
        query_ids=query_ids,
        model_ids=model_ids,
        **collector_kwargs,
    )
