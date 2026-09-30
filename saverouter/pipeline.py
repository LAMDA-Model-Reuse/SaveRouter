"""End-to-end SAVERouter paper reproduction on ORBIT benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .benchmarks.orbit_data import get_loader
from .embeddings import CLIPEncoder, MiniLMEncoder, cached_embeddings
from .evaluation import evaluate_predictions
from .features import TextFeatureEncoder
from .grouping import GroupAssigner
from .profiles import get_profile
from .router import FixedKSparseRouter
from .supervision import simulate_fixed_k_supervision


def _matrices(frame, n_models):
    quality = frame[
        [f"model_{index}_performance" for index in range(n_models)]
    ].to_numpy(dtype=np.float32)
    costs = frame[
        [f"model_{index}_cost" for index in range(n_models)]
    ].to_numpy(dtype=np.float32)
    return quality, costs


def _safe_json(value):
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def benchmark_args(profile, *, data_root):
    return {
        "modality": profile["modality"],
        "seed": profile["seed"],
        "dataset": {
            "name": profile["dataset_name"],
            "dataset_dir": str(Path(data_root) / profile["key"]),
            "split": {
                "mode": "in-domain",
                "ratios": {"train": profile["train_ratio"], "test": 1.0 - profile["train_ratio"]},
            },
        },
    }


def load_benchmark(profile, *, data_root):
    args = benchmark_args(profile, data_root=data_root)
    loader = get_loader(profile["dataset_name"])(
        target_path=args["dataset"]["dataset_dir"], args=args
    )
    train, test, model_list = loader.run()
    if model_list is None:
        raise RuntimeError(f"{profile['dataset_name']} did not provide a model list.")
    return train, test, list(model_list)


def download_benchmark(name, *, data_root="data"):
    profile = get_profile(name)
    args = benchmark_args(profile, data_root=data_root)
    loader = get_loader(profile["dataset_name"])(
        target_path=args["dataset"]["dataset_dir"], args=args
    )
    return loader.download()


def _query_embeddings(profile, train, test, *, device, cache_root):
    cache_path = Path(cache_root) / f"{profile['key']}_seed{profile['seed']}.npz"

    def build():
        if profile["key"] == "mixinstruct":
            return (
                np.zeros((len(train), 1), dtype=np.float32),
                np.zeros((len(test), 1), dtype=np.float32),
            )
        if profile["key"] == "mmrbench":
            encoder = CLIPEncoder(device=device)
            return (
                encoder.encode(train["prompt"].astype(str), train["image_path"]),
                encoder.encode(test["prompt"].astype(str), test["image_path"]),
            )
        encoder = MiniLMEncoder(device=device, cache_dir=Path(cache_root) / "models")
        return (
            encoder.encode(train["prompt"].astype(str)),
            encoder.encode(test["prompt"].astype(str)),
        )

    return cached_embeddings(cache_path, build)


def _groups(profile, train, group_train, group_test):
    assigner = GroupAssigner(
        strategy=profile["group_strategy"],
        n_groups=profile["n_groups"],
        seed=profile["seed"],
    )
    labels = train["eval_name"].astype(str).to_numpy() if "eval_name" in train else None
    train_groups = assigner.fit_predict(group_train, labels)
    test_groups = assigner.predict(group_test)
    return assigner, train_groups, test_groups


def _context_features(profile, train, test, group_train, group_test):
    encoder = TextFeatureEncoder()
    use_dense = bool(profile["include_dense_context"])
    train_features = encoder.fit_transform(
        texts=train["prompt"].astype(str).tolist() if "prompt" in train else None,
        dense=group_train if use_dense else None,
    )
    test_features = encoder.transform(
        texts=test["prompt"].astype(str).tolist() if "prompt" in test else None,
        dense=group_test if use_dense else None,
    )
    return train_features, test_features


def run_benchmark(
    benchmark,
    *,
    output_root="outputs",
    data_root="data",
    cache_root=".cache/saverouter",
    device="auto",
    overrides=None,
):
    profile = get_profile(benchmark)
    profile.update(dict(overrides or {}))
    train, test, model_list = load_benchmark(profile, data_root=data_root)
    group_train, group_test = _query_embeddings(
        profile, train, test, device=device, cache_root=cache_root
    )
    _, train_groups, test_groups = _groups(
        profile, train, group_train, group_test
    )
    context_train, context_test = _context_features(
        profile, train, test, group_train, group_test
    )
    train_quality, train_cost = _matrices(train, len(model_list))
    test_quality, test_cost = _matrices(test, len(model_list))

    supervision = simulate_fixed_k_supervision(
        train_quality,
        train_groups,
        costs=train_cost,
        k=profile["k"],
        seed=profile["seed"],
        beta=profile["acquisition_beta"],
        prior_strength=profile["acquisition_prior_strength"],
        prior_alpha=profile["prior_alpha"],
        prior_beta=profile["prior_beta"],
        seed_stride=profile["seed_stride"],
        reset_each_pass=True,
        query_ids=(
            train["sample_id"].astype(str).tolist()
            if "sample_id" in train
            else [str(index) for index in train.index]
        ),
        model_ids=model_list,
    )
    router = FixedKSparseRouter(
        prior_strength=profile["prior_strength"],
        prior_ridge_alpha=profile["prior_ridge_alpha"],
        residual_ridge_alpha=profile["residual_ridge_alpha"],
        residual_gamma=profile["residual_gamma"],
        min_model_observations=profile["min_model_observations"],
    ).fit(context_train, supervision)

    predicted_quality = router.predict_scores(context_test, group_ids=test_groups)
    predicted_cost = router.predict_costs(context_test, group_ids=test_groups)
    result = evaluate_predictions(
        predicted_quality,
        predicted_cost,
        test_quality,
        test_cost,
        supervision_cost=float(supervision.costs.sum()),
        global_predicted_cost=router.global_costs_,
        quality_target_multiplier=profile["quality_target_multiplier"],
        horizon=profile["horizon"],
    )
    result.update({
        "benchmark": profile["dataset_name"],
        "profile": profile,
        "seed": profile["seed"],
        "models": model_list,
        "train_queries": len(train),
        "test_queries": len(test),
        "data_revision": getattr(
            get_loader(profile["dataset_name"]), "revision", None
        ),
        "supervision": {
            **router.training_summary_,
            "acquisition_cost": float(supervision.costs.sum()),
        },
    })

    destination = Path(output_root) / profile["key"]
    destination.mkdir(parents=True, exist_ok=True)
    supervision.save(destination / "supervision.npz")
    (destination / "result.json").write_text(
        json.dumps(_safe_json(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    with (destination / "pareto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cost", "quality"])
        writer.writeheader()
        writer.writerows(result["frontier"])
    return result
