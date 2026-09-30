"""Frozen paper profiles for the four reported ORBIT benchmarks."""

from __future__ import annotations

from copy import deepcopy


PAPER_PROFILES = {
    "llmrouterbench": {
        "dataset_name": "LLMRouterBench",
        "modality": "text",
        "k": 4,
        "group_strategy": "auto",
        "n_groups": None,
        "prior_strength": 40.0,
        "prior_ridge_alpha": 10.0,
        "residual_ridge_alpha": 200.0,
        "residual_gamma": 2.0,
        "include_dense_context": False,
    },
    "mixinstruct": {
        "dataset_name": "Mixinstruct",
        "modality": "text",
        "k": 4,
        "group_strategy": "latent",
        "n_groups": 1,
        "prior_strength": 40.0,
        "prior_ridge_alpha": 10.0,
        "residual_ridge_alpha": 100.0,
        "residual_gamma": 2.0,
        "include_dense_context": False,
    },
    "mmrbench": {
        "dataset_name": "MMRBench",
        "modality": "text+image",
        "k": 4,
        "group_strategy": "auto",
        "n_groups": None,
        "prior_strength": 120.0,
        "prior_ridge_alpha": 10.0,
        "residual_ridge_alpha": 100.0,
        "residual_gamma": 0.25,
        "include_dense_context": True,
    },
    "routerbench": {
        "dataset_name": "Routerbench",
        "modality": "text",
        "k": 4,
        "group_strategy": "auto",
        "n_groups": None,
        "prior_strength": 40.0,
        "prior_ridge_alpha": 10.0,
        "residual_ridge_alpha": 200.0,
        "residual_gamma": 2.0,
        "include_dense_context": False,
    },
}

ALIASES = {
    "llmrouterbench": "llmrouterbench",
    "llm-router-bench": "llmrouterbench",
    "mixinstruct": "mixinstruct",
    "mix-instruct": "mixinstruct",
    "mmrbench": "mmrbench",
    "mmr-bench": "mmrbench",
    "routerbench": "routerbench",
    "router-bench": "routerbench",
}


def get_profile(name: str):
    key = ALIASES.get(str(name).lower())
    if key is None:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(PAPER_PROFILES)}")
    result = deepcopy(PAPER_PROFILES[key])
    result["key"] = key
    result.update({
        "seed": 42,
        "train_ratio": 0.2,
        "acquisition_beta": 0.35,
        "acquisition_prior_strength": 10.0,
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
        "seed_stride": 1009,
        "min_model_observations": 8,
        "quality_target_multiplier": 1.0,
        "horizon": 1_000_000,
    })
    return result
