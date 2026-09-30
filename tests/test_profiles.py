from saverouter.profiles import PAPER_PROFILES, get_profile
from saverouter.reference import REFERENCE, verify_result


def test_all_paper_profiles_use_fixed_k4():
    assert set(PAPER_PROFILES) == {"llmrouterbench", "mixinstruct", "mmrbench", "routerbench"}
    assert all(get_profile(name)["k"] == 4 for name in PAPER_PROFILES)


def test_main_table_overrides_are_frozen():
    assert get_profile("MMR-Bench")["residual_gamma"] == 0.25
    assert get_profile("MMR-Bench")["prior_strength"] == 120.0
    assert get_profile("RouterBench")["residual_ridge_alpha"] == 200.0
    assert get_profile("MixInstruct")["n_groups"] == 1


def test_reference_check_rejects_old_broad_tolerance():
    result = dict(REFERENCE["llmrouterbench"])
    result["target_cost_ratio"] += 0.008
    assert not verify_result("llmrouterbench", result)["passed"]
