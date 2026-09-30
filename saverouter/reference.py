"""Published main-row references and reproducibility checks."""

from __future__ import annotations


REFERENCE = {
    "llmrouterbench": {"maximum_quality": 0.6338254213, "target_cost_ratio": 0.2293770837, "SA-BEP": 5246, "SA-CR@1000000": 0.2334190356},
    "mixinstruct": {"maximum_quality": 0.7497895956, "target_cost_ratio": 0.9359261195, "SA-BEP": 1231760, "SA-CR@1000000": 1.0148497028},
    "mmrbench": {"maximum_quality": 0.7540983558, "target_cost_ratio": 0.7764742392, "SA-BEP": 18285, "SA-CR@1000000": 0.7805612356},
    "routerbench": {"maximum_quality": 0.8084081411, "target_cost_ratio": 0.6810688584, "SA-BEP": 42705, "SA-CR@1000000": 0.6946885651},
}


def verify_result(key, result, *, quality_atol=5e-4, ratio_atol=1e-3, bep_rtol=5e-3):
    expected = REFERENCE[key]
    checks = {
        "maximum_quality": abs(result["maximum_quality"] - expected["maximum_quality"]) <= quality_atol,
        "target_cost_ratio": (
            result["target_cost_ratio"] is not None
            and abs(result["target_cost_ratio"] - expected["target_cost_ratio"]) <= ratio_atol
        ),
        "SA-BEP": (
            result["SA-BEP"] != float("inf")
            and abs(result["SA-BEP"] - expected["SA-BEP"]) / max(expected["SA-BEP"], 1) <= bep_rtol
        ),
        "SA-CR@1000000": (
            result["SA-CR@1000000"] is not None
            and abs(result["SA-CR@1000000"] - expected["SA-CR@1000000"]) <= ratio_atol
        ),
    }
    observed = {name: result.get(name) for name in expected}
    deltas = {
        name: (
            None
            if observed[name] is None
            else float(observed[name]) - float(expected[name])
        )
        for name in expected
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected": expected,
        "observed": observed,
        "deltas": deltas,
        "tolerances": {
            "quality_atol": quality_atol,
            "ratio_atol": ratio_atol,
            "bep_rtol": bep_rtol,
        },
    }
