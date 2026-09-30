"""Command-line interface for installation checks and paper reproduction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .profiles import PAPER_PROFILES, get_profile
from .reference import verify_result
from .smoke import run_smoke_test


def _names(value):
    return list(PAPER_PROFILES) if value == "all" else [get_profile(value)["key"]]


def reproduce(args):
    from .pipeline import run_benchmark

    summaries = []
    failed = []
    for name in _names(args.benchmark):
        print(f"\n=== SAVERouter: {name} ===", flush=True)
        result = run_benchmark(
            name,
            output_root=args.output,
            data_root=args.data_root,
            cache_root=args.cache_root,
            device=args.device,
        )
        verification = verify_result(name, result) if args.verify else None
        if verification is not None and not verification["passed"]:
            failed.append(name)
            print(json.dumps({"benchmark": name, "verification": verification}, indent=2))
        summaries.append({
            "benchmark": name,
            "Max": result["maximum_quality"],
            "CR": result["target_cost_ratio"],
            "SA-BEP": result["SA-BEP"],
            "SA-CR@1M": result["SA-CR@1000000"],
            "density": result["supervision"]["density"],
            "verification": "PASS" if verification and verification["passed"] else ("FAIL" if verification else "SKIP"),
        })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "main_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    if failed:
        raise SystemExit(f"Reference tolerance failed for: {', '.join(failed)}")


def download(args):
    from .pipeline import download_benchmark

    for name in _names(args.benchmark):
        print(f"Downloading {name}...", flush=True)
        download_benchmark(name, data_root=args.data_root)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="saverouter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce_parser = subparsers.add_parser("reproduce", help="run paper profiles")
    reproduce_parser.add_argument("--benchmark", default="all")
    reproduce_parser.add_argument("--output", default="outputs")
    reproduce_parser.add_argument("--data-root", default="data")
    reproduce_parser.add_argument("--cache-root", default=".cache/saverouter")
    reproduce_parser.add_argument("--device", default="auto")
    reproduce_parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    reproduce_parser.set_defaults(func=reproduce)

    download_parser = subparsers.add_parser("download", help="download benchmark data")
    download_parser.add_argument("--benchmark", default="all")
    download_parser.add_argument("--data-root", default="data")
    download_parser.set_defaults(func=download)

    smoke_parser = subparsers.add_parser("smoke-test", help="run without external data")
    smoke_parser.set_defaults(func=lambda _: print(json.dumps(run_smoke_test(), indent=2)))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
