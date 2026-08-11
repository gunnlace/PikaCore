#!/usr/bin/env python3
"""Run the offline Phase 6 benchmark suite and write a JSON report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import (
    BASELINE,
    CONTEXT_POLICY_OFF,
    DEFAULT_ABLATIONS,
    WORKING_MEMORY_OFF,
    BenchmarkRunner,
)

_VARIANTS = {
    "baseline": (BASELINE,),
    "working-memory-off": (WORKING_MEMORY_OFF,),
    "context-policy-off": (CONTEXT_POLICY_OFF,),
    "all": DEFAULT_ABLATIONS,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic PikaCore fixtures with ScriptedFakeLLM.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        dest="fixtures",
        help="Run one fixture id; repeat for multiple fixtures (default: all 10).",
    )
    parser.add_argument(
        "--ablation",
        choices=tuple(_VARIANTS),
        default="all",
        help="Select baseline or one of the two supported ablations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".pikacore/benchmarks/phase6-report.json"),
        help="JSON report path.",
    )
    args = parser.parse_args(argv)

    runner = BenchmarkRunner()
    report = runner.run_suite(
        fixture_ids=tuple(args.fixtures) if args.fixtures else None,
        variants=_VARIANTS[args.ablation],
    )
    destination = runner.write_report(report, args.output)
    print(
        f"PikaCore Phase 6: {report.passed_count}/{report.outcome_count} passed; "
        f"report={destination}"
    )
    return _exit_code(report)


def _exit_code(report) -> int:
    if report.completed_count != report.outcome_count:
        return 1
    baseline_failed = any(
        outcome.variant == "baseline" and not outcome.passed
        for outcome in report.outcomes
    )
    return 1 if baseline_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
