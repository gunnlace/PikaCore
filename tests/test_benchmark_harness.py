"""Phase 6 fixture harness determinism and ablation coverage."""

import json
from dataclasses import replace

import pytest

from benchmarks.harness import FixtureSpec
from benchmarks.harness import (
    BASELINE,
    CONTEXT_POLICY_OFF,
    DEFAULT_ABLATIONS,
    PHASE6_BENCHMARK_IDS,
    WORKING_MEMORY_OFF,
    BenchmarkRunner,
)
from benchmarks.run_benchmarks import _exit_code, main
from benchmarks.scripted_llm import ScriptedFakeLLM
from pikacore.agent import Agent


def test_exactly_ten_documented_fixture_tasks_and_repos_exist():
    runner = BenchmarkRunner()

    assert set(runner.fixture_ids()) == set(PHASE6_BENCHMARK_IDS)
    assert len(runner.fixture_ids()) == 10
    for benchmark_id in PHASE6_BENCHMARK_IDS:
        spec = runner.load_fixture(benchmark_id)
        assert spec.benchmark_id == benchmark_id
        assert spec.repo_files
        assert (runner.repos_dir / benchmark_id).is_dir()


def test_fixture_materialization_copies_only_explicit_manifest(tmp_path):
    benchmark_root = tmp_path / "benchmarks"
    source = benchmark_root / "repos" / "edit-basic"
    source.mkdir(parents=True)
    (source / "clean.txt").write_text("clean\n", encoding="utf-8")
    (source / ".env").write_text("SECRET=local\n", encoding="utf-8")
    (source / "extra_test.py").write_text("raise RuntimeError\n", encoding="utf-8")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"machine-local-bytecode")
    spec = FixtureSpec(
        benchmark_id="edit-basic",
        request="fixture",
        responses=(),
        repo_files=("clean.txt",),
    )
    destination = tmp_path / "materialized"

    BenchmarkRunner(benchmark_root).materialize_fixture(spec, destination)

    assert [
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ] == ["clean.txt"]
    assert (destination / "clean.txt").read_text(encoding="utf-8") == "clean\n"


def test_scripted_fake_llm_is_strict_repeatable_and_never_needs_an_api_key():
    script = [{
        "content": "deterministic",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "tool_calls": [{
            "id": "call-1",
            "name": "read_file",
            "arguments": {"file_path": "a.txt"},
        }],
    }]
    first = ScriptedFakeLLM(script)
    second = ScriptedFakeLLM(script)

    first_response = first.chat([{"role": "user", "content": "same"}])
    second_response = second.chat([{"role": "user", "content": "same"}])

    assert first_response == second_response
    assert first.total_prompt_tokens == second.total_prompt_tokens == 7
    assert first.total_completion_tokens == second.total_completion_tokens == 3
    assert first.remaining == second.remaining == 0
    with pytest.raises(AssertionError, match="no response"):
        first.chat([])


def test_same_fixture_and_fake_responses_have_equal_deterministic_outcomes():
    runner = BenchmarkRunner()

    first = runner.run_fixture("edit-basic", BASELINE)
    second = runner.run_fixture("edit-basic", BASELINE)

    assert first == second
    assert first.deterministic_dict() == second.deterministic_dict()
    assert first.passed is True
    assert "duration_ms" not in first.deterministic_dict()


def test_all_ten_baseline_benchmarks_pass_offline():
    report = BenchmarkRunner().run_suite(variants=(BASELINE,))

    assert report.fixture_count == 10
    assert report.variant_count == 1
    assert report.outcome_count == 10
    assert report.passed_count == 10
    assert report.completed_count == 10
    assert report.success_rate == 1.0
    assert report.completion_rate == 1.0
    assert {outcome.benchmark_id for outcome in report.outcomes} == set(
        PHASE6_BENCHMARK_IDS
    )


def test_only_working_memory_and_context_policy_ablations_are_defined():
    assert DEFAULT_ABLATIONS == (
        BASELINE,
        WORKING_MEMORY_OFF,
        CONTEXT_POLICY_OFF,
    )
    assert WORKING_MEMORY_OFF.working_memory_enabled is False
    assert WORKING_MEMORY_OFF.context_policy_enabled is True
    assert CONTEXT_POLICY_OFF.working_memory_enabled is True
    assert CONTEXT_POLICY_OFF.context_policy_enabled is False


def test_ablation_outcomes_change_only_the_targeted_fixture_contracts():
    runner = BenchmarkRunner()
    report = runner.run_suite(
        fixture_ids=("large-output", "working-memory-stale"),
        variants=DEFAULT_ABLATIONS,
    )
    outcomes = {
        (outcome.variant, outcome.benchmark_id): outcome
        for outcome in report.outcomes
    }

    assert outcomes[("baseline", "large-output")].passed
    assert outcomes[("baseline", "working-memory-stale")].passed
    assert outcomes[("working-memory-off", "large-output")].passed
    assert not outcomes[("working-memory-off", "working-memory-stale")].passed
    assert "check:working_memory_file" in outcomes[
        ("working-memory-off", "working-memory-stale")
    ].failure_categories
    assert outcomes[("context-policy-off", "working-memory-stale")].passed
    assert not outcomes[("context-policy-off", "large-output")].passed
    assert "check:compression_min" in outcomes[
        ("context-policy-off", "large-output")
    ].failure_categories


def test_json_report_keeps_observations_but_comparison_excludes_nondeterminism(tmp_path):
    runner = BenchmarkRunner()
    first = runner.run_suite(
        fixture_ids=("bad-args-retry",),
        variants=(BASELINE,),
    )
    second = runner.run_suite(
        fixture_ids=("bad-args-retry",),
        variants=(BASELINE,),
    )
    destination = runner.write_report(first, tmp_path / "report.json")
    persisted = json.loads(destination.read_text(encoding="utf-8"))

    assert first == second
    assert first.deterministic_dict() == second.deterministic_dict()
    assert "generated_at" in persisted
    assert "duration_ms" in persisted
    assert "generated_at" not in first.deterministic_dict()
    assert "duration_ms" not in first.deterministic_dict()
    serialized_comparison = json.dumps(first.deterministic_dict(), sort_keys=True)
    assert "session_" not in serialized_comparison
    assert "run_" not in serialized_comparison
    assert "checkpoint_" not in serialized_comparison


def test_runner_cli_writes_json_without_real_api(tmp_path):
    destination = tmp_path / "cli-report.json"

    exit_code = main([
        "--fixture",
        "bad-args-retry",
        "--ablation",
        "all",
        "--output",
        str(destination),
    ])

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["outcome_count"] == 3
    assert report["completed_count"] == 3


def test_cli_exit_code_fails_completed_baseline_regression_but_not_off_ablation():
    report = BenchmarkRunner().run_suite(
        fixture_ids=("bad-args-retry",),
        variants=(BASELINE,),
    )
    failed_baseline = replace(report.outcomes[0], passed=False)
    baseline_report = replace(
        report,
        passed_count=0,
        success_rate=0.0,
        outcomes=(failed_baseline,),
    )
    failed_off = replace(failed_baseline, variant="working-memory-off")
    off_report = replace(baseline_report, outcomes=(failed_off,))

    assert _exit_code(baseline_report) == 1
    assert _exit_code(off_report) == 0
    assert _exit_code(replace(report, completed_count=0)) == 1


@pytest.mark.parametrize("process_exception", [KeyboardInterrupt(), SystemExit(2)])
def test_runner_propagates_process_level_exceptions(monkeypatch, process_exception):
    def interrupt(_agent, _request):
        raise process_exception

    monkeypatch.setattr(Agent, "chat", interrupt)

    with pytest.raises(type(process_exception)):
        BenchmarkRunner().run_fixture("bad-args-retry", BASELINE)
