"""Machine-readable benchmark outcomes with deterministic comparison views."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkOutcome:
    benchmark_id: str
    variant: str
    working_memory_enabled: bool
    context_policy_enabled: bool
    passed: bool
    completed: bool
    model_attempts: int
    tool_steps: int
    prompt_tokens: int
    completion_tokens: int
    approval_count: int
    compression_count: int
    recovery_status: str | None
    stop_reason: str | None
    failure_categories: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)
    duration_ms: int = field(default=0, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_categories"] = list(self.failure_categories)
        return data

    def deterministic_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("duration_ms", None)
        return data


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    schema_version: int
    suite: str
    fixture_count: int
    variant_count: int
    outcome_count: int
    passed_count: int
    completed_count: int
    success_rate: float
    completion_rate: float
    outcomes: tuple[BenchmarkOutcome, ...]
    generated_at: str = field(compare=False)
    duration_ms: int = field(default=0, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "generated_at": self.generated_at,
            "duration_ms": self.duration_ms,
            "fixture_count": self.fixture_count,
            "variant_count": self.variant_count,
            "outcome_count": self.outcome_count,
            "passed_count": self.passed_count,
            "completed_count": self.completed_count,
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    def deterministic_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("generated_at", None)
        data.pop("duration_ms", None)
        data["outcomes"] = [
            outcome.deterministic_dict() for outcome in self.outcomes
        ]
        return data
