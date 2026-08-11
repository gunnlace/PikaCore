"""Layered context compression and observability tests."""

import copy
import json

from pikacore.context import CompressionResult, ContextManager, estimate_tokens
from pikacore.llm import LLMResponse
from pikacore.state import FileMemory, WorkingMemory
from tests.protocol_assertions import assert_tool_pairing


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(copy.deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _tool_turn(call_id, name, arguments, content):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def test_noop_returns_explicit_compression_result():
    messages = [{"role": "user", "content": "short"}]

    result = ContextManager(max_tokens=1000).maybe_compress(messages)

    assert result == CompressionResult(
        changed=False,
        strategy=None,
        before_tokens=1,
        after_tokens=1,
        removed_messages=0,
        summarized_messages=0,
    )
    assert bool(result) is False


def test_tool_snip_preserves_tool_path_exit_error_tail_and_pairing():
    output = "first\n" + "noise\n" * 500 + "ERROR final failure\n[exit code: 7]"
    messages = [{"role": "user", "content": "run it"}]
    messages += _tool_turn(
        "bash-1",
        "bash",
        {"command": "pytest tests/", "path": "tests"},
        output,
    )

    result = ContextManager(max_tokens=100).maybe_compress(messages)

    assert result.changed
    assert result.strategy == "tool-output-snip"
    assert result.after_tokens < result.before_tokens
    assert result.removed_messages == 0
    compressed = messages[-1]["content"]
    assert "tool: bash" in compressed
    assert "path: tests" in compressed
    assert "exit_code: 7" in compressed
    assert "ERROR final failure" in compressed
    assert "truncated: true" in compressed
    assert_tool_pairing(messages)


def test_duplicate_old_reads_rely_on_working_memory_and_keep_latest_result():
    messages = []
    for index in range(4):
        messages.append({"role": "user", "content": f"read {index}"})
        messages += _tool_turn(
            f"read-{index}",
            "read_file",
            {"file_path": "source.py"},
            f"version-{index} " * 45,
        )
    memory = WorkingMemory(files=[
        FileMemory("source.py", "read", "latest source summary", "hash", True)
    ])

    result = ContextManager(max_tokens=1000).maybe_compress(
        messages,
        working_memory=memory,
    )

    assert result.strategy == "duplicate-read-merge"
    assert "Duplicate read omitted" in messages[2]["content"]
    assert "latest summary and freshness are in Working Memory" in messages[2]["content"]
    assert "version-3" in messages[-1]["content"]
    assert_tool_pairing(messages)


def test_old_grep_and_bash_use_local_extraction_without_llm():
    grep_output = "\n".join(f"src/file.py:{index}:needle" for index in range(60))
    messages = [{"role": "user", "content": "search"}]
    messages += _tool_turn(
        "grep-1",
        "grep",
        {"pattern": "needle", "path": "src"},
        grep_output,
    )
    for index in range(4):
        messages.extend([
            {"role": "user", "content": f"follow-up {index} " + "x" * 50},
            {"role": "assistant", "content": "noted " + "y" * 50},
        ])

    result = ContextManager(max_tokens=700).maybe_compress(messages)

    assert result.strategy == "local-tool-extract"
    extracted = messages[2]["content"]
    assert "tool: grep" in extracted
    assert "path: src" in extracted
    assert "truncated: true" in extracted
    assert len(extracted) < len(grep_output)
    assert_tool_pairing(messages)


def test_llm_summary_preserves_recent_structured_tool_turn_and_safe_split():
    messages = []
    for index in range(6):
        messages.extend([
            {"role": "user", "content": f"request {index} " + "u" * 500},
            {"role": "assistant", "content": f"answer {index} " + "a" * 500},
        ])
    messages += _tool_turn(
        "recent-read",
        "read_file",
        {"file_path": "recent.py"},
        "recent result",
    )
    recent_turn = copy.deepcopy(messages[-2:])
    llm = FakeLLM([LLMResponse(content="decisions and errors summarized")])

    result = ContextManager(max_tokens=2500).maybe_compress(messages, llm)

    assert result.strategy == "llm-summary"
    assert result.removed_messages > 0
    assert result.summarized_messages > 0
    assert len(llm.calls) == 1
    assert messages[-2:] == recent_turn
    assert_tool_pairing(messages)


def test_failed_summary_call_records_local_fallback_strategy():
    messages = []
    for index in range(7):
        messages.extend([
            {"role": "user", "content": f"request {index} source.py " + "u" * 500},
            {"role": "assistant", "content": f"answer {index} " + "a" * 500},
        ])
    llm = FakeLLM([RuntimeError("summary unavailable")])

    result = ContextManager(max_tokens=2500).maybe_compress(messages, llm)

    assert "local-summary" in result.strategy
    assert "llm-summary" not in result.strategy
    assert len(llm.calls) == 1


def test_hard_collapse_preserves_recovery_current_request_and_recent_legal_turn():
    recovery = {
        "role": "user",
        "content": "[PikaCore recovery: files-stale]\nReread source.py",
    }
    current = {"role": "user", "content": "Fix source.py now"}
    messages = [recovery, current]
    for index in range(6):
        messages.extend([
            {"role": "assistant", "content": f"old {index} " + "a" * 900},
            {"role": "user", "content": f"turn {index} " + "u" * 900},
        ])
    messages += _tool_turn(
        "recent",
        "read_file",
        {"file_path": "source.py"},
        "latest structured result",
    )
    recent_turn = copy.deepcopy(messages[-2:])

    memory = WorkingMemory(current_request=current["content"])
    result = ContextManager(max_tokens=2000).maybe_compress(
        messages,
        working_memory=memory,
    )

    assert "hard-collapse" in result.strategy
    assert recovery in messages
    assert current in messages
    assert messages[-2:] == recent_turn
    assert_tool_pairing(messages)


def test_working_memory_is_counted_but_not_rewritten_by_compression():
    memory = WorkingMemory(
        task_summary="task " * 100,
        files=[FileMemory("a.py", "modified", "summary", "hash", False)],
    )
    before = copy.deepcopy(memory)
    messages = [{"role": "user", "content": "short"}]

    result = ContextManager(max_tokens=10_000).maybe_compress(
        messages,
        working_memory=memory,
    )

    assert result.before_tokens > estimate_tokens(messages)
    assert memory == before


def test_summary_input_reserves_old_turn_budget_when_working_memory_is_full():
    memory = WorkingMemory(
        task_summary="large task",
        files=[
            FileMemory(
                f"file-{index}.py",
                "read",
                f"summary-{index} " + "m" * 580,
                f"hash-{index}",
                True,
            )
            for index in range(30)
        ],
    )
    messages = []
    for index in range(14):
        content = f"turn {index} " + "x" * 500
        if index == 5:
            content += " DECISION-XYZ"
        if index == 4:
            content += " ERROR-XYZ"
        messages.append({"role": "user", "content": content})
    llm = FakeLLM([LLMResponse(content="summary")])

    result = ContextManager(max_tokens=4000).maybe_compress(
        messages,
        llm,
        memory,
    )
    summary_input = llm.calls[0][1]["content"]

    assert result.changed
    assert "DECISION-XYZ" in summary_input
    assert "ERROR-XYZ" in summary_input
    assert "[Prior structured turns]" in summary_input
    assert "[Working Memory reference]" in summary_input
    assert len(summary_input) <= 15_000
