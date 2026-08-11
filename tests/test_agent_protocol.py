"""Protocol invariants for the Agent loop, exercised with a local FakeLLM."""

import copy
import threading
from collections.abc import Callable

import pytest

from pikacore.agent import Agent
from pikacore.checkpoint import INTERRUPTED_TOOL_RESULT
from pikacore.llm import LLMResponse, ToolCall
from pikacore.permissions import PermissionPolicy
from pikacore.tools.agent import AgentTool
from pikacore.tools.base import Tool
from tests.protocol_assertions import assert_tool_pairing


class FakeLLM:
    """Scripted LLM that records requests and never creates a network client."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        *,
        response_factory: Callable[[int], LLMResponse] | None = None,
        stream_chunks: dict[int, list[str]] | None = None,
    ):
        self.responses = responses or []
        self.response_factory = response_factory
        self.stream_chunks = stream_chunks or {}
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_token=None) -> LLMResponse:
        call_index = len(self.calls)
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })

        if self.response_factory is not None:
            response = self.response_factory(call_index)
        else:
            if call_index >= len(self.responses):
                raise AssertionError(f"FakeLLM has no scripted response for call {call_index}")
            response = self.responses[call_index]

        if on_token:
            for chunk in self.stream_chunks.get(call_index, []):
                on_token(chunk)
        return response


class EchoTool(Tool):
    name = "echo"
    description = "Return the provided value."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, value: str) -> str:
        return f"echo:{value}"


class NamedTool(EchoTool):
    def __init__(self, name: str):
        self.name = name


class ReverseCompletionTool(EchoTool):
    name = "ordered"

    def __init__(self):
        self.second_finished = threading.Event()
        self.completion_order: list[str] = []

    def execute(self, value: str) -> str:
        if value == "first":
            if not self.second_finished.wait(timeout=2):
                raise AssertionError("parallel tool call did not start")
        else:
            self.completion_order.append(value)
            self.second_finished.set()
            return f"result:{value}"

        self.completion_order.append(value)
        return f"result:{value}"


class BoomTool(Tool):
    name = "boom"
    description = "Raise a TypeError inside the tool."
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        raise TypeError("internal explosion")


class InterruptTool(Tool):
    name = "interrupt"
    description = "Simulate Ctrl+C during tool execution."
    parameters = {"type": "object", "properties": {}, "required": []}

    def execute(self) -> str:
        raise KeyboardInterrupt


def _tool_call(call_id: str, name: str = "echo", **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _tool_round(*tool_calls: ToolCall) -> LLMResponse:
    return LLMResponse(tool_calls=list(tool_calls))


def _final(content: str) -> LLMResponse:
    return LLMResponse(content=content)


def test_tool_calls_are_paired_exactly_once_across_rounds():
    llm = FakeLLM([
        _tool_round(_tool_call("call-a", value="a"), _tool_call("call-b", value="b")),
        _tool_round(_tool_call("call-c", value="c")),
        _final("done"),
    ])
    agent = Agent(llm=llm, tools=[EchoTool()], persist=False)

    assert agent.chat("run tools") == "done"
    assert_tool_pairing(agent.messages)

    tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call-a", "call-b", "call-c"]

    duplicate_history = copy.deepcopy(agent.messages)
    second_tool_index = next(
        index
        for index, message in enumerate(duplicate_history)
        if message.get("tool_call_id") == "call-b"
    )
    duplicate_history.insert(second_tool_index + 1, copy.deepcopy(duplicate_history[second_tool_index]))
    with pytest.raises(AssertionError):
        assert_tool_pairing(duplicate_history)


def test_parallel_results_keep_model_order_even_when_completion_order_differs():
    tool = ReverseCompletionTool()
    llm = FakeLLM([
        _tool_round(
            _tool_call("first-id", name="ordered", value="first"),
            _tool_call("second-id", name="ordered", value="second"),
        ),
        _final("ordered"),
    ])
    agent = Agent(llm=llm, tools=[tool], persist=False)

    assert agent.chat("run in parallel") == "ordered"
    assert tool.completion_order == ["second", "first"]
    tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
    assert [(message["tool_call_id"], message["content"]) for message in tool_messages] == [
        ("first-id", "result:first"),
        ("second-id", "result:second"),
    ]
    assert_tool_pairing(agent.messages)


def test_bad_arguments_and_internal_type_errors_become_distinct_tool_results():
    llm = FakeLLM([
        _tool_round(
            _tool_call("bad-args", name="echo"),
            _tool_call("tool-error", name="boom"),
        ),
        _final("recovered"),
    ])
    agent = Agent(llm=llm, tools=[EchoTool(), BoomTool()], persist=False)

    assert agent.chat("exercise errors") == "recovered"
    tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
    assert "bad arguments for echo" in tool_messages[0]["content"]
    assert "Error executing boom: internal explosion" in tool_messages[1]["content"]
    assert "bad arguments" not in tool_messages[1]["content"]
    assert_tool_pairing(agent.messages)


def test_interrupt_backfills_every_pending_tool_call():
    llm = FakeLLM([
        _tool_round(
            _tool_call("interrupted-id", name="interrupt"),
            _tool_call("other-id", value="other"),
        ),
    ])
    agent = Agent(llm=llm, tools=[InterruptTool(), EchoTool()], persist=False)

    with pytest.raises(KeyboardInterrupt):
        agent.chat("interrupt this round")

    tool_messages = [message for message in agent.messages if message.get("role") == "tool"]
    assert [(message["tool_call_id"], message["content"]) for message in tool_messages] == [
        ("interrupted-id", INTERRUPTED_TOOL_RESULT),
        ("other-id", INTERRUPTED_TOOL_RESULT),
    ]
    assert_tool_pairing(agent.messages)


def test_streaming_callback_and_complete_assistant_history_are_both_preserved():
    llm = FakeLLM(
        [_final("streamed reply")],
        stream_chunks={0: ["streamed ", "reply"]},
    )
    agent = Agent(llm=llm, tools=[], persist=False)
    streamed: list[str] = []

    result = agent.chat("stream", on_token=streamed.append)

    assert result == "streamed reply"
    assert streamed == ["streamed ", "reply"]
    assert agent.messages[-1] == {"role": "assistant", "content": "streamed reply"}


def test_tool_resolution_is_scoped_to_each_agent_instance():
    outside_llm = FakeLLM([
        _tool_round(
            _tool_call("outside-own", name="outside", value="own"),
            _tool_call("outside-cross", name="inside", value="cross"),
        ),
        _final("outside done"),
    ])
    outside_agent = Agent(llm=outside_llm, tools=[NamedTool("outside")], persist=False)

    inside_llm = FakeLLM([
        _tool_round(
            _tool_call("inside-own", name="inside", value="own"),
            _tool_call("inside-cross", name="outside", value="cross"),
        ),
        _final("inside done"),
    ])
    inside_agent = Agent(llm=inside_llm, tools=[NamedTool("inside")], persist=False)

    assert outside_agent.chat("check outside scope") == "outside done"
    assert inside_agent.chat("check inside scope") == "inside done"

    assert [schema["function"]["name"] for schema in outside_llm.calls[0]["tools"]] == ["outside"]
    assert [schema["function"]["name"] for schema in inside_llm.calls[0]["tools"]] == ["inside"]

    outside_results = {
        message["tool_call_id"]: message["content"]
        for message in outside_agent.messages
        if message.get("role") == "tool"
    }
    inside_results = {
        message["tool_call_id"]: message["content"]
        for message in inside_agent.messages
        if message.get("role") == "tool"
    }
    assert outside_results == {
        "outside-own": "echo:own",
        "outside-cross": "Error: unknown tool 'inside'",
    }
    assert inside_results == {
        "inside-own": "echo:own",
        "inside-cross": "Error: unknown tool 'outside'",
    }
    assert_tool_pairing(outside_agent.messages)
    assert_tool_pairing(inside_agent.messages)


def test_sub_agent_cannot_access_agent_tool_recursively():
    llm = FakeLLM([
        _tool_round(_tool_call("parent-agent", name="agent", task="delegate")),
        _tool_round(_tool_call("child-agent", name="agent", task="recurse")),
        _final("child finished"),
        _final("parent finished"),
    ])
    agent = Agent(
        llm=llm,
        tools=[AgentTool()],
        permission_policy=PermissionPolicy("auto"),
        persist=False,
    )

    assert agent.chat("use a sub-agent") == "parent finished"
    assert llm.calls[1]["tools"] == []

    child_tool_message = next(
        message
        for message in llm.calls[2]["messages"]
        if message.get("tool_call_id") == "child-agent"
    )
    assert child_tool_message["content"] == "Error: unknown tool 'agent'"

    parent_tool_message = next(
        message
        for message in agent.messages
        if message.get("tool_call_id") == "parent-agent"
    )
    assert parent_tool_message["content"] == "[Sub-agent completed]\nchild finished"
    assert_tool_pairing(agent.messages)


def test_max_rounds_stops_after_exactly_the_configured_tool_rounds():
    def keep_calling(call_index: int) -> LLMResponse:
        return _tool_round(_tool_call(f"round-{call_index}", value=str(call_index)))

    llm = FakeLLM(response_factory=keep_calling)
    agent = Agent(llm=llm, tools=[EchoTool()], max_rounds=3, persist=False)

    assert agent.chat("never finish") == "(reached maximum tool-call rounds)"
    assert len(llm.calls) == 3
    assert [
        message["tool_call_id"]
        for message in agent.messages
        if message.get("role") == "tool"
    ] == ["round-0", "round-1", "round-2"]
    assert_tool_pairing(agent.messages)
