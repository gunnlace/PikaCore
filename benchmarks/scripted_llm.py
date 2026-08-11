"""Strict fake LLM driven entirely by checked-in response scripts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from pikacore.llm import LLMResponse, ToolCall


@dataclass(frozen=True)
class ScriptedResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScriptedResponse:
        if not isinstance(data, dict):
            raise ValueError("A scripted response must be a JSON object")
        calls = []
        for raw_call in data.get("tool_calls", []):
            calls.append(ToolCall(
                id=str(raw_call["id"]),
                name=str(raw_call["name"]),
                arguments=copy.deepcopy(raw_call.get("arguments", {})),
            ))
        return cls(
            content=str(data.get("content", "")),
            tool_calls=tuple(calls),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
        )

    def to_llm_response(self) -> LLMResponse:
        return LLMResponse(
            content=self.content,
            tool_calls=list(copy.deepcopy(self.tool_calls)),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


class ScriptedFakeLLM:
    """A network-free LLM whose calls and token usage are fully scripted."""

    model = "scripted-fake"

    def __init__(self, responses: list[dict[str, Any] | ScriptedResponse]):
        self._responses = tuple(
            response
            if isinstance(response, ScriptedResponse)
            else ScriptedResponse.from_dict(response)
            for response in responses
        )
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._cursor

    def chat(self, messages, tools=None, on_token=None) -> LLMResponse:
        if self._cursor >= len(self._responses):
            raise AssertionError(
                f"ScriptedFakeLLM has no response for call {self._cursor + 1}"
            )
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
        })
        scripted = self._responses[self._cursor]
        self._cursor += 1
        response = scripted.to_llm_response()
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        if on_token and response.content:
            on_token(response.content)
        return response
