"""Shared assertions for OpenAI-compatible message protocol invariants."""


def assert_tool_pairing(messages: list[dict]) -> None:
    """Require every assistant tool call to have exactly one adjacent reply."""
    expected_ids: list[str] = []
    expected_tool_indexes: list[int] = []

    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls", [])
        if not tool_calls:
            continue

        assert message.get("role") == "assistant"
        call_ids = [call["id"] for call in tool_calls]
        reply_indexes = list(range(index + 1, index + 1 + len(call_ids)))
        replies = messages[index + 1:index + 1 + len(call_ids)]

        assert len(replies) == len(call_ids)
        assert [reply.get("role") for reply in replies] == ["tool"] * len(call_ids)
        assert [reply.get("tool_call_id") for reply in replies] == call_ids

        expected_ids.extend(call_ids)
        expected_tool_indexes.extend(reply_indexes)

    actual_tool_indexes = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
    ]
    actual_ids = [messages[index].get("tool_call_id") for index in actual_tool_indexes]

    assert actual_tool_indexes == expected_tool_indexes
    assert actual_ids == expected_ids
    assert len(expected_ids) == len(set(expected_ids))
