"""Protocol-safe, observable context compression."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .working_memory import render_working_memory

if TYPE_CHECKING:
    from .llm import LLM
    from .state import WorkingMemory

_SUMMARY_INPUT_LIMIT = 15_000
_SUMMARY_MEMORY_LIMIT = 3_500
_SUMMARY_TURNS_LIMIT = 11_000


@dataclass(frozen=True)
class CompressionResult:
    changed: bool
    strategy: str | None
    before_tokens: int
    after_tokens: int
    removed_messages: int
    summarized_messages: int

    def __bool__(self) -> bool:
        return self.changed


def _approx_tokens(text: str) -> int:
    """Rough token count, roughly 3 chars per token for mixed en/zh content."""
    return len(text) // 3


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        if message.get("content"):
            total += _approx_tokens(str(message["content"]))
        if message.get("tool_calls"):
            total += _approx_tokens(str(message["tool_calls"]))
    return total


class ContextManager:
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        self._snip_at = int(max_tokens * 0.50)
        self._summarize_at = int(max_tokens * 0.70)
        self._collapse_at = int(max_tokens * 0.90)

    def maybe_compress(
        self,
        messages: list[dict],
        llm: LLM | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> CompressionResult:
        """Apply layered compression and report the exact decision."""
        before_tokens = self._total_tokens(messages, working_memory)
        current = before_tokens
        before_count = len(messages)
        before_messages = Counter(self._message_signature(item) for item in messages)
        strategies: list[str] = []
        metadata = self._tool_metadata(messages)

        # Normally only old turns are compacted. Under hard pressure, verbose
        # content in the recent turn may be snipped while its structure remains.
        old_limit = self._safe_split(messages, keep_recent=8)
        if current > self._collapse_at:
            old_limit = len(messages)

        if current > self._snip_at:
            count = self._snip_tool_outputs(messages, old_limit, metadata)
            if count:
                strategies.append("tool-output-snip")
                current = self._total_tokens(messages, working_memory)

        if current > self._snip_at:
            count = self._merge_duplicate_reads(
                messages,
                old_limit,
                metadata,
                working_memory,
            )
            if count:
                strategies.append("duplicate-read-merge")
                current = self._total_tokens(messages, working_memory)

        if current > self._snip_at:
            count = self._extract_old_search_and_commands(
                messages,
                old_limit,
                metadata,
            )
            if count:
                strategies.append("local-tool-extract")
                current = self._total_tokens(messages, working_memory)

        if current > self._summarize_at and len(messages) > 10:
            count, summary_strategy = self._summarize_old(
                messages,
                llm,
                working_memory,
                keep_recent=8,
            )
            if count:
                strategies.append(summary_strategy)
                current = self._total_tokens(messages, working_memory)

        if current > self._collapse_at and len(messages) > 4:
            count = self._hard_collapse(messages, llm, working_memory)
            if count:
                strategies.append("hard-collapse")

        after_tokens = self._total_tokens(messages, working_memory)
        after_messages = Counter(self._message_signature(item) for item in messages)
        summarized = sum(
            max(0, count - after_messages.get(signature, 0))
            for signature, count in before_messages.items()
        )
        changed = bool(strategies)
        return CompressionResult(
            changed=changed,
            strategy="+".join(strategies) if strategies else None,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            removed_messages=max(0, before_count - len(messages)),
            summarized_messages=summarized,
        )

    @staticmethod
    def _snip_tool_outputs(
        messages: list[dict],
        limit: int | None = None,
        metadata: dict[str, dict] | None = None,
    ) -> int:
        """Trim verbose tool results while retaining execution metadata."""
        changed = 0
        metadata = metadata or {}
        upper = len(messages) if limit is None else min(limit, len(messages))
        for message in messages[:upper]:
            if message.get("role") != "tool":
                continue
            content = str(message.get("content", "") or "")
            if len(content) <= 1500 or content.startswith("[Tool output compressed]"):
                continue
            info = metadata.get(str(message.get("tool_call_id")), {})
            message["content"] = ContextManager._compressed_tool_output(
                content,
                info,
            )
            changed += 1
        return changed

    @staticmethod
    def _safe_split(messages: list[dict], keep_recent: int) -> int:
        """Return a boundary that never orphans a tool result."""
        split = max(0, len(messages) - keep_recent)
        while split > 0 and messages[split].get("role") == "tool":
            split -= 1
        return split

    def _merge_duplicate_reads(
        self,
        messages: list[dict],
        limit: int,
        metadata: dict[str, dict],
        working_memory: WorkingMemory | None,
    ) -> int:
        latest_by_path: dict[str, int] = {}
        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            info = metadata.get(str(message.get("tool_call_id")), {})
            if info.get("name") == "read_file" and info.get("path"):
                latest_by_path[str(info["path"])] = index

        memory_paths = {
            item.path for item in working_memory.files
        } if working_memory is not None else set()
        changed = 0
        for index, message in enumerate(messages[:limit]):
            if message.get("role") != "tool":
                continue
            info = metadata.get(str(message.get("tool_call_id")), {})
            path = info.get("path")
            if (
                info.get("name") != "read_file"
                or not path
                or latest_by_path.get(str(path)) == index
            ):
                continue
            if working_memory is not None and str(path) not in memory_paths:
                continue
            replacement = (
                "[Duplicate read omitted]\n"
                f"tool: read_file\npath: {path}\n"
                "latest summary and freshness are in Working Memory\n"
                "truncated: true"
            )
            if message.get("content") != replacement:
                message["content"] = replacement
                changed += 1
        return changed

    @staticmethod
    def _extract_old_search_and_commands(
        messages: list[dict],
        limit: int,
        metadata: dict[str, dict],
    ) -> int:
        changed = 0
        for message in messages[:limit]:
            if message.get("role") != "tool":
                continue
            info = metadata.get(str(message.get("tool_call_id")), {})
            name = info.get("name")
            if name not in {"grep", "glob", "bash"}:
                continue
            content = str(message.get("content", "") or "")
            if len(content) <= 600:
                continue
            lines = content.splitlines()
            key_lines = [
                line for line in lines
                if any(token in line.lower() for token in (
                    "error", "failed", "failure", "traceback", "denied", "exit code"
                ))
            ]
            if name in {"grep", "glob"}:
                selected = lines[:8] + (lines[-2:] if len(lines) > 10 else [])
            else:
                selected = key_lines[-5:] + lines[-3:]
            selected = list(dict.fromkeys(line for line in selected if line))
            header = ["[Local tool result extraction]", f"tool: {name}"]
            if info.get("path"):
                header.append(f"path: {info['path']}")
            if info.get("command"):
                header.append(f"command: {info['command']}")
            if info.get("exit_code") is not None:
                header.append(f"exit_code: {info['exit_code']}")
            header.extend(("truncated: true", *selected))
            replacement = "\n".join(header)
            if replacement != content and len(replacement) < len(content):
                message["content"] = replacement
                changed += 1
        return changed

    def _summarize_old(
        self,
        messages: list[dict],
        llm: LLM | None,
        working_memory: WorkingMemory | None = None,
        keep_recent: int = 8,
    ) -> tuple[int, str]:
        """Summarize complete old turns and keep recent structured messages."""
        if len(messages) <= keep_recent:
            return 0, "local-summary"
        split = self._safe_split(messages, keep_recent)
        if split <= 0:
            return 0, "local-summary"
        old = messages[:split]
        tail = messages[split:]
        summary, strategy = self._get_summary(old, llm, working_memory)
        replacement = self._summary_pair(
            "[Context compressed - conversation summary]",
            summary,
        )
        replacement.extend(self._special_messages(old, tail, working_memory))
        if estimate_tokens(replacement) >= estimate_tokens(old):
            return 0, strategy
        messages[:] = replacement + tail
        return len(old), strategy

    def _hard_collapse(
        self,
        messages: list[dict],
        llm: LLM | None,
        working_memory: WorkingMemory | None,
    ) -> int:
        """Keep recovery context, current request, and a recent legal turn."""
        split = self._safe_split(messages, 4 if len(messages) > 4 else 2)
        if split <= 0:
            return 0
        old = messages[:split]
        tail = messages[split:]
        summary, _ = self._get_summary(old, llm, working_memory)
        replacement = self._summary_pair("[Hard context reset]", summary)

        candidate = replacement + self._special_messages(old, tail, working_memory) + tail
        if estimate_tokens(candidate) >= estimate_tokens(messages):
            return 0
        messages[:] = candidate
        return len(old)

    def _get_summary(
        self,
        messages: list[dict],
        llm: LLM | None,
        working_memory: WorkingMemory | None = None,
    ) -> tuple[str, str]:
        flat = self._flatten(messages)
        memory_text = render_working_memory(working_memory) if working_memory else ""
        if llm:
            try:
                response = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Compress complete prior turns into a brief summary. "
                                "Preserve decisions, errors, recovery notices, and task state. "
                                "Working Memory is authoritative for file freshness."
                            ),
                        },
                        {
                            "role": "user",
                            "content": self._summary_input(flat, memory_text),
                        },
                    ],
                )
                summary = str(response.content).strip()
                if summary:
                    return summary[:4000], "llm-summary"
            except Exception:
                pass
        return self._extract_key_info(messages, working_memory), "local-summary"

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "?")
            text = str(message.get("content", "") or "")
            if text:
                parts.append(f"[{role}] {ContextManager._message_excerpt(text)}")
        return "\n".join(parts)

    @staticmethod
    def _message_excerpt(text: str, limit: int = 400) -> str:
        if len(text) <= limit:
            return text
        key_snippets = [
            match.group().strip()
            for match in re.finditer(
                r".{0,80}(?:decision|error|failed|failure|recovery|blocker).{0,160}",
                text,
                flags=re.IGNORECASE,
            )
        ]
        if key_snippets:
            keys = " ... ".join(key_snippets)
            head_size = max(0, limit - len(keys) - 20)
            return f"{text[:head_size]} ... [key] {keys}"[:limit]
        head_size = limit // 2
        tail_size = limit - head_size - 18
        return f"{text[:head_size]} ... [truncated] ... {text[-tail_size:]}"[:limit]

    @staticmethod
    def _summary_input(flat_turns: str, memory_text: str) -> str:
        turns = ContextManager._bounded_text(
            flat_turns,
            _SUMMARY_TURNS_LIMIT,
            preserve_key_lines=True,
        )
        memory = ContextManager._bounded_text(
            memory_text,
            _SUMMARY_MEMORY_LIMIT,
            preserve_key_lines=False,
        )
        prompt = (
            f"[Prior structured turns]\n{turns}\n\n"
            f"[Working Memory reference]\n{memory}"
        )
        return prompt[:_SUMMARY_INPUT_LIMIT]

    @staticmethod
    def _bounded_text(text: str, limit: int, *, preserve_key_lines: bool) -> str:
        if len(text) <= limit:
            return text
        marker = "\n... [section truncated] ...\n"
        key_text = ""
        if preserve_key_lines:
            key_lines = [
                line for line in text.splitlines()
                if any(token in line.lower() for token in (
                    "decision", "error", "failed", "failure", "recovery", "blocker"
                ))
            ]
            if key_lines:
                key_text = "\n[Preserved key lines]\n" + "\n".join(key_lines)
                key_text = key_text[: min(3_000, limit // 3)]
        remaining = max(0, limit - len(marker) - len(key_text))
        head_size = remaining // 2
        tail_size = remaining - head_size
        tail = text[-tail_size:] if tail_size else ""
        return f"{text[:head_size]}{marker}{key_text}{tail}"[:limit]

    @staticmethod
    def _extract_key_info(
        messages: list[dict],
        working_memory: WorkingMemory | None = None,
    ) -> str:
        files_seen = set()
        errors = []
        for message in messages:
            text = str(message.get("content", "") or "")
            files_seen.update(re.findall(r"[\w./\-]+\.\w{1,5}", text))
            errors.extend(
                line.strip()[:150]
                for line in text.splitlines()
                if "error" in line.lower()
            )
        if working_memory is not None:
            files_seen.update(item.path for item in working_memory.files)
        parts = []
        if files_seen:
            parts.append(f"Files relevant: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"

    @staticmethod
    def _tool_metadata(messages: list[dict]) -> dict[str, dict]:
        metadata: dict[str, dict] = {}
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                metadata[str(call.get("id"))] = {
                    "name": function.get("name", "unknown"),
                    "path": arguments.get("file_path") or arguments.get("path"),
                    "command": arguments.get("command"),
                    "arguments": arguments,
                }
        for message in messages:
            if message.get("role") != "tool":
                continue
            info = metadata.setdefault(str(message.get("tool_call_id")), {})
            match = re.search(
                r"\[exit code:\s*(-?\d+)\]",
                str(message.get("content", "") or ""),
                flags=re.IGNORECASE,
            )
            if match:
                info["exit_code"] = int(match.group(1))
        return metadata

    @staticmethod
    def _compressed_tool_output(content: str, info: dict) -> str:
        lines = content.splitlines()
        error_lines = [
            line for line in lines
            if any(token in line.lower() for token in (
                "error", "failed", "failure", "traceback", "denied", "rejected"
            ))
        ]
        selected = lines[:3] + error_lines[-3:] + lines[-3:]
        selected = list(dict.fromkeys(line for line in selected if line))
        header = [
            "[Tool output compressed]",
            f"tool: {info.get('name', 'unknown')}",
        ]
        if info.get("path"):
            header.append(f"path: {info['path']}")
        if info.get("exit_code") is not None:
            header.append(f"exit_code: {info['exit_code']}")
        header.extend((
            "truncated: true",
            f"original_chars: {len(content)}",
            *selected,
        ))
        compressed = "\n".join(header)
        if len(compressed) < len(content):
            return compressed
        fallback_tail = error_lines[-1] if error_lines else lines[-1] if lines else ""
        fallback_tail = fallback_tail[-500:]
        return "\n".join([
            *header[:6],
            fallback_tail,
        ])

    @staticmethod
    def _summary_pair(label: str, summary: str) -> list[dict]:
        return [
            {"role": "user", "content": f"{label}\n{summary}"},
            {
                "role": "assistant",
                "content": "Context restored. Continuing from the structured state.",
            },
        ]

    @staticmethod
    def _special_messages(
        old: list[dict],
        tail: list[dict],
        working_memory: WorkingMemory | None,
    ) -> list[dict]:
        recovery = next((
            message for message in reversed(old)
            if message.get("role") == "user"
            and str(message.get("content", "")).startswith("[PikaCore recovery:")
        ), None)
        current_request = None
        if working_memory is not None and working_memory.current_request:
            current_request = next((
                message for message in reversed(old)
                if message.get("role") == "user"
                and message.get("content") == working_memory.current_request
            ), None)
        if current_request is None:
            current_request = next((
                message for message in reversed(old)
                if message.get("role") == "user"
                and not str(message.get("content", "")).startswith((
                    "[PikaCore recovery:",
                    "[Context compressed",
                    "[Hard context reset]",
                ))
            ), None)
        preserved = []
        for message in (recovery, current_request):
            if message is not None and message not in preserved and message not in tail:
                preserved.append(message)
        return preserved

    @staticmethod
    def _total_tokens(
        messages: list[dict],
        working_memory: WorkingMemory | None,
    ) -> int:
        memory_tokens = (
            _approx_tokens(render_working_memory(working_memory))
            if working_memory is not None
            else 0
        )
        return estimate_tokens(messages) + memory_tokens

    @staticmethod
    def _message_signature(message: dict) -> str:
        return json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
