"""Shared Qwen tool-calling protocol for DualSearch.

This module deliberately has no training or retrieval dependencies.  The same
schemas and validators are used by rollout, SFT data construction and reward
format checking, which prevents those three paths from quietly drifting apart.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


MAX_QUERY_CHARS = 512

SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the text knowledge base for information needed to answer the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise, standalone text retrieval query.",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS,
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

VISION_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "vision_search",
        "description": "Search visually similar corpus images using one input image and a text hint.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_index": {
                    "type": "integer",
                    "description": "One-based index of the input image to search.",
                    "minimum": 1,
                },
                "query": {
                    "type": "string",
                    "description": "A concise visual retrieval hint grounded in the question.",
                    "minLength": 1,
                    "maxLength": MAX_QUERY_CHARS,
                },
            },
            "required": ["image_index", "query"],
            "additionalProperties": False,
        },
    },
}

DUAL_SEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [SEARCH_TOOL_SCHEMA, VISION_SEARCH_TOOL_SCHEMA]
# A short, discoverable alias used by dataset builders.
TOOL_SCHEMAS = DUAL_SEARCH_TOOL_SCHEMAS


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return a defensive copy suitable for ``apply_chat_template``."""

    return copy.deepcopy(DUAL_SEARCH_TOOL_SCHEMAS)


def canonical_json(value: Any) -> str:
    """Serialize protocol JSON deterministically without ASCII escaping."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_tool_schemas_json() -> str:
    """Return the physical JSON-string representation used by SFT Parquet."""

    return canonical_json(DUAL_SEARCH_TOOL_SCHEMAS)


class ProtocolError(ValueError):
    """Raised when a tool call does not conform to the shared protocol."""


@dataclass(frozen=True)
class ToolCall:
    name: Literal["search", "vision_search"]
    arguments: dict[str, Any]

    @property
    def arguments_json(self) -> str:
        return canonical_json(self.arguments)


@dataclass(frozen=True)
class ParsedAssistantAction:
    kind: Literal["tool", "answer", "invalid"]
    tool_call: ToolCall | None = None
    answer: str | None = None
    error: str | None = None
    # Character offset immediately after the closing action tag.  Rollout uses
    # this only to select an original token prefix; it never decode/re-encodes.
    end_offset: int | None = None
    attempted_tool: bool = False


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_PREFIX_RE = re.compile(r"\s*<think>(.*?)</think>\s*", re.DOTALL)
_LEGACY_ACTION_RE = re.compile(
    r"</?(?:search|vision_search|information|vision_information)(?:\s[^>]*)?>",
    re.IGNORECASE,
)
_PROTOCOL_CONTROL_RE = re.compile(
    r"(?:</?(?:think|tool_call|tool_response|answer|search|vision_search|information|vision_information)"
    r"(?:\s[^>]*)?>|<\|im_(?:start|end)\|>)",
    re.IGNORECASE,
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(payload: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except ProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("tool call payload is not valid JSON") from exc


def _validate_query(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError("query must be a string")
    query = value.strip()
    if not query:
        raise ProtocolError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ProtocolError(f"query exceeds {MAX_QUERY_CHARS} characters")
    if _PROTOCOL_CONTROL_RE.search(query):
        raise ProtocolError("query contains a protocol control tag")
    return query


def validate_tool_call_payload(payload: Any, *, image_count: int | None = None) -> ToolCall:
    """Validate a decoded Qwen ``<tool_call>`` JSON object strictly.

    Besides JSON types, this enforces exact key sets.  In particular, Python's
    ``bool`` is not accepted as an integer image index.
    """

    if isinstance(payload, str):
        payload = _load_strict_json(payload)
    if not isinstance(payload, dict):
        raise ProtocolError("tool call payload must be a JSON object")
    if set(payload) != {"name", "arguments"}:
        raise ProtocolError("tool call must contain exactly name and arguments")

    name = payload["name"]
    arguments = payload["arguments"]
    if name not in {"search", "vision_search"}:
        raise ProtocolError(f"unknown tool: {name!r}")
    if not isinstance(arguments, dict):
        raise ProtocolError("arguments must be a JSON object")

    if name == "search":
        if set(arguments) != {"query"}:
            raise ProtocolError("search arguments must contain exactly query")
        return ToolCall(name="search", arguments={"query": _validate_query(arguments["query"])})

    if set(arguments) != {"image_index", "query"}:
        raise ProtocolError("vision_search arguments must contain exactly image_index and query")
    image_index = arguments["image_index"]
    if isinstance(image_index, bool) or not isinstance(image_index, int):
        raise ProtocolError("image_index must be an integer")
    if image_index < 1:
        raise ProtocolError("image_index must be at least 1")
    if image_count is not None and image_index > image_count:
        raise ProtocolError(f"image_index {image_index} is out of range for {image_count} input images")
    return ToolCall(
        name="vision_search",
        arguments={"image_index": image_index, "query": _validate_query(arguments["query"])},
    )


def _invalid(error: str, *, attempted_tool: bool = False) -> ParsedAssistantAction:
    return ParsedAssistantAction(kind="invalid", error=error, attempted_tool=attempted_tool)


def parse_assistant_action(text: str, *, image_count: int | None = None) -> ParsedAssistantAction:
    """Parse exactly one ``think + action`` assistant turn.

    A turn is either ``<think>...</think><tool_call>JSON</tool_call>`` or
    ``<think>...</think><answer>...</answer>``.  Legacy action tags, mixed
    answer/tool output, multiple calls and content after the action are rejected.
    """

    if not isinstance(text, str):
        return _invalid("assistant output must be text")
    if _LEGACY_ACTION_RE.search(text):
        return _invalid("legacy search action tags are not supported", attempted_tool=True)

    tool_open_count = text.count("<tool_call>")
    tool_close_count = text.count("</tool_call>")
    answer_open_count = text.count("<answer>")
    answer_close_count = text.count("</answer>")
    attempted_tool = tool_open_count > 0 or tool_close_count > 0 or "<tool_call" in text

    if tool_open_count != tool_close_count:
        return _invalid("mismatched tool_call tags", attempted_tool=attempted_tool)
    if answer_open_count != answer_close_count:
        return _invalid("mismatched answer tags", attempted_tool=attempted_tool)
    if tool_open_count and answer_open_count:
        return _invalid("a turn cannot mix a tool call and an answer", attempted_tool=True)
    if tool_open_count > 1:
        return _invalid("only one tool call is allowed per turn", attempted_tool=True)
    if answer_open_count > 1:
        return _invalid("only one answer is allowed per turn")

    think_match = _THINK_PREFIX_RE.match(text)
    if think_match is None or not think_match.group(1).strip():
        return _invalid("each assistant turn must start with a non-empty think block", attempted_tool=attempted_tool)
    remainder_start = think_match.end()
    remainder = text[remainder_start:]

    if tool_open_count == 1:
        match = _TOOL_CALL_RE.fullmatch(remainder.strip())
        if match is None:
            return _invalid("unexpected content around tool_call", attempted_tool=True)
        try:
            call = validate_tool_call_payload(match.group(1), image_count=image_count)
        except ProtocolError as exc:
            return _invalid(str(exc), attempted_tool=True)
        # The full-match above permits only whitespace after the closing tag.
        # Treat that whitespace as part of the accepted action so rollout keeps
        # a following hidden Qwen ``<|im_end|>`` token instead of truncating the
        # assistant role boundary at the visible closing tag.
        absolute_end = len(text)
        return ParsedAssistantAction(
            kind="tool",
            tool_call=call,
            end_offset=absolute_end,
            attempted_tool=True,
        )

    if answer_open_count == 1:
        match = _ANSWER_RE.fullmatch(remainder.strip())
        if match is None:
            return _invalid("unexpected content around answer")
        answer = match.group(1).strip()
        if not answer:
            return _invalid("answer must not be empty")
        if _PROTOCOL_CONTROL_RE.search(answer):
            return _invalid("answer contains a protocol control tag")
        absolute_end = len(text)
        return ParsedAssistantAction(kind="answer", answer=answer, end_offset=absolute_end)

    return _invalid("assistant turn contains neither a tool call nor an answer", attempted_tool=attempted_tool)


def sanitize_tool_response(content: Any) -> str:
    """Render untrusted retrieval content without allowing protocol injection."""

    text = str(content or "")
    # Escape the whole matched tag, including case/whitespace/attribute
    # variants such as ``<TOOL_CALL type=x>``.  Replacing only the canonical
    # literal would leave an avoidable prompt/format-validator injection path.
    return _PROTOCOL_CONTROL_RE.sub(
        lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text,
    )


def _split_title_text(content: Any) -> tuple[str, str]:
    text = str(content or "")
    if not text:
        return "", ""
    parts = text.split("\n")
    return parts[0].strip().strip('"'), "\n".join(parts[1:]).strip()


def _format_retrieval_results(results: Sequence[Mapping[str, Any]], label: str) -> str:
    lines: list[str] = []
    for index, item in enumerate(results):
        document = item.get("document", item)
        if not isinstance(document, Mapping):
            document = {"contents": str(document)}
        title, text = _split_title_text(document.get("contents", ""))
        lines.append(f"{label} {index + 1}(Title: {title}) {text}".rstrip())
    return sanitize_tool_response("\n".join(lines))


def format_text_results(results: Sequence[Mapping[str, Any]]) -> str:
    return _format_retrieval_results(results, "Doc")


def format_vision_results(results: Sequence[Mapping[str, Any]]) -> str:
    return _format_retrieval_results(results, "Caption")


_THINK_AT_RE = re.compile(r"\s*<think>(.*?)</think>\s*", re.DOTALL)
_TOOL_AT_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>\s*", re.DOTALL)
_TOOL_RESPONSE_AT_RE = re.compile(r"<tool_response>(.*?)</tool_response>\s*", re.DOTALL)
_ANSWER_AT_RE = re.compile(r"<answer>(.*?)</answer>\s*", re.DOTALL)
# Qwen's chat template renders tool observations as a ``user`` turn and emits
# the next generation prefix as an ``assistant`` turn.  Decoding with
# ``skip_special_tokens=True`` removes the surrounding ``<|im_*|>`` tokens but
# intentionally leaves these two exact, newline-terminated role names behind.
_QWEN_TOOL_RESPONSE_ROLE = "user\n"
_QWEN_ASSISTANT_GENERATION_ROLE = "assistant\n"


def validate_sequence(text: str) -> tuple[bool, str]:
    """Validate a complete flattened native-Qwen DualSearch trajectory."""

    if not isinstance(text, str):
        return False, "solution must be text"
    if _LEGACY_ACTION_RE.search(text):
        return False, "legacy search action tags are not supported"

    cursor = 0
    tool_calls = 0
    while True:
        think = _THINK_AT_RE.match(text, cursor)
        if think is None or not think.group(1).strip():
            return False, f"expected a non-empty think block at offset {cursor}"
        cursor = think.end()

        tool = _TOOL_AT_RE.match(text, cursor)
        if tool is not None:
            try:
                validate_tool_call_payload(tool.group(1))
            except ProtocolError as exc:
                return False, f"invalid tool call: {exc}"
            cursor = tool.end()
            # Consume the decoded Qwen role only at the structural transition
            # from an assistant tool call to its tool response.  In particular,
            # do not normalize role words globally: retrieval content may
            # legitimately contain either ``user`` or ``assistant``.
            if text.startswith(_QWEN_TOOL_RESPONSE_ROLE, cursor):
                cursor += len(_QWEN_TOOL_RESPONSE_ROLE)
            response = _TOOL_RESPONSE_AT_RE.match(text, cursor)
            if response is None:
                return False, f"expected tool_response after tool call at offset {cursor}"
            if _PROTOCOL_CONTROL_RE.search(response.group(1)):
                return False, "tool_response contains an unescaped protocol control tag"
            cursor = response.end()
            # Likewise, the decoded generation-prefix role is legal only after
            # a complete tool response and immediately before the next
            # assistant turn parsed by the loop.
            if text.startswith(_QWEN_ASSISTANT_GENERATION_ROLE, cursor):
                cursor += len(_QWEN_ASSISTANT_GENERATION_ROLE)
            tool_calls += 1
            continue

        answer = _ANSWER_AT_RE.match(text, cursor)
        if answer is None:
            return False, f"expected a tool call or final answer at offset {cursor}"
        if not answer.group(1).strip():
            return False, "answer must not be empty"
        if _PROTOCOL_CONTROL_RE.search(answer.group(1)):
            return False, "answer contains a nested protocol control tag"
        cursor = answer.end()
        if text[cursor:].strip():
            return False, "unexpected content after final answer"
        return True, f"valid native tool trajectory with {tool_calls} tool call(s)"


def extract_answer(text: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(text or ""))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer or None
