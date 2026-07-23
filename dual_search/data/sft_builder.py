"""Build verified DualSearch cold-start SFT trajectories.

The builder consumes *already materialized* EVQA train/corpus artifacts.  It
does not download datasets, build retrieval indexes, or call the real
retrieval services.  Instead it constructs deterministic oracle observations
from the held-out-safe corpora and asks an OpenAI-compatible multimodal teacher
for three small JSON decisions:

``vision_search -> search -> answer``.

Only single-hop, retrieval-resolvable training examples are eligible.  Teacher
failures are skipped and deterministically supplemented from the same question
type until that stratum reaches its target or runs out of image groups.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from dual_search.protocol import (
    ProtocolError,
    canonical_json,
    canonical_tool_schemas_json,
    format_text_results,
    format_vision_results,
    sanitize_tool_response,
    validate_tool_call_payload,
)


RESERVED_TAG_RE = re.compile(
    r"</?(?:think|tool_call|tool_response|answer|search|vision_search)(?:\s[^>]*)?>|<\|im_(?:start|end)\|>",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MULTI_HOP_MARKERS = {"2", "2hop", "twohop", "multihop", "multiplehop"}
DEFAULT_TOOL_RESPONSE_TOKENS = 500
DEFAULT_TOOL_WRAPPER_TOKEN_RESERVE = 32
SFT_PARQUET_COLUMNS = (
    "data_source",
    "messages",
    "tools",
    "images",
    "sample_id",
    "image_key",
    "category_key",
    "question_type",
    "retrieval_resolvable",
    "extra_info",
)


VISION_STAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "think": {"type": "string", "minLength": 1},
        "vision_search": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "required": ["think", "vision_search"],
    "additionalProperties": False,
}

SEARCH_STAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "think": {"type": "string", "minLength": 1},
        "search": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "required": ["think", "search"],
    "additionalProperties": False,
}

ANSWER_STAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "think": {"type": "string", "minLength": 1},
        "answer": {"type": "string", "minLength": 1},
    },
    "required": ["think", "answer"],
    "additionalProperties": False,
}


class TeacherClient(Protocol):
    """Minimal interface used by the pure trajectory builder."""

    def generate(
        self,
        *,
        stage: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TeacherConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    temperature: float = 0.0
    max_tokens: int = 768


@dataclass(frozen=True)
class SFTBuilderConfig:
    sample_fraction: float = 0.05
    validation_fraction: float = 0.10
    seed: int = 42
    oracle_top_k: int = 3
    image_index: int = 1
    data_source: str = "dual_search_sft"
    max_tool_response_tokens: int = DEFAULT_TOOL_RESPONSE_TOKENS
    fallback_wrapper_token_reserve: int = DEFAULT_TOOL_WRAPPER_TOKEN_RESERVE

    def __post_init__(self) -> None:
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.oracle_top_k < 2:
            raise ValueError("oracle_top_k must be at least 2")
        if self.image_index < 1:
            raise ValueError("image_index must be one-based")
        if self.max_tool_response_tokens <= 0:
            raise ValueError("max_tool_response_tokens must be positive")
        if not 0 <= self.fallback_wrapper_token_reserve < self.max_tool_response_tokens:
            raise ValueError("fallback_wrapper_token_reserve must be smaller than max_tool_response_tokens")


@dataclass
class SFTBuildResult:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    report: dict[str, Any]


class TeacherRequestError(RuntimeError):
    """Teacher transport error after retrying transient failures."""


class TrajectoryBuildError(ValueError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


class VLLMTeacherClient:
    """Strict JSON client for a local/remote vLLM OpenAI-compatible server."""

    _TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: TeacherConfig, session: Any | None = None):
        self.config = config
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def generate(
        self,
        *,
        stage: str,
        messages: list[dict[str, Any]],
        response_schema: dict[str, Any],
    ) -> Mapping[str, Any]:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/v1/chat/completions"):
            url = base_url
        elif base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"dual_search_{stage}",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }

        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
            except Exception as exc:
                if attempt + 1 >= attempts:
                    raise TeacherRequestError(f"teacher request failed: {exc}") from exc
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                continue

            if response.status_code in self._TRANSIENT_STATUS:
                if attempt + 1 >= attempts:
                    raise TeacherRequestError(
                        f"teacher returned transient HTTP {response.status_code} after {attempts} attempts"
                    )
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                continue
            try:
                response.raise_for_status()
            except Exception as exc:
                raise TeacherRequestError(f"teacher returned HTTP {response.status_code}") from exc

            try:
                body = response.json()
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise TeacherRequestError("teacher response is missing choices[0].message.content") from exc
            if not isinstance(content, str):
                raise TeacherRequestError("teacher message content must be a JSON string")
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError as exc:
                # Schema/format failures are deliberately not retried.
                raise TrajectoryBuildError(stage, "teacher returned malformed JSON") from exc
            if not isinstance(decoded, dict):
                raise TrajectoryBuildError(stage, "teacher JSON must be an object")
            return decoded

        raise AssertionError("unreachable")


def _stable_key(seed: int, *parts: Any) -> str:
    material = canonical_json([seed, *[str(part) for part in parts]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return True
    try:
        result = value != value
        if hasattr(result, "item"):
            result = result.item()
        return bool(result) if isinstance(result, bool) else False
    except Exception:
        return False


def _plain(value: Any) -> Any:
    """Convert pandas/Arrow/numpy containers to ordinary Python values."""

    if _is_missing(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if not _is_missing(item)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _plain(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _lookup(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = record.get(key)
    if not _is_missing(value):
        return _plain(value)
    extra_info = _plain(record.get("extra_info")) or {}
    if isinstance(extra_info, Mapping):
        value = extra_info.get(key)
        if not _is_missing(value):
            return _plain(value)
    return default


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
                if isinstance(decoded, list):
                    return decoded
            except json.JSONDecodeError:
                pass
        # EVQA uses ``|`` for answer variants while ``&&`` joins components
        # of one multi-answer target. Preserve the latter as one value.
        return [part.strip() for part in stripped.split("|") if part.strip()]
    return [value]


def _normalized_hop_marker(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _is_explicit_multi_hop(value: Any) -> bool:
    for marker in _as_list(value):
        if isinstance(marker, (int, float)) and not isinstance(marker, bool):
            try:
                if float(marker) == 2:
                    return True
            except (TypeError, ValueError):
                pass
        normalized = _normalized_hop_marker(marker)
        if normalized in MULTI_HOP_MARKERS:
            return True
        if normalized.startswith("2hop") or normalized.startswith("twohop"):
            return True
    return False


def _referenced_page_identities(record: Mapping[str, Any]) -> set[str]:
    """Collect explicit Wikipedia/page identities without guessing from prose."""

    urls: set[str] = set()
    titles: set[str] = set()
    wiki_pairs = _lookup(record, "wiki_pairs", [])
    for pair in _as_list(wiki_pairs):
        if not isinstance(pair, Mapping):
            continue
        url = _normalize_url(pair.get("normalized_url") or pair.get("url"))
        title = str(pair.get("title") or "").strip().casefold()
        if url:
            urls.add(url)
        elif title:
            titles.add(title)

    for key in ("wikipedia_url", "wikipedia_urls", "page_url", "page_urls"):
        for value in _as_list(_lookup(record, key, [])):
            normalized = _normalize_url(value)
            if normalized:
                urls.add(normalized)

    # Some exported EVQA variants materialize page/title lists without URLs.
    for key in ("wikipedia_title", "wikipedia_titles", "wikipedia_page", "wikipedia_pages", "page_ids"):
        values = [str(value).strip().casefold() for value in _as_list(_lookup(record, key, []))]
        for value in values:
            if value:
                titles.add(value)
    # Do not add URL and title counts together because they commonly describe
    # the same page. Use whichever representation exposes more distinct pages.
    url_identities = {f"url:{value}" for value in urls}
    title_identities = {f"title:{value}" for value in titles}
    return title_identities if len(title_identities) > len(url_identities) else url_identities


def is_multi_hop_record(record: Mapping[str, Any]) -> bool:
    """Recognize multi-hop rows across EVQA schema variants.

    ``question_type`` remains untouched and is still used as the sampling
    stratum.  This predicate only controls SFT eligibility.
    """

    if _is_explicit_multi_hop(_lookup(record, "question_type", "")):
        return True
    if _is_explicit_multi_hop(_lookup(record, "hop_type", "")):
        return True
    return len(_referenced_page_identities(record)) > 1


def _normalize_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    scheme = parts.scheme.lower()
    host = parts.netloc.lower()
    path = re.sub(r"/+", "/", parts.path).rstrip("/")
    return urlunsplit((scheme, host, path, parts.query, ""))


def _first_image_path(record: Mapping[str, Any]) -> str:
    images = _as_list(record.get("images"))
    if not images:
        image = _lookup(record, "image", "")
        return str(image or "").strip()
    first = _plain(images[0])
    if isinstance(first, Mapping):
        return str(first.get("image") or first.get("path") or first.get("image_url") or "").strip()
    return str(first or "").strip()


def _gold_answers(record: Mapping[str, Any]) -> list[str]:
    reward_model = _plain(record.get("reward_model")) or {}
    if isinstance(reward_model, Mapping):
        ground_truth = reward_model.get("ground_truth") or {}
        if isinstance(ground_truth, Mapping):
            targets = _as_list(ground_truth.get("target"))
            values = [str(target).strip() for target in targets if str(target).strip()]
            if values:
                return values
    return [str(answer).strip() for answer in _as_list(_lookup(record, "answer", "")) if str(answer).strip()]


def _student_prompt(record: Mapping[str, Any], question: str) -> str:
    prompt = _plain(record.get("prompt"))
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    return (
        "<image>\nAnswer the question about the image. Use the provided tools when external visual "
        "or textual evidence is needed, call at most one tool in each assistant turn, reason inside "
        "<think>...</think>, and put the final answer "
        f"inside <answer>...</answer>.\nQuestion: {question}"
    )


def _image_as_data_url(path_or_url: str) -> str:
    if path_or_url.startswith(("data:", "http://", "https://")):
        return path_or_url
    path = Path(path_or_url).expanduser()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TrajectoryBuildError("preflight", f"cannot read query image: {path}") from exc
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _flat_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if not isinstance(value, list):
        raise TypeError(f"tokenizer returned unsupported token IDs: {type(value).__name__}")
    return [int(token.item() if hasattr(token, "item") else token) for token in value]


def _truncate_utf8_bytes(text: str, byte_budget: int) -> str:
    if byte_budget <= 0:
        return ""
    payload = text.encode("utf-8")
    if len(payload) <= byte_budget:
        return text
    return payload[:byte_budget].decode("utf-8", errors="ignore")


def _lightweight_apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Dependency-light equivalent of ``verl.utils.chat_template``.

    The fallback is important for templates such as Qwen3.5 that reject a
    conversation without a user turn. It intentionally mirrors veRL's exact
    dummy-user prefix removal behavior.
    """

    kwargs = dict(chat_template_kwargs or {})
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=None,
            return_dict=False,
            **kwargs,
        )
        return _flat_token_ids(rendered)
    except Exception:
        dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_prefix = tokenizer.apply_chat_template(
            dummy_user,
            tokenize=True,
            add_generation_prompt=False,
            tools=None,
            return_dict=False,
            **kwargs,
        )
        output = tokenizer.apply_chat_template(
            dummy_user + messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=None,
            return_dict=False,
            **kwargs,
        )
        prefix_ids = _flat_token_ids(dummy_prefix)
        return _flat_token_ids(output)[len(prefix_ids) :]


def _lightweight_system_prompt_ids(
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Dependency-light equivalent of ``initialize_system_prompt``."""

    kwargs = dict(chat_template_kwargs or {})
    one_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=False,
        tokenize=True,
        **kwargs,
    )
    two_users = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        tokenize=True,
        **kwargs,
    )
    one_user_ids = _flat_token_ids(one_user)
    two_user_ids = _flat_token_ids(two_users)
    added_user_length = len(two_user_ids) - len(one_user_ids)
    return one_user_ids[:-added_user_length] if added_user_length > 0 else []


def truncate_tool_observation(
    content: str,
    *,
    tokenizer: Any | None = None,
    max_tokens: int = DEFAULT_TOOL_RESPONSE_TOKENS,
    fallback_wrapper_token_reserve: int = DEFAULT_TOOL_WRAPPER_TOKEN_RESERVE,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Apply the rollout's native tool-response token budget to SFT content.

    With a Qwen tokenizer this mirrors ``DualSearchAgentLoop``: tokenize the
    visible content, render it as a native role=tool message plus the next
    assistant generation prefix, remove the inferred system prompt, then
    binary-search the longest prefix whose runtime-visible length is at most
    ``max_tokens``.

    Without a tokenizer, the conservative fallback budgets UTF-8 bytes after a
    fixed wrapper reserve. Qwen's byte-level tokenizer cannot produce more
    content tokens than UTF-8 bytes, making this safe (though intentionally
    conservative) for code-only/fixture dataset construction.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= fallback_wrapper_token_reserve < max_tokens:
        raise ValueError("fallback_wrapper_token_reserve must be smaller than max_tokens")
    content = sanitize_tool_response(content)
    if tokenizer is None:
        return _truncate_utf8_bytes(content, max_tokens - fallback_wrapper_token_reserve)

    content_ids = _flat_token_ids(tokenizer.encode(content, add_special_tokens=False))[:max_tokens]

    def decode(candidate_ids: Sequence[int]) -> str:
        return str(tokenizer.decode(list(candidate_ids), skip_special_tokens=True))

    if getattr(tokenizer, "apply_chat_template", None) is None:
        return decode(content_ids[: max(0, max_tokens - fallback_wrapper_token_reserve)])

    system_prompt_ids = _lightweight_system_prompt_ids(tokenizer, chat_template_kwargs)

    def rendered_length(candidate_ids: Sequence[int]) -> int:
        candidate = decode(candidate_ids)
        rendered = _lightweight_apply_chat_template(
            tokenizer,
            [{"role": "tool", "content": candidate}],
            add_generation_prompt=True,
            chat_template_kwargs=chat_template_kwargs,
        )
        # Mirrors AgentLoopBase.apply_chat_template(remove_system_prompt=True).
        return len(rendered[len(system_prompt_ids) :])

    if rendered_length(content_ids) <= max_tokens:
        return decode(content_ids)
    if rendered_length([]) > max_tokens:
        raise ValueError("max_tokens is smaller than the native tool-response wrapper")

    low, high = 0, len(content_ids)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if rendered_length(content_ids[:middle]) <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return decode(content_ids[:best])


def _teacher_messages(
    *,
    stage: str,
    sample_id: str,
    question: str,
    answers: Sequence[str],
    image_path: str,
    image_index: int = 1,
    vision_observation: str | None = None,
    text_observation: str | None = None,
    vision_think: str | None = None,
    vision_query: str | None = None,
    search_think: str | None = None,
    search_query: str | None = None,
) -> list[dict[str, Any]]:
    common = (
        "Return exactly one JSON object matching the supplied response schema. The official answer is planning "
        "context, not permission to reveal it early. Before the final stage, do not copy an answer or hidden "
        "entity name unless it already appears in the question or an observation. Do not emit XML/protocol tags."
    )
    context = {
        "sample_id": sample_id,
        "stage": stage,
        "question": question,
    }
    if vision_observation is not None:
        context["vision_observation"] = vision_observation
    if text_observation is not None:
        context["text_observation"] = text_observation

    if stage == "vision":
        instruction = (
            "Describe concise reasoning and a visual retrieval hint. The query should use visible attributes and "
            "the information need, without naming an unrevealed target entity or answer."
        )
    elif stage == "search":
        instruction = (
            "Using the visual observation, produce concise reasoning and a standalone text knowledge-base query."
        )
    elif stage == "answer":
        instruction = "Using both observations, produce concise reasoning and the final answer."
    else:
        raise ValueError(f"unknown teacher stage: {stage}")

    # Every stage is a separate HTTP request, so include the actual query image
    # each time rather than relying on server-side conversational state.
    initial_user_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _image_as_data_url(image_path)}},
        {
            "type": "text",
            "text": (
                f"Question: {question}\nOfficial answers for planning only: "
                f"{json.dumps(list(answers), ensure_ascii=False)}"
            ),
        },
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": common},
        {"role": "user", "content": initial_user_content},
    ]

    def append_tool_history(
        *,
        think: str,
        name: str,
        arguments: Mapping[str, Any],
        observation: str,
    ) -> None:
        tool_call_id = f"dual_search_{name}"
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{think}</think>",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": canonical_json(arguments)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": tool_call_id,
                "content": observation,
            }
        )

    if stage in {"search", "answer"}:
        if not all((vision_think, vision_query, vision_observation)):
            raise ValueError("search/answer teacher stages require complete vision tool history")
        append_tool_history(
            think=str(vision_think),
            name="vision_search",
            arguments={"image_index": image_index, "query": str(vision_query)},
            observation=str(vision_observation),
        )
    if stage == "answer":
        if not all((search_think, search_query, text_observation)):
            raise ValueError("answer teacher stage requires complete text tool history")
        append_tool_history(
            think=str(search_think),
            name="search",
            arguments={"query": str(search_query)},
            observation=str(text_observation),
        )

    stage_request = [
        {
            "type": "text",
            "text": f"{instruction}\nContext:\n{json.dumps(context, ensure_ascii=False, indent=2)}",
        }
    ]
    if stage == "vision":
        # Keep the first stage as one natural multimodal user turn. Later
        # stages need a new user turn after their structured tool history.
        initial_user_content.extend(stage_request)
    else:
        messages.append({"role": "user", "content": stage_request})
    return messages


def _strict_stage_output(stage: str, value: Mapping[str, Any]) -> tuple[str, str]:
    expected = {
        "vision": ({"think", "vision_search"}, "vision_search"),
        "search": ({"think", "search"}, "search"),
        "answer": ({"think", "answer"}, "answer"),
    }
    keys, payload_key = expected[stage]
    if not isinstance(value, Mapping) or set(value) != keys:
        raise TrajectoryBuildError(stage, f"teacher JSON keys must be exactly {sorted(keys)}")
    think = value.get("think")
    payload = value.get(payload_key)
    if stage in {"vision", "search"}:
        if not isinstance(payload, Mapping) or set(payload) != {"query"}:
            raise TrajectoryBuildError(stage, f"{payload_key} must contain exactly query")
        output = payload.get("query")
    else:
        output = payload
    if not isinstance(think, str) or not think.strip():
        raise TrajectoryBuildError(stage, "think must be a non-empty string")
    if not isinstance(output, str) or not output.strip():
        raise TrajectoryBuildError(stage, f"{payload_key} must be a non-empty string")
    if RESERVED_TAG_RE.search(think) or RESERVED_TAG_RE.search(output):
        raise TrajectoryBuildError(stage, "teacher output contains a reserved protocol tag")
    return think.strip(), output.strip()


def _normalized_phrase(value: Any) -> str:
    return " ".join(WORD_RE.sub(" ", str(value or "").casefold()).split())


def _normalized_tokens(value: Any) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_RE.findall(str(value or "")))


def _contains_token_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(tuple(tokens[index : index + width]) == tuple(phrase) for index in range(len(tokens) - width + 1))


def _check_early_leak(
    *,
    stage: str,
    generated: Sequence[str],
    protected_phrases: Sequence[str],
    visible_context: str,
) -> None:
    visible_tokens = _normalized_tokens(visible_context)
    generated_tokens = [_normalized_tokens(value) for value in generated]
    for raw_phrase in protected_phrases:
        phrase_tokens = _normalized_tokens(raw_phrase)
        # Single-character targets such as multiple-choice labels are too
        # ambiguous to use as an early-leak signal.
        if len("".join(phrase_tokens)) < 2:
            continue
        if _contains_token_phrase(visible_tokens, phrase_tokens):
            continue
        if any(_contains_token_phrase(tokens, phrase_tokens) for tokens in generated_tokens):
            raise TrajectoryBuildError(stage, f"early leakage of protected phrase: {raw_phrase!r}")


def _document_id(document: Mapping[str, Any]) -> str:
    return str(document.get("id") or document.get("image_key") or document.get("section_id") or "").strip()


def _deterministic_pick(
    candidates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    seed: int,
    sample_id: str,
    purpose: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        (_plain(candidate) for candidate in candidates),
        key=lambda item: (_stable_key(seed, sample_id, purpose, _document_id(item)), _document_id(item)),
    )
    return [dict(item) for item in ordered[:count]]


def _oracle_vision_documents(
    record: Mapping[str, Any],
    vision_corpus: Sequence[Mapping[str, Any]],
    *,
    config: SFTBuilderConfig,
) -> list[dict[str, Any]]:
    sample_id = str(_lookup(record, "sample_id", "")).strip()
    category_key = str(_lookup(record, "category_key", "")).strip()
    image_key = str(_lookup(record, "image_key", "")).strip()
    positives = [
        doc
        for doc in vision_corpus
        if str(doc.get("category_key", "")) == category_key and str(doc.get("image_key", "")) != image_key
    ]
    negatives = [doc for doc in vision_corpus if str(doc.get("category_key", "")) != category_key]
    if not positives:
        raise TrajectoryBuildError("oracle_vision", "no non-heldout positive visual candidate")
    negative_count = config.oracle_top_k - 1
    if len(negatives) < negative_count:
        raise TrajectoryBuildError("oracle_vision", f"need {negative_count} visual distractors")
    selected = _deterministic_pick(
        positives, 1, seed=config.seed, sample_id=sample_id, purpose="vision_positive"
    ) + _deterministic_pick(
        negatives,
        negative_count,
        seed=config.seed,
        sample_id=sample_id,
        purpose="vision_distractor",
    )
    return sorted(selected, key=lambda doc: _stable_key(config.seed, sample_id, "vision_position", _document_id(doc)))


def _text_match_score(record: Mapping[str, Any], document: Mapping[str, Any]) -> int:
    section_ids = {str(value).strip() for value in _as_list(_lookup(record, "evidence_section_id", []))}
    urls = {_normalize_url(value) for value in _as_list(_lookup(record, "wikipedia_url", []))}
    titles = {str(value).strip().casefold() for value in _as_list(_lookup(record, "wikipedia_title", []))}
    evidence = str(_lookup(record, "evidence", "")).strip().casefold()
    doc_ids = {
        str(document.get("id", "")).strip(),
        str(document.get("section_id", "")).strip(),
    }
    score = 0
    if section_ids and any(value and value in section_ids for value in doc_ids):
        score += 16
    if urls and _normalize_url(document.get("url")) in urls:
        score += 8
    if titles and str(document.get("title", "")).strip().casefold() in titles:
        score += 4
    if evidence and evidence in str(document.get("contents", "")).casefold():
        score += 2
    return score


def _is_exact_text_evidence(record: Mapping[str, Any], document: Mapping[str, Any]) -> bool:
    """Require the configured evidence section, not merely the same page.

    A URL/title match is useful for ranking, but accepting it on its own when
    an evidence-section ID or evidence text is available can create a falsely
    supervised Oracle trace. The RL data's ``text_resolvable`` flag already
    promises that the exact evidence exists; this is the corresponding hard
    assertion at SFT construction time.
    """

    section_ids = {
        str(value).strip()
        for value in _as_list(_lookup(record, "evidence_section_id", []))
        if str(value).strip()
    }
    document_ids = {
        str(document.get("id", "")).strip(),
        str(document.get("section_id", "")).strip(),
    }
    if section_ids:
        return any(value in section_ids for value in document_ids if value)

    evidence = str(_lookup(record, "evidence", "")).strip().casefold()
    if evidence:
        return evidence in str(document.get("contents", "")).casefold()

    urls = {_normalize_url(value) for value in _as_list(_lookup(record, "wikipedia_url", []))}
    if any(urls):
        return _normalize_url(document.get("url")) in urls
    titles = {
        str(value).strip().casefold()
        for value in _as_list(_lookup(record, "wikipedia_title", []))
        if str(value).strip()
    }
    return bool(titles and str(document.get("title", "")).strip().casefold() in titles)


def _oracle_text_documents(
    record: Mapping[str, Any],
    text_corpus: Sequence[Mapping[str, Any]],
    *,
    config: SFTBuilderConfig,
) -> list[dict[str, Any]]:
    sample_id = str(_lookup(record, "sample_id", "")).strip()
    positives = [doc for doc in text_corpus if _is_exact_text_evidence(record, doc)]
    if not positives:
        raise TrajectoryBuildError("oracle_text", "no matching evidence section in text corpus")
    positive = sorted(
        positives,
        key=lambda doc: (
            -_text_match_score(record, doc),
            _stable_key(config.seed, sample_id, "text_positive", _document_id(doc)),
        ),
    )[0]
    positive_id = _document_id(positive)
    positive_url = _normalize_url(positive.get("url"))
    negatives = [
        doc
        for doc in text_corpus
        if _document_id(doc) != positive_id and _normalize_url(doc.get("url")) != positive_url
    ]
    negative_count = config.oracle_top_k - 1
    if len(negatives) < negative_count:
        raise TrajectoryBuildError("oracle_text", f"need {negative_count} text distractors")
    selected = [dict(_plain(positive))] + _deterministic_pick(
        negatives,
        negative_count,
        seed=config.seed,
        sample_id=sample_id,
        purpose="text_distractor",
    )
    return sorted(selected, key=lambda doc: _stable_key(config.seed, sample_id, "text_position", _document_id(doc)))


def _entity_aliases(record: Mapping[str, Any], positive_vision_doc: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("entity_aliases", "category_title", "wikipedia_title"):
        values.extend(_as_list(_lookup(record, key, [])))
    values.extend(_as_list(positive_vision_doc.get("aliases")))
    if positive_vision_doc.get("title"):
        values.append(positive_vision_doc["title"])
    contents = str(positive_vision_doc.get("contents", ""))
    if contents:
        values.append(contents.split("\n", 1)[0].strip().strip('"'))
    seen: set[str] = set()
    aliases: list[str] = []
    for value in values:
        text = str(value or "").strip()
        normalized = _normalized_phrase(text)
        if text and normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(text)
    return aliases


def _assistant_tool_message(*, think: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": f"<think>{think}</think>",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    # Keep this as a canonical JSON string. Qwen's native
                    # template accepts both objects and strings, while Arrow
                    # cannot safely union heterogeneous argument structs.
                    "arguments": canonical_json(arguments),
                },
            }
        ],
    }


def _build_trajectory(
    record: Mapping[str, Any],
    vision_corpus: Sequence[Mapping[str, Any]],
    text_corpus: Sequence[Mapping[str, Any]],
    teacher: TeacherClient,
    config: SFTBuilderConfig,
    observation_tokenizer: Any | None = None,
) -> dict[str, Any]:
    sample_id = str(_lookup(record, "sample_id", "")).strip()
    image_key = str(_lookup(record, "image_key", "")).strip()
    question = str(_lookup(record, "question", "")).strip()
    question_type = str(_lookup(record, "question_type", "")).strip()
    image_path = _first_image_path(record)
    answers = _gold_answers(record)
    if not all((sample_id, image_key, question, question_type, image_path)) or not answers:
        raise TrajectoryBuildError("preflight", "sample is missing required SFT fields")

    vision_documents = _oracle_vision_documents(record, vision_corpus, config=config)
    text_documents = _oracle_text_documents(record, text_corpus, config=config)
    vision_result_text = format_vision_results([{"document": doc} for doc in vision_documents])
    vision_observation = truncate_tool_observation(
        f"image_index={config.image_index}:\n{vision_result_text}".rstrip(),
        tokenizer=observation_tokenizer,
        max_tokens=config.max_tool_response_tokens,
        fallback_wrapper_token_reserve=config.fallback_wrapper_token_reserve,
    )
    text_observation = truncate_tool_observation(
        format_text_results([{"document": doc} for doc in text_documents]),
        tokenizer=observation_tokenizer,
        max_tokens=config.max_tool_response_tokens,
        fallback_wrapper_token_reserve=config.fallback_wrapper_token_reserve,
    )
    category_key = str(_lookup(record, "category_key", ""))
    positive_vision_document = next(
        document for document in vision_documents if str(document.get("category_key", "")) == category_key
    )
    protected = [*answers, *_entity_aliases(record, positive_vision_document)]

    try:
        vision_raw = teacher.generate(
            stage="vision",
            messages=_teacher_messages(
                stage="vision",
                sample_id=sample_id,
                question=question,
                answers=answers,
                image_path=image_path,
                image_index=config.image_index,
            ),
            response_schema=VISION_STAGE_SCHEMA,
        )
    except TeacherRequestError as exc:
        raise TrajectoryBuildError("vision", "teacher transport failure") from exc
    vision_think, vision_query = _strict_stage_output("vision", vision_raw)
    try:
        vision_query = validate_tool_call_payload(
            {
                "name": "vision_search",
                "arguments": {"image_index": config.image_index, "query": vision_query},
            },
            image_count=1,
        ).arguments["query"]
    except ProtocolError as exc:
        raise TrajectoryBuildError("vision", f"invalid vision_search query: {exc}") from exc
    _check_early_leak(
        stage="vision",
        generated=[vision_think, vision_query],
        protected_phrases=protected,
        visible_context=question,
    )

    try:
        search_raw = teacher.generate(
            stage="search",
            messages=_teacher_messages(
                stage="search",
                sample_id=sample_id,
                question=question,
                answers=answers,
                image_path=image_path,
                image_index=config.image_index,
                vision_observation=vision_observation,
                vision_think=vision_think,
                vision_query=vision_query,
            ),
            response_schema=SEARCH_STAGE_SCHEMA,
        )
    except TeacherRequestError as exc:
        raise TrajectoryBuildError("search", "teacher transport failure") from exc
    search_think, search_query = _strict_stage_output("search", search_raw)
    try:
        search_query = validate_tool_call_payload(
            {"name": "search", "arguments": {"query": search_query}}
        ).arguments["query"]
    except ProtocolError as exc:
        raise TrajectoryBuildError("search", f"invalid search query: {exc}") from exc
    _check_early_leak(
        stage="search",
        generated=[search_think, search_query],
        protected_phrases=protected,
        visible_context=f"{question}\n{vision_observation}",
    )

    try:
        answer_raw = teacher.generate(
            stage="answer",
            messages=_teacher_messages(
                stage="answer",
                sample_id=sample_id,
                question=question,
                answers=answers,
                image_path=image_path,
                image_index=config.image_index,
                vision_observation=vision_observation,
                text_observation=text_observation,
                vision_think=vision_think,
                vision_query=vision_query,
                search_think=search_think,
                search_query=search_query,
            ),
            response_schema=ANSWER_STAGE_SCHEMA,
        )
    except TeacherRequestError as exc:
        raise TrajectoryBuildError("answer", "teacher transport failure") from exc
    answer_think, final_answer = _strict_stage_output("answer", answer_raw)
    # Deliberately no semantic/correctness comparison against the gold answer.

    messages = [
        {"role": "user", "content": _student_prompt(record, question)},
        _assistant_tool_message(
            think=vision_think,
            name="vision_search",
            arguments={"image_index": config.image_index, "query": vision_query},
        ),
        {"role": "tool", "name": "vision_search", "content": vision_observation},
        _assistant_tool_message(think=search_think, name="search", arguments={"query": search_query}),
        {"role": "tool", "name": "search", "content": text_observation},
        {"role": "assistant", "content": f"<think>{answer_think}</think>\n<answer>{final_answer}</answer>"},
    ]
    return {
        "data_source": config.data_source,
        "messages": messages,
        "tools": canonical_tool_schemas_json(),
        "images": [{"image": image_path}],
        "sample_id": sample_id,
        "image_key": image_key,
        "category_key": str(_lookup(record, "category_key", "")),
        "question_type": question_type,
        "retrieval_resolvable": True,
        "extra_info": {
            "sample_id": sample_id,
            "image_key": image_key,
            "question_type": question_type,
            "source_data_source": record.get("data_source"),
            "oracle_vision_ids": [_document_id(doc) for doc in vision_documents],
            "oracle_text_ids": [_document_id(doc) for doc in text_documents],
        },
    }


def _eligible_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for raw_record in records:
        record = dict(_plain(raw_record))
        if is_multi_hop_record(record):
            excluded["two_hop"] += 1
            continue
        if _lookup(record, "retrieval_resolvable", False) is not True:
            excluded["retrieval_unresolvable"] += 1
            continue
        if not str(_lookup(record, "image_key", "")).strip():
            excluded["missing_image_key"] += 1
            continue
        eligible.append(record)
    return eligible, excluded


def _sample_targets(records: Sequence[Mapping[str, Any]], fraction: float) -> dict[str, int]:
    counts = Counter(str(_lookup(record, "question_type", "")) for record in records)
    return {
        question_type: min(count, max(1, int(math.ceil(count * fraction))))
        for question_type, count in sorted(counts.items())
    }


def _grouped_train_val_split(
    rows: Sequence[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows or validation_fraction <= 0:
        return list(rows), []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_key"])].append(row)
    if len(groups) < 2:
        return list(rows), []

    by_type = Counter(str(row["question_type"]) for row in rows)
    desired = {
        question_type: max(1, int(round(count * validation_fraction)))
        for question_type, count in by_type.items()
    }
    val_keys: set[str] = set()
    current: Counter[str] = Counter()
    candidates = sorted(groups, key=lambda key: _stable_key(seed, "sft_val", key))

    while any(current[key] < desired[key] for key in desired) and len(val_keys) < len(groups) - 1:
        best_key: str | None = None
        best_score: tuple[float, str] | None = None
        for image_key in candidates:
            if image_key in val_keys:
                continue
            contribution = Counter(str(row["question_type"]) for row in groups[image_key])
            gain = sum(min(contribution[key], max(0, desired[key] - current[key])) for key in desired)
            if gain <= 0:
                continue
            overshoot = sum(max(0, current[key] + contribution[key] - desired[key]) for key in desired)
            score = (gain - 0.01 * overshoot, _stable_key(seed, "sft_val_choice", image_key))
            if best_score is None or score > best_score:
                best_score = score
                best_key = image_key
        if best_key is None:
            break
        val_keys.add(best_key)
        current.update(str(row["question_type"]) for row in groups[best_key])

    train_rows = [row for row in rows if str(row["image_key"]) not in val_keys]
    val_rows = [row for row in rows if str(row["image_key"]) in val_keys]
    train_rows.sort(key=lambda row: _stable_key(seed, "sft_train_order", row["sample_id"]))
    val_rows.sort(key=lambda row: _stable_key(seed, "sft_val_order", row["sample_id"]))
    return train_rows, val_rows


def build_sft_records(
    train_records: Sequence[Mapping[str, Any]],
    vision_corpus: Sequence[Mapping[str, Any]],
    text_corpus: Sequence[Mapping[str, Any]],
    heldout_image_keys: set[str],
    teacher: TeacherClient,
    config: SFTBuilderConfig | None = None,
    observation_tokenizer: Any | None = None,
) -> SFTBuildResult:
    """Build SFT rows without performing file I/O.

    This is the primary test seam: fixtures can pass a fake teacher and tiny
    in-memory corpora without a model, retrieval server, or real EVQA data.
    """

    config = config or SFTBuilderConfig()
    vision_corpus = [dict(_plain(row)) for row in vision_corpus]
    text_corpus = [dict(_plain(row)) for row in text_corpus]
    leaked = sorted(
        {str(row.get("image_key", "")) for row in vision_corpus if str(row.get("image_key", "")) in heldout_image_keys}
    )
    if leaked:
        raise ValueError(f"vision corpus contains held-out query images: {leaked[:5]}")

    eligible, excluded = _eligible_records(train_records)
    targets = _sample_targets(eligible, config.sample_fraction)
    by_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in eligible:
        by_type[str(_lookup(record, "question_type", ""))][str(_lookup(record, "image_key", ""))].append(record)

    successes: list[dict[str, Any]] = []
    success_counts: Counter[str] = Counter()
    attempted_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    failures_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    failures_by_stage: Counter[str] = Counter()

    for question_type in sorted(by_type):
        image_groups = by_type[question_type]
        ordered_group_keys = sorted(
            image_groups,
            key=lambda key: _stable_key(config.seed, "sft_candidate_group", question_type, key),
        )
        for image_key in ordered_group_keys:
            if success_counts[question_type] >= targets[question_type]:
                break
            group_records = sorted(
                image_groups[image_key],
                key=lambda record: _stable_key(config.seed, "sft_candidate", _lookup(record, "sample_id", "")),
            )
            # Selecting an image group is atomic: attempt every row in it even
            # if an earlier row reaches the approximate per-type target.
            for record in group_records:
                attempted_counts[question_type] += 1
                try:
                    trajectory = _build_trajectory(
                        record,
                        vision_corpus,
                        text_corpus,
                        teacher,
                        config,
                        observation_tokenizer=observation_tokenizer,
                    )
                except TrajectoryBuildError as exc:
                    key = f"{exc.stage}:{exc.reason}"
                    failure_reasons[key] += 1
                    failures_by_type[question_type][key] += 1
                    failures_by_stage[exc.stage] += 1
                    continue
                successes.append(trajectory)
                success_counts[question_type] += 1

    train_rows, val_rows = _grouped_train_val_split(successes, config.validation_fraction, config.seed)
    train_image_keys = {row["image_key"] for row in train_rows}
    val_image_keys = {row["image_key"] for row in val_rows}
    if train_image_keys & val_image_keys:
        raise AssertionError("SFT train/validation image groups overlap")

    report = {
        "schema_version": 1,
        "config": asdict(config),
        "input": {
            "train_rows": len(train_records),
            "vision_corpus_rows": len(vision_corpus),
            "text_corpus_rows": len(text_corpus),
            "heldout_image_keys": len(heldout_image_keys),
        },
        "eligibility": {
            "eligible": len(eligible),
            "excluded": dict(sorted(excluded.items())),
            "eligible_by_question_type": dict(
                sorted(Counter(str(_lookup(row, "question_type", "")) for row in eligible).items())
            ),
        },
        "sampling": {
            "targets_by_question_type": targets,
            "attempted_by_question_type": dict(sorted(attempted_counts.items())),
            "successful_by_question_type": dict(sorted(success_counts.items())),
            "target_shortfall_by_question_type": {
                key: max(0, targets[key] - success_counts[key]) for key in sorted(targets)
            },
        },
        "failures": {
            "total": sum(failure_reasons.values()),
            "by_reason": dict(sorted(failure_reasons.items())),
            "by_stage": dict(sorted(failures_by_stage.items())),
            "by_question_type": {
                key: dict(sorted(counter.items())) for key, counter in sorted(failures_by_type.items())
            },
        },
        "output": {
            "successful": len(successes),
            "sft_train": len(train_rows),
            "sft_val": len(val_rows),
            "sft_train_image_groups": len(train_image_keys),
            "sft_val_image_groups": len(val_image_keys),
            "image_group_overlap": 0,
        },
        "observation_truncation": {
            "max_tokens_including_native_wrapper": config.max_tool_response_tokens,
            "mode": "tokenizer_exact" if observation_tokenizer is not None else "conservative_utf8",
            "fallback_wrapper_token_reserve": config.fallback_wrapper_token_reserve,
        },
    }
    return SFTBuildResult(train_rows=train_rows, val_rows=val_rows, report=report)


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _extract_heldout_image_keys(manifest: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()

    def visit(value: Any, context: str = "") -> None:
        if isinstance(value, Mapping):
            image_key = value.get("image_key")
            if image_key and any(token in context for token in ("heldout", "query_image")):
                result.add(str(image_key))
            for key, item in value.items():
                key_context = f"{context}.{str(key).casefold()}"
                normalized_key = str(key).casefold()
                is_key_list = normalized_key in {"heldout_image_keys", "query_image_keys"} or (
                    normalized_key == "image_keys" and any(token in key_context for token in ("heldout", "query_image"))
                )
                if is_key_list:
                    for candidate in _as_list(item):
                        if isinstance(candidate, Mapping) and candidate.get("image_key"):
                            result.add(str(candidate["image_key"]))
                        elif candidate:
                            result.add(str(candidate))
                visit(item, key_context)
        elif isinstance(value, list):
            for item in value:
                visit(item, context)

    visit(manifest)
    return result


def _load_heldout_image_keys(manifest: Mapping[str, Any], manifest_path: Path) -> set[str]:
    """Load embedded keys and an optional compact heldout-manifest reference."""

    result = _extract_heldout_image_keys(manifest)
    heldout = manifest.get("heldout") if isinstance(manifest, Mapping) else None
    referenced_path = heldout.get("manifest_path") if isinstance(heldout, Mapping) else None
    if referenced_path and not result:
        raw_path = Path(str(referenced_path)).expanduser()
        candidates = [raw_path] if raw_path.is_absolute() else [raw_path, manifest_path.parent / raw_path]
        external_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
        if not external_path.is_file():
            raise FileNotFoundError(f"referenced heldout manifest does not exist: {external_path}")
        external_value = _load_json(external_path)
        if not isinstance(external_value, Mapping):
            raise ValueError(f"heldout manifest is not a JSON object: {external_path}")
        # Seed the traversal context so a compact root-level ``image_keys``
        # list is recognized without copying it into build_manifest.json.
        wrapped = {"heldout": external_value}
        result.update(_extract_heldout_image_keys(wrapped))
    return result


def _stage_parquet(rows: Sequence[Mapping[str, Any]], final_path: Path) -> Path:
    import pandas as pd

    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        # Preserve a loadable physical schema even when no teacher trajectory
        # succeeded, or when fewer than two image groups make a zero-overlap
        # validation split impossible. A zero-column Parquet file cannot be
        # opened by MultiTurnSFTDataset because it has no ``messages`` field.
        frame = pd.DataFrame(list(rows), columns=SFT_PARQUET_COLUMNS)
        frame.to_parquet(temp_path, index=False)
        reloaded = pd.read_parquet(temp_path)
        if len(reloaded) != len(rows):
            raise ValueError(f"Parquet round-trip row count mismatch for {final_path}")
        for _, physical_row in reloaded.iterrows():
            tools = physical_row.get("tools")
            decoded_tools = json.loads(tools) if isinstance(tools, str) else tools
            if not isinstance(_plain(decoded_tools), list):
                raise ValueError("SFT tools column did not round-trip as a JSON list")
            messages = _plain(physical_row.get("messages")) or []
            for message in messages:
                for tool_call in (message or {}).get("tool_calls") or []:
                    arguments = ((tool_call or {}).get("function") or {}).get("arguments")
                    if arguments is not None:
                        if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                            raise ValueError("SFT function.arguments must round-trip as a JSON object string")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _stage_json(value: Mapping[str, Any], final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return Path(temp_name)


def build_sft_files(
    *,
    train_parquet: str | Path,
    vision_corpus_path: str | Path,
    text_corpus_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    teacher: TeacherClient,
    config: SFTBuilderConfig | None = None,
    observation_tokenizer: Any | None = None,
) -> SFTBuildResult:
    import pandas as pd

    train_path = Path(train_parquet)
    vision_path = Path(vision_corpus_path)
    text_path = Path(text_corpus_path)
    manifest_file = Path(manifest_path)
    for path in (train_path, vision_path, text_path, manifest_file):
        if not path.is_file():
            raise FileNotFoundError(f"required SFT input does not exist: {path}")

    train_records = [_plain(row) for row in pd.read_parquet(train_path).to_dict(orient="records")]
    vision_corpus = _load_jsonl(vision_path)
    text_corpus = _load_jsonl(text_path)
    manifest = _load_json(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"build manifest is not a JSON object: {manifest_file}")
    heldout_keys = _load_heldout_image_keys(manifest, manifest_file)
    if not heldout_keys:
        raise ValueError("manifest contains no held-out/query image keys; refusing to build SFT data")

    result = build_sft_records(
        train_records,
        vision_corpus,
        text_corpus,
        heldout_keys,
        teacher,
        config=config,
        observation_tokenizer=observation_tokenizer,
    )
    output = Path(output_dir)
    train_final = output / "sft_train.parquet"
    val_final = output / "sft_val.parquet"
    report_final = output / "sft_build_report.json"
    staged: list[tuple[Path, Path]] = []
    try:
        staged.append((_stage_parquet(result.train_rows, train_final), train_final))
        staged.append((_stage_parquet(result.val_rows, val_final), val_final))
        report = dict(result.report)
        report["paths"] = {
            "train_parquet": str(train_path.resolve()),
            "vision_corpus": str(vision_path.resolve()),
            "text_corpus": str(text_path.resolve()),
            "manifest": str(manifest_file.resolve()),
            "sft_train": str(train_final.resolve()),
            "sft_val": str(val_final.resolve()),
        }
        staged.append((_stage_json(report, report_final), report_final))
        for temporary, final in staged:
            os.replace(temporary, final)
        result.report = report
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified DualSearch cold-start SFT Parquet files.")
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--vision-corpus", required=True)
    parser.add_argument("--text-corpus", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle-top-k", type=int, default=3)
    parser.add_argument(
        "--observation-tokenizer",
        default=None,
        help="Optional local Qwen tokenizer path for exact wrapper-inclusive observation truncation.",
    )
    parser.add_argument("--max-tool-response-tokens", type=int, default=DEFAULT_TOOL_RESPONSE_TOKENS)
    parser.add_argument("--teacher-timeout", type=float, default=120.0)
    parser.add_argument("--teacher-max-retries", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    observation_tokenizer = None
    if args.observation_tokenizer:
        tokenizer_path = Path(args.observation_tokenizer).expanduser()
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"observation tokenizer must be local: {tokenizer_path}")
        from transformers import AutoTokenizer

        observation_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    teacher = VLLMTeacherClient(
        TeacherConfig(
            base_url=args.teacher_base_url,
            model=args.teacher_model,
            api_key=os.getenv(args.teacher_api_key_env),
            timeout_seconds=args.teacher_timeout,
            max_retries=args.teacher_max_retries,
        )
    )
    result = build_sft_files(
        train_parquet=args.train_parquet,
        vision_corpus_path=args.vision_corpus,
        text_corpus_path=args.text_corpus,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        teacher=teacher,
        config=SFTBuilderConfig(
            sample_fraction=args.sample_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            oracle_top_k=args.oracle_top_k,
            max_tool_response_tokens=args.max_tool_response_tokens,
        ),
        observation_tokenizer=observation_tokenizer,
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
