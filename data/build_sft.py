"""Build verified, closed-loop DualSearch cold-start SFT trajectories.

The builder consumes already-materialized RL/corpus/index artifacts.  For each
eligible single-image example it asks a multimodal teacher for one small JSON
decision at a time, executes the generated query against the same HTTP
retrieval services used by the RL agent, and exposes only the real Top-K
observation to the next stage:

``vision_search -> search -> gold-conditioned answer``.

Only successful, causally consistent trajectories are published.  The builder
does not download data, build indexes, or start model services.
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
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_search.protocol import (
    ProtocolError,
    canonical_json,
    canonical_tool_schemas_json,
    format_text_results,
    format_vision_results,
    sanitize_tool_response,
    validate_tool_call_payload,
)
from dual_search.search.fingerprints import (
    artifact_fingerprint,
    corpus_fingerprint,
    load_and_validate_index_meta,
    model_fingerprint,
    sha256_file,
    stable_digest,
)


RESERVED_TAG_RE = re.compile(
    r"</?(?:think|tool_call|tool_response|answer|search|vision_search)(?:\s[^>]*)?>|<\|im_(?:start|end)\|>",
    re.IGNORECASE,
)
MEDIA_PLACEHOLDER_RE = re.compile(r"<(?:image|video)>", re.IGNORECASE)
WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MULTI_HOP_MARKERS = {"2", "2hop", "twohop", "multihop", "multiplehop"}
DEFAULT_TOOL_RESPONSE_TOKENS = 500
DEFAULT_TOOL_WRAPPER_TOKEN_RESERVE = 32
SFT_SCHEMA_VERSION = 3
RL_INPUT_SCHEMA_VERSION = 2
ALLOWED_COARSE_TAXA = ("Plants", "Insects", "Birds")
SFT_PARQUET_COLUMNS = (
    "schema_version",
    "data_source",
    "messages",
    "tools",
    "images",
    "sample_id",
    "parent_sample_id",
    "source_image_index",
    "image_key",
    "category_key",
    "coarse_taxon",
    "question_type",
    "retrieval_resolvable",
    "extra_info",
)

VISION_SYSTEM_PROMPT = """Generate a visual retrieval query for the given image and question.

Write:
- "think": a concise explanation of which visible characteristics are useful for retrieval.
- "query": a concise English retrieval query grounded in the visible image.

Do not answer the question.
Do not guess an entity or species name unless it already appears in the question.
Do not use hidden labels or external knowledge.
Focus on discriminative visual properties rather than writing a generic query such as "identify this image".

Return exactly one JSON object:
{"think":"...","query":"..."}

Do not output Markdown, XML tags, or any other text."""

SEARCH_SYSTEM_PROMPT = """Generate a text knowledge-base query using the original question and the real visual retrieval result.

Write:
- "think": a concise explanation of what textual information is needed.
- "query": a standalone English search query targeting that information.

Use entity names only when they appear in the question or visual retrieval result.
Target the property requested by the question, such as habitat, diet, distribution, behavior, taxonomy, or morphology.
Do not answer the question.
Do not use a reference answer or hidden evidence label.

Return exactly one JSON object:
{"think":"...","query":"..."}

Do not output Markdown, XML tags, or any other text."""

ANSWER_SYSTEM_PROMPT = """Generate a concise final reasoning summary and answer using the provided question, real retrieval results, and reference answer.

Write:
- "think": a short explanation connecting the visual identification, textual evidence, and answer.
- "answer": the final answer.

Every factual statement in "think" must be supported by the question or retrieval results.
Do not introduce unsupported facts.
Do not mention that a reference answer was provided.
Do not describe the data-generation process.

Return exactly one JSON object:
{"think":"...","answer":"..."}

Do not output Markdown, XML tags, or any other text."""

VISION_USER_REQUEST = (
    "Generate the visual reasoning and retrieval query for the attached image."
)
SEARCH_USER_REQUEST = (
    "Based on the previous visual retrieval trajectory, generate the text "
    "reasoning and a standalone text search query needed to answer the question."
)
ANSWER_USER_REQUEST = (
    "Generate a concise reasoning summary grounded in the previous trajectory "
    "and the reference answer.\n"
    "Also generate a final answer. The generated answer will be canonicalized "
    "separately."
)
NO_PREVIOUS_TOOL_INTERACTION = "No previous tool interaction."
TEACHER_USER_TEXT_TEMPLATE = (
    "[QUESTION]\n{question}\n\n"
    "[CONTENT]\n{content}\n\n"
    "[USER REQUEST]\n{request}"
)
TEACHER_TOOL_HISTORY_TEMPLATE = (
    "<think>{think}</think>\n"
    "<tool_call>\n{tool_call}</tool_call>\n"
    "<tool_response>\n{observation}\n</tool_response>"
)
REFERENCE_ANSWER_TEMPLATE = "Reference answer:\n{answer}"

QUERY_STAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "think": {"type": "string", "minLength": 1},
        "query": {"type": "string", "minLength": 1},
    },
    "required": ["think", "query"],
    "additionalProperties": False,
}

VISION_STAGE_SCHEMA = QUERY_STAGE_SCHEMA
SEARCH_STAGE_SCHEMA = QUERY_STAGE_SCHEMA
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


class RetrieverClient(Protocol):
    """Small testable interface for runtime-equivalent retrieval."""

    def vision_search(
        self,
        *,
        query: str,
        image: str,
        image_index: int,
        top_k: int,
    ) -> list[dict[str, Any]]: ...

    def text_search(self, *, query: str, top_k: int) -> list[dict[str, Any]]: ...


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
    sample_fraction: float = 0.10
    validation_fraction: float = 0.10
    seed: int = 42
    retrieval_top_k: int = 3
    data_source: str = "dual_search_sft"
    max_tool_response_tokens: int = DEFAULT_TOOL_RESPONSE_TOKENS
    fallback_wrapper_token_reserve: int = DEFAULT_TOOL_WRAPPER_TOKEN_RESERVE

    def __post_init__(self) -> None:
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        if self.data_source != "dual_search_sft":
            raise ValueError("data_source is fixed to 'dual_search_sft'")
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


class RetrievalRequestError(RuntimeError):
    """Retriever transport or response-contract failure."""


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


@dataclass(frozen=True)
class HTTPRetrieverConfig:
    vision_url: str
    text_url: str
    timeout_seconds: float = 120.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0


class HTTPRetrieverClient:
    """Strict client for the same retrieval endpoints used by the agent loop."""

    _TRANSIENT_STATUS = VLLMTeacherClient._TRANSIENT_STATUS

    def __init__(self, config: HTTPRetrieverConfig, session: Any | None = None):
        self.config = config
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def _post(
        self,
        *,
        endpoint: str,
        payload: Mapping[str, Any],
        label: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.session.post(
                    endpoint,
                    json=dict(payload),
                    timeout=self.config.timeout_seconds,
                )
            except Exception as exc:
                if attempt + 1 >= attempts:
                    raise RetrievalRequestError(f"{label} request failed: {exc}") from exc
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                continue

            if response.status_code in self._TRANSIENT_STATUS:
                if attempt + 1 >= attempts:
                    raise RetrievalRequestError(
                        f"{label} returned transient HTTP {response.status_code} "
                        f"after {attempts} attempts"
                    )
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
                continue
            try:
                response.raise_for_status()
            except Exception as exc:
                raise RetrievalRequestError(
                    f"{label} returned HTTP {response.status_code}"
                ) from exc

            try:
                body = response.json()
                batches = body["result"]
            except (KeyError, TypeError, ValueError) as exc:
                raise RetrievalRequestError(
                    f"{label} response must contain a result list"
                ) from exc
            if (
                not isinstance(batches, list)
                or len(batches) != 1
                or not isinstance(batches[0], list)
            ):
                raise RetrievalRequestError(
                    f"{label} response must contain exactly one result batch"
                )
            raw_results = batches[0]
            if len(raw_results) != top_k:
                raise RetrievalRequestError(
                    f"{label} returned {len(raw_results)} results; expected Top-{top_k}"
                )

            normalized: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for rank, raw_item in enumerate(raw_results, start=1):
                if not isinstance(raw_item, Mapping):
                    raise RetrievalRequestError(
                        f"{label} result at rank {rank} is not an object"
                    )
                if set(raw_item) != {"document", "score"}:
                    raise RetrievalRequestError(
                        f"{label} result at rank {rank} must contain exactly "
                        "document and score"
                    )
                document = raw_item.get("document")
                score = raw_item.get("score")
                if not isinstance(document, Mapping):
                    raise RetrievalRequestError(
                        f"{label} result at rank {rank} has no document object"
                    )
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise RetrievalRequestError(
                        f"{label} result at rank {rank} has an invalid score"
                    )
                document = dict(_plain(document))
                document_id = _document_id(document)
                if not document_id:
                    raise RetrievalRequestError(
                        f"{label} result at rank {rank} has no stable document id"
                    )
                if document_id in seen_ids:
                    raise RetrievalRequestError(
                        f"{label} returned duplicate document id {document_id!r}"
                    )
                seen_ids.add(document_id)
                normalized.append({"document": document, "score": float(score)})
            return normalized

        raise AssertionError("unreachable")

    def vision_search(
        self,
        *,
        query: str,
        image: str,
        image_index: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        payload = {
            "queries": [
                {
                    "query": query,
                    "image_index": image_index,
                    "image": _image_as_data_url(image),
                }
            ],
            "topk": top_k,
            "return_scores": True,
        }
        return self._post(
            endpoint=self.config.vision_url,
            payload=payload,
            label="vision retriever",
            top_k=top_k,
        )

    def text_search(self, *, query: str, top_k: int) -> list[dict[str, Any]]:
        payload = {
            "queries": [query],
            "topk": top_k,
            "return_scores": True,
        }
        return self._post(
            endpoint=self.config.text_url,
            payload=payload,
            label="text retriever",
            top_k=top_k,
        )


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

    This predicate only controls SFT eligibility; accepted records are later
    sampled independently within each coarse taxon.
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
    # SFT children always contain exactly one image. Do not copy the parent
    # RL prompt because a multi-image parent has numbered placeholders whose
    # image_index semantics no longer apply after expansion.
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
    question: str,
    image_path: str,
    vision_observation: str | None = None,
    text_observation: str | None = None,
    vision_think: str | None = None,
    vision_query: str | None = None,
    search_think: str | None = None,
    search_query: str | None = None,
    canonical_gold_answer: str | None = None,
) -> list[dict[str, Any]]:
    if stage == "vision":
        system_prompt = VISION_SYSTEM_PROMPT
        history = NO_PREVIOUS_TOOL_INTERACTION
        request = VISION_USER_REQUEST
    elif stage == "search":
        if not all((vision_think, vision_query, vision_observation)):
            raise ValueError("search stage requires complete visual tool history")
        system_prompt = SEARCH_SYSTEM_PROMPT
        history = _teacher_tool_history(
            think=str(vision_think),
            name="vision_search",
            arguments={"image_index": 1, "query": str(vision_query)},
            observation=str(vision_observation),
        )
        request = SEARCH_USER_REQUEST
    elif stage == "answer":
        if not all(
            (
                vision_think,
                vision_query,
                vision_observation,
                search_think,
                search_query,
                text_observation,
                canonical_gold_answer,
            )
        ):
            raise ValueError("answer stage requires complete tool history and reference answer")
        system_prompt = ANSWER_SYSTEM_PROMPT
        history = "\n\n".join(
            [
                _teacher_tool_history(
                    think=str(vision_think),
                    name="vision_search",
                    arguments={"image_index": 1, "query": str(vision_query)},
                    observation=str(vision_observation),
                ),
                _teacher_tool_history(
                    think=str(search_think),
                    name="search",
                    arguments={"query": str(search_query)},
                    observation=str(text_observation),
                ),
                REFERENCE_ANSWER_TEMPLATE.format(answer=canonical_gold_answer),
            ]
        )
        request = ANSWER_USER_REQUEST
    else:
        raise ValueError(f"unknown teacher stage: {stage}")

    prompt_text = TEACHER_USER_TEXT_TEMPLATE.format(
        question=question,
        content=history,
        request=request,
    )
    # Each stage is an independent teacher request, so attach the actual image
    # every time. The flattened CONTENT is synthesis-only and is never written
    # into the student's structured SFT conversation.
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_as_data_url(image_path)}},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def _teacher_tool_history(
    *,
    think: str,
    name: str,
    arguments: Mapping[str, Any],
    observation: str,
) -> str:
    payload = canonical_json({"name": name, "arguments": dict(arguments)})
    return TEACHER_TOOL_HISTORY_TEMPLATE.format(
        think=think,
        tool_call=payload,
        observation=observation,
    )


def _strict_stage_output(stage: str, value: Mapping[str, Any]) -> tuple[str, str]:
    expected = {
        "vision": ({"think", "query"}, "query"),
        "search": ({"think", "query"}, "query"),
        "answer": ({"think", "answer"}, "answer"),
    }
    keys, payload_key = expected[stage]
    if not isinstance(value, Mapping) or set(value) != keys:
        raise TrajectoryBuildError(stage, f"teacher JSON keys must be exactly {sorted(keys)}")
    think = value.get("think")
    output = value.get(payload_key)
    if not isinstance(think, str) or not think.strip():
        raise TrajectoryBuildError(stage, "think must be a non-empty string")
    if not isinstance(output, str) or not output.strip():
        raise TrajectoryBuildError(stage, f"{payload_key} must be a non-empty string")
    if (
        RESERVED_TAG_RE.search(think)
        or RESERVED_TAG_RE.search(output)
        or MEDIA_PLACEHOLDER_RE.search(think)
        or MEDIA_PLACEHOLDER_RE.search(output)
    ):
        raise TrajectoryBuildError(stage, "teacher output contains a reserved control tag")
    return think.strip(), output.strip()


def _sanitize_media_placeholders(content: str) -> str:
    """Prevent retrieved prose from becoming multimodal loader controls."""

    return MEDIA_PLACEHOLDER_RE.sub(
        lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        content,
    )


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


def _corpus_table(
    corpus: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for row_number, raw_document in enumerate(corpus, start=1):
        document = dict(_plain(raw_document))
        document_id = _document_id(document)
        if not document_id:
            raise ValueError(f"{label} corpus row {row_number} has no stable id")
        if document_id in table:
            raise ValueError(f"{label} corpus contains duplicate id {document_id!r}")
        table[document_id] = document
    return table


def _validate_retrieved_documents(
    results: Sequence[Mapping[str, Any]],
    *,
    corpus_by_id: Mapping[str, Mapping[str, Any]],
    label: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    if len(results) != expected_count:
        raise TrajectoryBuildError(
            label,
            f"retriever returned {len(results)} results; expected Top-{expected_count}",
        )
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for rank, raw_result in enumerate(results, start=1):
        result = dict(_plain(raw_result))
        document = result.get("document")
        score = result.get("score")
        if not isinstance(document, Mapping):
            raise TrajectoryBuildError(label, f"retrieval result {rank} has no document")
        document = dict(document)
        document_id = _document_id(document)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise TrajectoryBuildError(
                label,
                f"retrieval result {document_id or rank!r} has an invalid score",
            )
        expected = corpus_by_id.get(document_id)
        if expected is None:
            raise TrajectoryBuildError(
                label,
                f"retrieval result {document_id!r} is absent from the published corpus",
            )
        if document_id in seen_ids:
            raise TrajectoryBuildError(
                label,
                f"retriever returned duplicate document id {document_id!r}",
            )
        seen_ids.add(document_id)
        if stable_digest(document) != stable_digest(dict(expected)):
            raise TrajectoryBuildError(
                label,
                f"retrieval result {document_id!r} differs from the published corpus",
            )
        validated.append({"document": document, "score": float(score)})
    return validated


def _target_evidence_pairs(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    raw_pairs = [
        pair
        for pair in _as_list(_lookup(record, "wiki_pairs", []))
        if isinstance(pair, Mapping)
    ]
    urls = [
        _normalize_url(pair.get("normalized_url") or pair.get("url"))
        for pair in raw_pairs
    ]
    urls = [url for url in urls if url]
    if not urls:
        urls = [
            _normalize_url(value)
            for value in _as_list(_lookup(record, "wikipedia_url", []))
            if _normalize_url(value)
        ]
    section_ids = [
        str(value).strip()
        for value in _as_list(_lookup(record, "evidence_section_id", []))
        if str(value).strip()
    ]
    if len(urls) != 1 or not section_ids:
        raise TrajectoryBuildError(
            "preflight",
            "single-hop SFT requires one Wikipedia URL and at least one evidence section id",
        )
    return {(urls[0], section_id) for section_id in section_ids}


def _document_evidence_pair(document: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _normalize_url(document.get("url") or document.get("source_url")),
        str(document.get("section_id", "")).strip(),
    )


def _visible_vision_marker(document: Mapping[str, Any], rank: int) -> str:
    contents = str(document.get("contents", ""))
    title = contents.split("\n", 1)[0].strip().strip('"')
    if not title:
        title = str(document.get("title", "")).strip()
    # Keep this marker aligned with ``format_vision_results`` in
    # ``dual_search.protocol``.  Checking the formatted marker, rather than a
    # hidden document id, proves that the positive result remains visible to
    # the model after the runtime-equivalent observation truncation.
    return f"Caption {rank}(Title: {title})"


def _visible_text_marker(document: Mapping[str, Any], rank: int) -> str:
    contents = str(document.get("contents", ""))
    title = contents.split("\n", 1)[0].strip().strip('"')
    if not title:
        title = str(document.get("title", "")).strip()
    return f"Doc {rank}(Title: {title})"


def _visible_text_segments(
    results: Sequence[Mapping[str, Any]],
    observation: str,
) -> dict[int, str]:
    """Return normalized visible text belonging to each non-truncated result."""

    normalized_observation = _normalized_phrase(observation)
    markers = {
        rank: _normalized_phrase(_visible_text_marker(result["document"], rank))
        for rank, result in enumerate(results, start=1)
    }
    positions = {
        rank: normalized_observation.find(marker)
        for rank, marker in markers.items()
    }
    segments: dict[int, str] = {}
    for rank, start in positions.items():
        if start < 0:
            continue
        later_starts = [
            position
            for later_rank, position in positions.items()
            if later_rank > rank and position > start
        ]
        end = min(later_starts) if later_starts else len(normalized_observation)
        segments[rank] = normalized_observation[start:end]
    return segments


def _evidence_phrases(record: Mapping[str, Any]) -> list[str]:
    return [
        phrase
        for phrase in (
            str(value).strip()
            for value in _as_list(_lookup(record, "evidence", []))
        )
        if phrase
    ]


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
    vision_corpus_by_id: Mapping[str, Mapping[str, Any]],
    text_corpus_by_id: Mapping[str, Mapping[str, Any]],
    teacher: TeacherClient,
    retriever: RetrieverClient,
    config: SFTBuilderConfig,
    observation_tokenizer: Any | None = None,
) -> dict[str, Any]:
    sample_id = str(_lookup(record, "sample_id", "")).strip()
    image_key = str(_lookup(record, "image_key", "")).strip()
    question = str(_lookup(record, "question", "")).strip()
    question_type = str(_lookup(record, "question_type", "")).strip()
    coarse_taxon = str(_lookup(record, "coarse_taxon", "")).strip()
    image_path = _first_image_path(record)
    answers = _gold_answers(record)
    if (
        not all((sample_id, image_key, question, question_type, coarse_taxon, image_path))
        or not answers
    ):
        raise TrajectoryBuildError("preflight", "sample is missing required SFT fields")
    canonical_gold_answer = answers[0]
    if RESERVED_TAG_RE.search(question) or MEDIA_PLACEHOLDER_RE.search(question):
        raise TrajectoryBuildError("preflight", "question contains a reserved control tag")
    if RESERVED_TAG_RE.search(canonical_gold_answer) or MEDIA_PLACEHOLDER_RE.search(
        canonical_gold_answer
    ):
        raise TrajectoryBuildError("preflight", "canonical answer contains a reserved control tag")
    category_key = str(_lookup(record, "category_key", "")).strip()
    target_evidence_pairs = _target_evidence_pairs(record)
    evidence_phrases = _evidence_phrases(record)
    if not evidence_phrases:
        raise TrajectoryBuildError("preflight", "sample has no evidence text for visibility validation")

    try:
        vision_raw = teacher.generate(
            stage="vision",
            messages=_teacher_messages(
                stage="vision",
                question=question,
                image_path=image_path,
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
                "arguments": {"image_index": 1, "query": vision_query},
            },
            image_count=1,
        ).arguments["query"]
    except ProtocolError as exc:
        raise TrajectoryBuildError("vision", f"invalid vision_search query: {exc}") from exc
    _check_early_leak(
        stage="vision",
        generated=[vision_think, vision_query],
        protected_phrases=[
            *answers,
            *_as_list(_lookup(record, "wikipedia_title", [])),
        ],
        visible_context=question,
    )

    try:
        vision_results = retriever.vision_search(
            query=vision_query,
            image=image_path,
            image_index=1,
            top_k=config.retrieval_top_k,
        )
    except RetrievalRequestError as exc:
        raise TrajectoryBuildError("vision_retrieval", str(exc)) from exc
    vision_results = _validate_retrieved_documents(
        vision_results,
        corpus_by_id=vision_corpus_by_id,
        label="vision_retrieval",
        expected_count=config.retrieval_top_k,
    )
    vision_positive_ranks = [
        rank
        for rank, result in enumerate(vision_results, start=1)
        if str(result["document"].get("category_key", "")).strip() == category_key
        and str(result["document"].get("image_key", "")).strip() != image_key
    ]
    if not vision_positive_ranks:
        raise TrajectoryBuildError("vision_hit", "real visual Top-K missed category_key")
    positive_vision_document = vision_results[vision_positive_ranks[0] - 1]["document"]
    protected = [*answers, *_entity_aliases(record, positive_vision_document)]
    _check_early_leak(
        stage="vision",
        generated=[vision_think, vision_query],
        protected_phrases=protected,
        visible_context=question,
    )
    vision_observation = truncate_tool_observation(
        _sanitize_media_placeholders(
            (
                "image_index=1:\n"
                + format_vision_results(vision_results)
            ).rstrip()
        ),
        tokenizer=observation_tokenizer,
        max_tokens=config.max_tool_response_tokens,
        fallback_wrapper_token_reserve=config.fallback_wrapper_token_reserve,
    )
    visible_vision_ranks = [
        rank
        for rank in vision_positive_ranks
        if _normalized_phrase(
            _visible_vision_marker(vision_results[rank - 1]["document"], rank)
        )
        in _normalized_phrase(vision_observation)
    ]
    if not visible_vision_ranks:
        raise TrajectoryBuildError(
            "vision_observation",
            "positive visual result was removed by observation truncation",
        )

    try:
        search_raw = teacher.generate(
            stage="search",
            messages=_teacher_messages(
                stage="search",
                question=question,
                image_path=image_path,
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
        text_results = retriever.text_search(
            query=search_query,
            top_k=config.retrieval_top_k,
        )
    except RetrievalRequestError as exc:
        raise TrajectoryBuildError("text_retrieval", str(exc)) from exc
    text_results = _validate_retrieved_documents(
        text_results,
        corpus_by_id=text_corpus_by_id,
        label="text_retrieval",
        expected_count=config.retrieval_top_k,
    )
    returned_pairs: dict[tuple[str, str], int] = {}
    for rank, result in enumerate(text_results, start=1):
        returned_pairs.setdefault(_document_evidence_pair(result["document"]), rank)
    missing_pairs = target_evidence_pairs.difference(returned_pairs)
    if missing_pairs:
        raise TrajectoryBuildError(
            "text_hit",
            f"real text Top-K missed evidence sections {sorted(missing_pairs)!r}",
        )
    text_observation = truncate_tool_observation(
        _sanitize_media_placeholders(format_text_results(text_results)),
        tokenizer=observation_tokenizer,
        max_tokens=config.max_tool_response_tokens,
        fallback_wrapper_token_reserve=config.fallback_wrapper_token_reserve,
    )
    target_ranks = {
        returned_pairs[target_pair]
        for target_pair in target_evidence_pairs
    }
    visible_text_segments = _visible_text_segments(text_results, text_observation)
    invisible_target_ranks = sorted(target_ranks.difference(visible_text_segments))
    if invisible_target_ranks:
        raise TrajectoryBuildError(
            "text_observation",
            f"target evidence documents were removed by observation truncation: "
            f"ranks {invisible_target_ranks}",
        )
    evidence_source_ranks: dict[str, list[int]] = {}
    for phrase in evidence_phrases:
        normalized_phrase = _normalized_phrase(phrase)
        ranks = [
            rank
            for rank in sorted(target_ranks)
            if normalized_phrase
            in _normalized_phrase(text_results[rank - 1]["document"].get("contents", ""))
        ]
        if not ranks:
            raise TrajectoryBuildError(
                "text_hit",
                "annotated evidence is absent from the retrieved target section",
            )
        evidence_source_ranks[phrase] = ranks
    invisible_evidence = [
        phrase
        for phrase in evidence_phrases
        if not any(
            _normalized_phrase(phrase) in visible_text_segments[rank]
            for rank in evidence_source_ranks[phrase]
        )
    ]
    if invisible_evidence:
        raise TrajectoryBuildError(
            "text_observation",
            "target evidence was removed by observation truncation",
        )

    try:
        answer_raw = teacher.generate(
            stage="answer",
            messages=_teacher_messages(
                stage="answer",
                question=question,
                image_path=image_path,
                vision_observation=vision_observation,
                text_observation=text_observation,
                vision_think=vision_think,
                vision_query=vision_query,
                search_think=search_think,
                search_query=search_query,
                canonical_gold_answer=canonical_gold_answer,
            ),
            response_schema=ANSWER_STAGE_SCHEMA,
        )
    except TeacherRequestError as exc:
        raise TrajectoryBuildError("answer", "teacher transport failure") from exc
    answer_think, _teacher_answer = _strict_stage_output("answer", answer_raw)
    # Deliberately ignore the Teacher answer. This stage is gold-conditioned,
    # and the published answer is always the canonical official target.

    messages = [
        {"role": "user", "content": _student_prompt(record, question)},
        _assistant_tool_message(
            think=vision_think,
            name="vision_search",
            arguments={"image_index": 1, "query": vision_query},
        ),
        {"role": "tool", "name": "vision_search", "content": vision_observation},
        _assistant_tool_message(think=search_think, name="search", arguments={"query": search_query}),
        {"role": "tool", "name": "search", "content": text_observation},
        {
            "role": "assistant",
            "content": (
                f"<think>{answer_think}</think>\n"
                f"<answer>{canonical_gold_answer}</answer>"
            ),
        },
    ]
    vision_ids = [_document_id(result["document"]) for result in vision_results]
    text_ids = [_document_id(result["document"]) for result in text_results]
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "data_source": config.data_source,
        "messages": messages,
        "tools": canonical_tool_schemas_json(),
        "images": [{"image": image_path}],
        "sample_id": sample_id,
        "parent_sample_id": str(_lookup(record, "parent_sample_id", "")).strip(),
        "source_image_index": int(_lookup(record, "source_image_index", 0)),
        "image_key": image_key,
        "category_key": category_key,
        "coarse_taxon": coarse_taxon,
        "question_type": question_type,
        "retrieval_resolvable": True,
        "extra_info": {
            "sample_id": sample_id,
            "parent_sample_id": str(_lookup(record, "parent_sample_id", "")).strip(),
            "source_image_index": int(_lookup(record, "source_image_index", 0)),
            "image_key": image_key,
            "category_key": category_key,
            "coarse_taxon": coarse_taxon,
            "question_type": question_type,
            "source_data_source": record.get("data_source"),
            "observation_source": "real_retriever",
            "retrieval_top_k": config.retrieval_top_k,
            "retrieved_vision_ids": vision_ids,
            "retrieved_vision_scores": [result["score"] for result in vision_results],
            "vision_positive_ranks": vision_positive_ranks,
            "vision_visible_positive_ranks": visible_vision_ranks,
            "retrieved_text_ids": text_ids,
            "retrieved_text_scores": [result["score"] for result in text_results],
            # A fixed-shape list avoids turning every URL into a distinct
            # sparse Arrow struct field across the dataset.
            "text_evidence_ranks": [
                {
                    "url": url,
                    "section_id": section_id,
                    "rank": returned_pairs[(url, section_id)],
                }
                for url, section_id in sorted(target_evidence_pairs)
            ],
            "final_generation_mode": "gold_conditioned",
            "canonical_gold_answer": canonical_gold_answer,
        },
    }


_QUERY_IMAGE_KEYS = {
    "image_index",
    "dataset_image_id",
    "image_key",
    "image",
    "source_file_name",
    "source_split",
}
_LEGACY_REBUILD_MESSAGE = (
    "RL training data schema v1 is incompatible with closed-loop SFT v3; "
    "rerun data/build_rl.py before data/build_sft.py"
)


def _schema_version(value: Any) -> int | None:
    value = _plain(value)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _sft_child_id(parent_sample_id: str, image_key: str) -> str:
    """Return a stable child identity independent of sampling configuration."""

    identity = canonical_json(
        {
            "parent_sample_id": parent_sample_id,
            "image_key": image_key,
        }
    )
    return f"sft:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _validated_query_images(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_parent_sample_id = str(_lookup(record, "sample_id", "")).strip()
    parent_sample_id = raw_parent_sample_id or "<missing>"
    if _schema_version(_lookup(record, "schema_version")) != RL_INPUT_SCHEMA_VERSION:
        raise ValueError(f"{_LEGACY_REBUILD_MESSAGE}: parent {parent_sample_id}")
    for legacy_key in ("image_key", "dataset_image_id", "image", "source_file_name", "source_split"):
        if legacy_key in record and not _is_missing(record.get(legacy_key)):
            raise ValueError(
                f"schema v2 RL parent {parent_sample_id} contains ambiguous scalar {legacy_key!r}; "
                "rerun split, corpus, and sft"
            )

    raw_query_images = _as_list(_lookup(record, "query_images", []))
    raw_image_keys = [str(value).strip() for value in _as_list(_lookup(record, "image_keys", []))]
    raw_images = _as_list(record.get("images"))
    image_count = _lookup(record, "image_count")
    category_key = str(_lookup(record, "category_key", "")).strip()
    dataset_category_id = str(_lookup(record, "dataset_category_id", "")).strip()
    if (
        not raw_parent_sample_id
        or not category_key
        or not dataset_category_id
        or not raw_query_images
        or _schema_version(image_count) != len(raw_query_images)
        or len(raw_image_keys) != len(raw_query_images)
        or len(raw_images) != len(raw_query_images)
    ):
        raise ValueError(
            f"malformed schema v2 RL parent {parent_sample_id}: shared category identity, "
            "query_images, image_keys, image_count, and images must describe the same "
            "non-empty ordered image list"
        )

    query_images: list[dict[str, Any]] = []
    seen_image_keys: set[str] = set()
    for position, raw_query_image in enumerate(raw_query_images, start=1):
        query_image = _plain(raw_query_image)
        if not isinstance(query_image, Mapping) or set(query_image) != _QUERY_IMAGE_KEYS:
            raise ValueError(
                f"malformed schema v2 RL parent {parent_sample_id}: query_images[{position - 1}] "
                f"must contain exactly {sorted(_QUERY_IMAGE_KEYS)}"
            )
        image_index = _schema_version(query_image.get("image_index"))
        image_key = str(query_image.get("image_key") or "").strip()
        image_path = str(query_image.get("image") or "").strip()
        physical_image = _plain(raw_images[position - 1])
        if isinstance(physical_image, Mapping):
            physical_path = str(physical_image.get("image") or "").strip()
        else:
            physical_path = str(physical_image or "").strip()
        required_strings = (
            image_key,
            image_path,
            str(query_image.get("dataset_image_id") or "").strip(),
            str(query_image.get("source_file_name") or "").strip(),
            str(query_image.get("source_split") or "").strip(),
        )
        if (
            image_index != position
            or not all(required_strings)
            or raw_image_keys[position - 1] != image_key
            or physical_path != image_path
            or image_key in seen_image_keys
        ):
            raise ValueError(
                f"malformed schema v2 RL parent {parent_sample_id}: image position {position} "
                "has inconsistent index, identity, path, or duplicate image_key"
            )
        seen_image_keys.add(image_key)
        query_images.append(dict(query_image))

    prompt = _plain(record.get("prompt")) or []
    prompt_text = "".join(
        str(message.get("content") or "")
        for message in prompt
        if isinstance(message, Mapping)
    )
    if prompt_text.count("<image>") != len(query_images):
        raise ValueError(
            f"malformed schema v2 RL parent {parent_sample_id}: prompt/image placeholder count mismatch"
        )
    return query_images


def _single_image_child(
    parent: Mapping[str, Any],
    query_image: Mapping[str, Any],
) -> dict[str, Any]:
    parent_sample_id = str(_lookup(parent, "sample_id", "")).strip()
    source_image_index = int(query_image["image_index"])
    image_key = str(query_image["image_key"])
    image_path = str(query_image["image"])
    sample_id = _sft_child_id(parent_sample_id, image_key)
    question = str(_lookup(parent, "question", "")).strip()

    child = dict(_plain(parent))
    child.update(
        {
            "schema_version": SFT_SCHEMA_VERSION,
            "sample_id": sample_id,
            "parent_sample_id": parent_sample_id,
            "source_image_index": source_image_index,
            "image_key": image_key,
            "query_images": [{**dict(query_image), "image_index": 1}],
            "image_keys": [image_key],
            "image_count": 1,
            "images": [{"image": image_path}],
            "prompt": [{"role": "user", "content": _student_prompt({}, question)}],
        }
    )
    extra_info = _plain(parent.get("extra_info")) or {}
    if not isinstance(extra_info, Mapping):
        extra_info = {}
    child["extra_info"] = {
        **dict(extra_info),
        "schema_version": SFT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "parent_sample_id": parent_sample_id,
        "source_image_index": source_image_index,
        "image_key": image_key,
        "query_images": [{**dict(query_image), "image_index": 1}],
        "image_keys": [image_key],
        "image_count": 1,
    }
    return child


def _eligible_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    eligible_by_taxon: Counter[str] = Counter()
    for raw_record in records:
        record = dict(_plain(raw_record))
        query_images = _validated_query_images(record)
        if is_multi_hop_record(record):
            excluded["two_hop"] += 1
            continue
        question_type = str(_lookup(record, "question_type", "")).strip()
        if question_type != "automatic":
            excluded["non_automatic"] += 1
            continue
        if len(query_images) != 1:
            excluded["not_single_image"] += 1
            continue
        if _lookup(record, "retrieval_resolvable", False) is not True:
            excluded["retrieval_unresolvable"] += 1
            continue
        coarse_taxon = str(_lookup(record, "coarse_taxon", "")).strip()
        if coarse_taxon not in ALLOWED_COARSE_TAXA:
            excluded["unsupported_coarse_taxon"] += 1
            continue
        eligible.append(_single_image_child(record, query_images[0]))
        eligible_by_taxon[coarse_taxon] += 1
    return eligible, excluded, eligible_by_taxon


def _sample_targets(records: Sequence[Mapping[str, Any]], fraction: float) -> dict[str, int]:
    counts = Counter(str(_lookup(record, "coarse_taxon", "")) for record in records)
    return {
        coarse_taxon: min(count, max(1, int(math.ceil(count * fraction))))
        for coarse_taxon, count in sorted(counts.items())
    }


def _connected_train_val_split(
    rows: Sequence[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows or validation_fraction <= 0:
        return list(rows), []

    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen_images: dict[str, int] = {}
    seen_parents: dict[str, int] = {}
    for index, row in enumerate(rows):
        for value, seen in (
            (str(row["image_key"]), seen_images),
            (str(row["parent_sample_id"]), seen_parents),
        ):
            previous = seen.setdefault(value, index)
            union(index, previous)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    if len(groups) < 2:
        return list(rows), []

    by_taxon = Counter(str(row["coarse_taxon"]) for row in rows)
    desired = {
        coarse_taxon: max(1, int(round(count * validation_fraction)))
        for coarse_taxon, count in by_taxon.items()
    }
    val_keys: set[int] = set()
    current: Counter[str] = Counter()
    component_ids = {
        key: canonical_json(
            sorted(
                (str(row["parent_sample_id"]), str(row["image_key"]), str(row["sample_id"]))
                for row in component_rows
            )
        )
        for key, component_rows in groups.items()
    }
    candidates = sorted(groups, key=lambda key: _stable_key(seed, "sft_val", component_ids[key]))

    while any(current[key] < desired[key] for key in desired) and len(val_keys) < len(groups) - 1:
        best_key: int | None = None
        best_score: tuple[float, str] | None = None
        for component_key in candidates:
            if component_key in val_keys:
                continue
            contribution = Counter(str(row["coarse_taxon"]) for row in groups[component_key])
            gain = sum(min(contribution[key], max(0, desired[key] - current[key])) for key in desired)
            if gain <= 0:
                continue
            overshoot = sum(max(0, current[key] + contribution[key] - desired[key]) for key in desired)
            score = (
                gain - 0.01 * overshoot,
                _stable_key(seed, "sft_val_choice", component_ids[component_key]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_key = component_key
        if best_key is None:
            break
        val_keys.add(best_key)
        current.update(str(row["coarse_taxon"]) for row in groups[best_key])

    val_sample_ids = {
        str(row["sample_id"])
        for component_key in val_keys
        for row in groups[component_key]
    }
    train_rows = [row for row in rows if str(row["sample_id"]) not in val_sample_ids]
    val_rows = [row for row in rows if str(row["sample_id"]) in val_sample_ids]
    train_rows.sort(key=lambda row: _stable_key(seed, "sft_train_order", row["sample_id"]))
    val_rows.sort(key=lambda row: _stable_key(seed, "sft_val_order", row["sample_id"]))
    return train_rows, val_rows


def build_sft_records(
    train_records: Sequence[Mapping[str, Any]],
    vision_corpus: Sequence[Mapping[str, Any]],
    text_corpus: Sequence[Mapping[str, Any]],
    heldout_image_keys: set[str],
    teacher: TeacherClient,
    retriever: RetrieverClient,
    config: SFTBuilderConfig | None = None,
    observation_tokenizer: Any | None = None,
) -> SFTBuildResult:
    """Build SFT rows without performing file I/O.

    This is the primary test seam: fixtures can pass fake teacher/retriever
    clients and tiny in-memory corpora without real models or EVQA data.
    """

    config = config or SFTBuilderConfig()
    vision_corpus = [dict(_plain(row)) for row in vision_corpus]
    text_corpus = [dict(_plain(row)) for row in text_corpus]
    vision_corpus_by_id = _corpus_table(vision_corpus, label="vision")
    text_corpus_by_id = _corpus_table(text_corpus, label="text")
    leaked = sorted(
        {str(row.get("image_key", "")) for row in vision_corpus if str(row.get("image_key", "")) in heldout_image_keys}
    )
    if leaked:
        raise ValueError(f"vision corpus contains held-out query images: {leaked[:5]}")

    eligible, excluded, eligible_by_taxon = _eligible_records(train_records)
    missing_heldout = sorted(
        {
            str(_lookup(record, "image_key", ""))
            for record in eligible
            if str(_lookup(record, "image_key", "")) not in heldout_image_keys
        }
    )
    if missing_heldout:
        raise ValueError(
            "schema v2 RL query images are absent from the heldout manifest; "
            f"rerun split, corpus, and sft: {missing_heldout[:5]}"
        )
    targets = _sample_targets(eligible, config.sample_fraction)
    by_taxon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        by_taxon[str(_lookup(record, "coarse_taxon", ""))].append(record)

    successes: list[dict[str, Any]] = []
    success_counts: Counter[str] = Counter()
    attempted_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    failures_by_taxon: dict[str, Counter[str]] = defaultdict(Counter)
    failures_by_stage: Counter[str] = Counter()

    for coarse_taxon in ALLOWED_COARSE_TAXA:
        candidates = sorted(
            by_taxon.get(coarse_taxon, []),
            key=lambda record: _stable_key(
                config.seed,
                "sft_candidate",
                coarse_taxon,
                _lookup(record, "sample_id", ""),
            ),
        )
        for record in candidates:
            if success_counts[coarse_taxon] >= targets.get(coarse_taxon, 0):
                break
            attempted_counts[coarse_taxon] += 1
            try:
                trajectory = _build_trajectory(
                    record,
                    vision_corpus_by_id,
                    text_corpus_by_id,
                    teacher,
                    retriever,
                    config,
                    observation_tokenizer=observation_tokenizer,
                )
            except TrajectoryBuildError as exc:
                key = f"{exc.stage}:{exc.reason}"
                failure_reasons[key] += 1
                failures_by_taxon[coarse_taxon][key] += 1
                failures_by_stage[exc.stage] += 1
                continue
            successes.append(trajectory)
            success_counts[coarse_taxon] += 1

    train_rows, val_rows = _connected_train_val_split(
        successes, config.validation_fraction, config.seed
    )
    train_image_keys = {row["image_key"] for row in train_rows}
    val_image_keys = {row["image_key"] for row in val_rows}
    train_parent_ids = {row["parent_sample_id"] for row in train_rows}
    val_parent_ids = {row["parent_sample_id"] for row in val_rows}
    if train_image_keys & val_image_keys:
        raise AssertionError("SFT train/validation image groups overlap")
    if train_parent_ids & val_parent_ids:
        raise AssertionError("SFT train/validation parent question groups overlap")

    retrieval_report: dict[str, Any] = {
        "mode": (
            "real_http"
            if isinstance(retriever, HTTPRetrieverClient)
            else "injected_client"
        ),
        "top_k": config.retrieval_top_k,
        "response_documents_verified_against_local_corpus": True,
        "service_index_identity_verified": False,
        "positive_must_survive_observation_truncation": True,
    }
    if isinstance(retriever, HTTPRetrieverClient):
        retrieval_report.update(
            {
                "vision_url": retriever.config.vision_url,
                "text_url": retriever.config.text_url,
                "timeout_seconds": retriever.config.timeout_seconds,
                "max_retries": retriever.config.max_retries,
            }
        )

    report = {
        "schema_version": SFT_SCHEMA_VERSION,
        "config": asdict(config),
        "input": {
            "train_rows": len(train_records),
            "rl_parent_samples": len(train_records),
            "vision_corpus_rows": len(vision_corpus),
            "text_corpus_rows": len(text_corpus),
            "heldout_image_keys": len(heldout_image_keys),
        },
        "eligibility": {
            "eligible": len(eligible),
            "eligible_parent_samples": len(eligible),
            "single_image_candidates": len(eligible),
            "excluded": dict(sorted(excluded.items())),
            "eligible_by_coarse_taxon": {
                taxon: eligible_by_taxon.get(taxon, 0)
                for taxon in ALLOWED_COARSE_TAXA
            },
        },
        "sampling": {
            "targets_by_coarse_taxon": {
                taxon: targets.get(taxon, 0) for taxon in ALLOWED_COARSE_TAXA
            },
            "attempted_by_coarse_taxon": {
                taxon: attempted_counts.get(taxon, 0) for taxon in ALLOWED_COARSE_TAXA
            },
            "successful_by_coarse_taxon": {
                taxon: success_counts.get(taxon, 0) for taxon in ALLOWED_COARSE_TAXA
            },
            "target_shortfall_by_coarse_taxon": {
                key: max(0, targets[key] - success_counts[key]) for key in sorted(targets)
            },
            "cross_taxon_backfill": 0,
        },
        "failures": {
            "total": sum(failure_reasons.values()),
            "by_reason": dict(sorted(failure_reasons.items())),
            "by_stage": dict(sorted(failures_by_stage.items())),
            "by_coarse_taxon": {
                key: dict(sorted(counter.items()))
                for key, counter in sorted(failures_by_taxon.items())
            },
        },
        "output": {
            "successful": len(successes),
            "sft_train": len(train_rows),
            "sft_val": len(val_rows),
            "sft_train_image_groups": len(train_image_keys),
            "sft_val_image_groups": len(val_image_keys),
            "sft_train_parent_groups": len(train_parent_ids),
            "sft_val_parent_groups": len(val_parent_ids),
            "image_group_overlap": 0,
            "parent_group_overlap": 0,
            "sft_train_by_coarse_taxon": dict(
                sorted(Counter(row["coarse_taxon"] for row in train_rows).items())
            ),
            "sft_val_by_coarse_taxon": dict(
                sorted(Counter(row["coarse_taxon"] for row in val_rows).items())
            ),
        },
        "retrieval": retrieval_report,
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
    expected_digest = (
        heldout.get("image_keys_sha256")
        if isinstance(heldout, Mapping)
        else None
    )
    if not expected_digest:
        raise ValueError(
            "build_manifest.json is missing heldout.image_keys_sha256; "
            "rerun data/build_rl.py."
        )
    if stable_digest(sorted(result)) != str(expected_digest):
        raise ValueError(
            "heldout image keys do not match build_manifest.json; rerun "
            "data/build_rl.py."
        )
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


def _publish_staged_files(staged: Sequence[tuple[Path, Path]]) -> None:
    if not staged:
        return
    backup_root = Path(
        tempfile.mkdtemp(prefix=".sft-publish-backup-", dir=staged[0][1].parent)
    )
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, (_, final) in enumerate(staged):
            if final.exists():
                backup = backup_root / f"{index:02d}-{final.name}"
                os.replace(final, backup)
                backups.append((final, backup))
        for temporary, final in staged:
            os.replace(temporary, final)
            published.append(final)
    except Exception:
        for final in reversed(published):
            final.unlink(missing_ok=True)
        for final, backup in reversed(backups):
            if backup.exists():
                os.replace(backup, final)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def _validate_rl_artifact_generation(
    manifest: Mapping[str, Any],
    *,
    train_path: Path,
    vision_path: Path,
    text_path: Path,
) -> None:
    if (
        _schema_version(manifest.get("schema_version")) != RL_INPUT_SCHEMA_VERSION
        or manifest.get("stage") != "fixed_11k_rl"
        or manifest.get("status") != "complete"
        or not isinstance(manifest.get("build_fingerprint"), str)
        or not str(manifest.get("build_fingerprint")).strip()
    ):
        raise ValueError(
            "build_manifest.json is not a completed fixed-11K RL generation; "
            "rerun data/build_rl.py."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(
            "RL build manifest has no artifact fingerprint table; rerun build_rl.py."
        )
    for name, path in (
        ("train.parquet", train_path),
        ("vision_corpus.jsonl", vision_path),
        ("text_corpus.jsonl", text_path),
    ):
        record = artifacts.get(name)
        if not isinstance(record, Mapping) or not str(
            record.get("sha256") or ""
        ).strip():
            raise ValueError(
                f"RL build manifest has no SHA256 for {name}; rerun build_rl.py."
            )
        actual = sha256_file(path)
        if actual != str(record["sha256"]):
            raise ValueError(
                f"{name} does not match build_manifest.json; do not mix "
                "artifacts from different RL builds."
            )


def build_sft_files(
    *,
    train_parquet: str | Path,
    vision_corpus_path: str | Path,
    text_corpus_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    teacher: TeacherClient,
    retriever: RetrieverClient,
    config: SFTBuilderConfig | None = None,
    observation_tokenizer: Any | None = None,
    build_fingerprint: str | None = None,
) -> SFTBuildResult:
    import pandas as pd

    train_path = Path(train_parquet)
    vision_path = Path(vision_corpus_path)
    text_path = Path(text_corpus_path)
    manifest_file = Path(manifest_path)
    for path in (train_path, vision_path, text_path, manifest_file):
        if not path.is_file():
            raise FileNotFoundError(f"required SFT input does not exist: {path}")

    manifest = _load_json(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"build manifest is not a JSON object: {manifest_file}")
    if _schema_version(manifest.get("schema_version")) != RL_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"{_LEGACY_REBUILD_MESSAGE}: manifest {manifest_file} is not schema v2"
        )
    _validate_rl_artifact_generation(
        manifest,
        train_path=train_path,
        vision_path=vision_path,
        text_path=text_path,
    )
    train_records = [
        _plain(row)
        for row in pd.read_parquet(train_path).to_dict(orient="records")
    ]
    vision_corpus = _load_jsonl(vision_path)
    text_corpus = _load_jsonl(text_path)
    heldout_keys = _load_heldout_image_keys(manifest, manifest_file)
    if not heldout_keys:
        raise ValueError("manifest contains no held-out/query image keys; refusing to build SFT data")

    result = build_sft_records(
        train_records,
        vision_corpus,
        text_corpus,
        heldout_keys,
        teacher,
        retriever,
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
        if build_fingerprint is not None:
            report["build_fingerprint"] = build_fingerprint
        report["paths"] = {
            "train_parquet": str(train_path.resolve()),
            "vision_corpus": str(vision_path.resolve()),
            "text_corpus": str(text_path.resolve()),
            "manifest": str(manifest_file.resolve()),
            "sft_train": str(train_final.resolve()),
            "sft_val": str(val_final.resolve()),
        }
        staged.append((_stage_json(report, report_final), report_final))
        _publish_staged_files(staged)
        result.report = report
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return result


_SFT_CONFIG_KEYS = {
    "teacher_base_url",
    "teacher_model",
    "teacher_api_key_env",
    "observation_tokenizer_path",
    "timeout",
    "retries",
    "vision_retriever_url",
    "text_retriever_url",
    "retriever_timeout",
    "retriever_retries",
}
_FIXED_SFT_FIELDS = {
    "sample_fraction": 0.10,
    "validation_fraction": 0.10,
    "retrieval_top_k": 3,
    "max_tool_response_tokens": DEFAULT_TOOL_RESPONSE_TOKENS,
    "coarse_taxa": list(ALLOWED_COARSE_TAXA),
    "question_type": "automatic",
    "image_count": 1,
}
_SFT_CACHE_MARKER = "sft_build_manifest.json"
_SFT_OUTPUT_NAMES = (
    "sft_train.parquet",
    "sft_val.parquet",
    "sft_build_report.json",
)


def _standalone_config(config_path: str | Path) -> tuple[Path, int, dict[str, Any]]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    value = _load_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("Config root must be a JSON object.")

    output_text = os.path.expandvars(os.path.expanduser(str(value.get("output_dir") or "")))
    output_dir = Path(output_text)
    if not output_text or not output_dir.is_absolute():
        raise ValueError("output_dir must be an absolute path.")
    output_dir = output_dir.resolve()
    if output_dir == PROJECT_ROOT or output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError(
            "output_dir must be outside the DualSearch repository checkout."
        )

    raw_seed = value.get("seed", 42)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError("seed must be an integer.")

    raw_sft = value.get("sft")
    if not isinstance(raw_sft, Mapping):
        raise ValueError("sft must be a JSON object.")
    forbidden = sorted(set(raw_sft).intersection(_FIXED_SFT_FIELDS))
    if forbidden:
        fixed = ", ".join(
            f"{name}={_FIXED_SFT_FIELDS[name]!r}" for name in forbidden
        )
        raise ValueError(
            "SFT sampling/retrieval/truncation policy is fixed and cannot be "
            f"overridden in config: {fixed}."
        )
    unknown = sorted(set(raw_sft).difference(_SFT_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"Unsupported sft config fields: {unknown}")

    sft = dict(raw_sft)
    for required in (
        "teacher_base_url",
        "teacher_model",
        "observation_tokenizer_path",
        "vision_retriever_url",
        "text_retriever_url",
    ):
        if not str(sft.get(required) or "").strip():
            raise ValueError(f"sft.{required} must be a non-empty string.")
    for field in ("teacher_model", "observation_tokenizer_path"):
        raw_model_path = Path(
            os.path.expandvars(
                os.path.expanduser(str(sft[field]).strip())
            )
        )
        local_model_path = (
            raw_model_path
            if raw_model_path.is_absolute()
            else path.parent / raw_model_path
        ).resolve()
        if not local_model_path.is_dir():
            raise FileNotFoundError(
                f"sft.{field} must be an existing local model directory: "
                f"{local_model_path}"
            )
        sft[field] = str(local_model_path)
    timeout = sft.get("timeout", 120.0)
    if isinstance(timeout, bool):
        raise ValueError("sft.timeout must be positive.")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("sft.timeout must be positive.") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("sft.timeout must be positive.")
    retries = sft.get("retries", 2)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("sft.retries must be a non-negative integer.")
    sft["timeout"] = timeout
    sft["retries"] = retries
    retriever_timeout = sft.get("retriever_timeout", 120.0)
    if isinstance(retriever_timeout, bool):
        raise ValueError("sft.retriever_timeout must be positive.")
    try:
        retriever_timeout = float(retriever_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("sft.retriever_timeout must be positive.") from exc
    if not math.isfinite(retriever_timeout) or retriever_timeout <= 0:
        raise ValueError("sft.retriever_timeout must be positive.")
    retriever_retries = sft.get("retriever_retries", 2)
    if (
        isinstance(retriever_retries, bool)
        or not isinstance(retriever_retries, int)
        or retriever_retries < 0
    ):
        raise ValueError("sft.retriever_retries must be a non-negative integer.")
    for field in ("vision_retriever_url", "text_retriever_url"):
        url = str(sft[field]).strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"sft.{field} must be an HTTP(S) URL.")
        sft[field] = url
    sft["retriever_timeout"] = retriever_timeout
    sft["retriever_retries"] = retriever_retries
    return output_dir, raw_seed, sft


def _resolve_observation_tokenizer_path(
    config_path: Path,
    sft: Mapping[str, Any],
) -> Path | None:
    raw = str(sft.get("observation_tokenizer_path") or "").strip()
    if not raw:
        raise ValueError(
            "sft.observation_tokenizer_path is required and must be local."
        )
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    path = (
        expanded
        if expanded.is_absolute()
        else config_path.parent / expanded
    ).resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"observation tokenizer must be a local directory: {path}"
        )
    return path


def _query_pixel_fingerprint(train_path: Path) -> dict[str, Any]:
    import pandas as pd

    unique: dict[tuple[str, str], dict[str, str]] = {}
    frame = pd.read_parquet(train_path, columns=["query_images"])
    for raw_images in frame["query_images"].tolist():
        images = _plain(raw_images)
        if not isinstance(images, list):
            raise ValueError("train.parquet query_images must be a list.")
        for raw_image in images:
            if not isinstance(raw_image, Mapping):
                raise ValueError(
                    "train.parquet contains an invalid query image record."
                )
            image_key = str(raw_image.get("image_key") or "").strip()
            raw_path = str(raw_image.get("image") or "").strip()
            if not image_key or not raw_path:
                raise ValueError(
                    "train.parquet contains an incomplete query image record."
                )
            normalized_path = os.path.normcase(
                str(
                    Path(
                        os.path.expandvars(os.path.expanduser(raw_path))
                    ).resolve(strict=False)
                )
            )
            key = (image_key, normalized_path)
            if key in unique:
                continue
            image_path = Path(normalized_path)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"RL query image does not exist: {image_path}"
                )
            unique[key] = {
                "image_key": image_key,
                "normalized_path": normalized_path,
                "file_sha256": sha256_file(image_path),
            }
    ordered = [unique[key] for key in sorted(unique)]
    return {"count": len(ordered), "sha256": stable_digest(ordered)}


def _resolved_artifact_path(value: Any, *, base_dir: Path) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not text:
        raise ValueError("index generation contains an empty artifact path")
    path = Path(text)
    return (path if path.is_absolute() else base_dir / path).resolve()


def _validate_index_generation(
    *,
    output_dir: Path,
    rl_manifest: Mapping[str, Any],
    vision_corpus_path: Path,
    text_corpus_path: Path,
    index_report: Mapping[str, Any],
    index_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the local indexes still belong to the current RL corpora."""

    for label, value in (
        ("index_report.json", index_report),
        ("index_build_manifest.json", index_manifest),
    ):
        if (
            _schema_version(value.get("schema_version")) != 2
            or value.get("stage") != "index"
            or value.get("status") != "complete"
        ):
            raise ValueError(
                f"{label} is not a completed index generation; "
                "rerun data/build_index.py."
            )

    build_fingerprint = index_report.get("build_fingerprint")
    if (
        not isinstance(build_fingerprint, str)
        or not build_fingerprint
        or index_manifest.get("build_fingerprint") != build_fingerprint
    ):
        raise ValueError(
            "index report and cache manifest have different build fingerprints; "
            "rerun data/build_index.py."
        )
    cached_inputs = index_manifest.get("inputs")
    if (
        not isinstance(cached_inputs, Mapping)
        or _schema_version(cached_inputs.get("schema_version")) != 2
        or cached_inputs.get("builder") != "dual_search_indexes"
        or stable_digest(dict(cached_inputs)) != build_fingerprint
    ):
        raise ValueError(
            "index cache input fingerprint is missing or corrupt; "
            "rerun data/build_index.py."
        )

    report_rl_generation = index_report.get("rl_generation")
    cached_rl_generation = cached_inputs.get("rl_generation")
    if (
        not isinstance(report_rl_generation, Mapping)
        or not isinstance(cached_rl_generation, Mapping)
        or stable_digest(dict(report_rl_generation))
        != stable_digest(dict(cached_rl_generation))
    ):
        raise ValueError(
            "index report and cache manifest reference different RL generations; "
            "rerun data/build_index.py."
        )
    rl_build_fingerprint = str(rl_manifest.get("build_fingerprint") or "")
    if (
        report_rl_generation.get("build_fingerprint") != rl_build_fingerprint
        or report_rl_generation.get("manifest_sha256")
        != sha256_file(output_dir / "build_manifest.json")
    ):
        raise ValueError(
            "indexes do not belong to the current RL build manifest; "
            "rerun data/build_index.py."
        )

    current_corpus_fingerprints = {
        "vision": corpus_fingerprint(
            vision_corpus_path,
            id_keys=("id", "image_key"),
        ),
        "text": corpus_fingerprint(text_corpus_path, id_keys=("id",)),
    }
    expected_index_dirs = {
        "vision": (output_dir / "indexes" / "vision").resolve(),
        "text": (output_dir / "indexes" / "text").resolve(),
    }
    report_tables: dict[str, Mapping[str, Any]] = {}
    for kind, corpus_path, encoder_key in (
        ("vision", vision_corpus_path, "vision_encoder_config"),
        ("text", text_corpus_path, "text_encoder_config"),
    ):
        report_table = index_report.get(kind)
        if not isinstance(report_table, Mapping):
            raise ValueError(f"index report has no {kind} section")
        report_tables[kind] = report_table
        expected_corpus = current_corpus_fingerprints[kind]
        for source, fingerprint in (
            ("index report", report_table.get("corpus_fingerprint")),
            ("index report RL generation", report_rl_generation.get(f"{kind}_corpus")),
            ("index cache RL generation", cached_rl_generation.get(f"{kind}_corpus")),
        ):
            if (
                not isinstance(fingerprint, Mapping)
                or stable_digest(dict(fingerprint)) != stable_digest(expected_corpus)
            ):
                raise ValueError(
                    f"{source} has a stale {kind} corpus fingerprint; "
                    "rerun data/build_index.py."
                )
        if _resolved_artifact_path(
            report_table.get("corpus"),
            base_dir=output_dir,
        ) != corpus_path.resolve():
            raise ValueError(
                f"index report points at a different {kind} corpus; "
                "rerun data/build_index.py."
            )
        if _resolved_artifact_path(
            report_table.get("index_dir"),
            base_dir=output_dir,
        ) != expected_index_dirs[kind]:
            raise ValueError(
                f"index report points at a different {kind} index directory; "
                "rerun data/build_index.py."
            )
        cached_encoder = cached_inputs.get(encoder_key)
        report_encoder = report_table.get("encoder_config")
        if (
            not isinstance(cached_encoder, Mapping)
            or not isinstance(report_encoder, Mapping)
            or stable_digest(dict(cached_encoder))
            != stable_digest(dict(report_encoder))
        ):
            raise ValueError(
                f"index report and cache manifest have different {kind} encoders; "
                "rerun data/build_index.py."
            )

    expected_outputs = {
        "vision": expected_index_dirs["vision"],
        "text": expected_index_dirs["text"],
        "report": (output_dir / "index_report.json").resolve(),
    }
    recorded_outputs = index_manifest.get("outputs")
    if not isinstance(recorded_outputs, Mapping):
        raise ValueError("index cache manifest has no output fingerprints")
    actual_output_fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in expected_outputs.items():
        record = recorded_outputs.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"index cache manifest is missing output {name!r}")
        if _resolved_artifact_path(record.get("path"), base_dir=output_dir) != path:
            raise ValueError(
                f"index cache output {name!r} points at a different path; "
                "rerun data/build_index.py."
            )
        expected_fingerprint = record.get("fingerprint")
        actual_fingerprint = artifact_fingerprint(path)
        if (
            not isinstance(expected_fingerprint, Mapping)
            or stable_digest(dict(expected_fingerprint))
            != stable_digest(actual_fingerprint)
        ):
            raise ValueError(
                f"index output {name!r} was modified after publication; "
                "rerun data/build_index.py."
            )
        actual_output_fingerprints[name] = actual_fingerprint

    load_and_validate_index_meta(
        expected_index_dirs["vision"] / "vision_index_meta.json",
        vision_corpus_path,
        expected_kind="vision",
        id_keys=("id", "image_key"),
        expected_encoder_config=report_tables["vision"]["encoder_config"],
    )
    load_and_validate_index_meta(
        expected_index_dirs["text"] / "text_index_meta.json",
        text_corpus_path,
        expected_kind="text",
        id_keys=("id",),
        expected_encoder_config=report_tables["text"]["encoder_config"],
    )
    return {
        "build_fingerprint": build_fingerprint,
        "rl_build_fingerprint": rl_build_fingerprint,
        "corpus_fingerprints": current_corpus_fingerprints,
        "output_fingerprints": actual_output_fingerprints,
    }


def _sft_build_fingerprint(
    *,
    output_dir: Path,
    seed: int,
    sft: Mapping[str, Any],
    tokenizer_path: Path | None,
) -> tuple[str, dict[str, Any]]:
    inputs: dict[str, Any] = {}
    for name in (
        "train.parquet",
        "vision_corpus.jsonl",
        "text_corpus.jsonl",
        "build_manifest.json",
        "index_report.json",
    ):
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required SFT input does not exist: {path}")
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    index_manifest_path = output_dir / ".build_cache" / "index_build_manifest.json"
    if not index_manifest_path.is_file():
        raise FileNotFoundError(
            "Required SFT input does not exist: "
            f"{index_manifest_path}. Run data/build_index.py before build_sft.py."
        )
    inputs["index_build_manifest.json"] = {
        "path": str(index_manifest_path),
        "sha256": sha256_file(index_manifest_path),
    }
    manifest = _load_json(output_dir / "build_manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("build_manifest.json must be a JSON object.")
    _validate_rl_artifact_generation(
        manifest,
        train_path=output_dir / "train.parquet",
        vision_path=output_dir / "vision_corpus.jsonl",
        text_path=output_dir / "text_corpus.jsonl",
    )
    heldout_keys = _load_heldout_image_keys(
        manifest,
        output_dir / "build_manifest.json",
    )
    if not heldout_keys:
        raise ValueError(
            "build_manifest.json contains no heldout image keys."
        )
    query_pixels = _query_pixel_fingerprint(output_dir / "train.parquet")
    expected_pixel_table = manifest.get("pixel_fingerprints")
    expected_query_pixels = (
        expected_pixel_table.get("query")
        if isinstance(expected_pixel_table, Mapping)
        else None
    )
    if not isinstance(expected_query_pixels, Mapping) or (
        str(expected_query_pixels.get("sha256") or "")
        != query_pixels["sha256"]
        or int(expected_query_pixels.get("count", -1))
        != int(query_pixels["count"])
    ):
        raise ValueError(
            "RL query image pixels do not match build_manifest.json; rerun "
            "data/build_rl.py before building SFT."
        )
    inputs["heldout_image_keys"] = {
        "count": len(heldout_keys),
        "sha256": stable_digest(sorted(heldout_keys)),
    }
    inputs["query_pixels"] = query_pixels
    index_report = _load_json(output_dir / "index_report.json")
    index_manifest = _load_json(index_manifest_path)
    if not isinstance(index_report, Mapping) or not isinstance(index_manifest, Mapping):
        raise ValueError("index report and cache manifest must be JSON objects")
    inputs["index_generation"] = {
        **_validate_index_generation(
            output_dir=output_dir,
            rl_manifest=manifest,
            vision_corpus_path=output_dir / "vision_corpus.jsonl",
            text_corpus_path=output_dir / "text_corpus.jsonl",
            index_report=index_report,
            index_manifest=index_manifest,
        ),
        "report_sha256": inputs["index_report.json"]["sha256"],
        "manifest_sha256": inputs["index_build_manifest.json"]["sha256"],
    }
    execution_config = {
        key: sft.get(key)
        for key in sorted(_SFT_CONFIG_KEYS)
        if key != "teacher_api_key_env"
    }
    execution_config["teacher_api_key_env"] = str(
        sft.get("teacher_api_key_env") or "VLLM_API_KEY"
    )
    execution_config["teacher_model_fingerprint"] = model_fingerprint(
        str(sft["teacher_model"])
    )
    tokenizer_fingerprint = (
        model_fingerprint(tokenizer_path)
        if tokenizer_path is not None
        else {"mode": "unavailable"}
    )
    payload = {
        "schema_version": SFT_SCHEMA_VERSION,
        "builder": "dual_search_closed_loop_sft_v3",
        "seed": seed,
        "fixed_recipe": _FIXED_SFT_FIELDS,
        "execution": execution_config,
        "prompts": {
            "vision_sha256": stable_digest(
                {
                    "system": VISION_SYSTEM_PROMPT,
                    "request": VISION_USER_REQUEST,
                }
            ),
            "search_sha256": stable_digest(
                {
                    "system": SEARCH_SYSTEM_PROMPT,
                    "request": SEARCH_USER_REQUEST,
                }
            ),
            "answer_sha256": stable_digest(
                {
                    "system": ANSWER_SYSTEM_PROMPT,
                    "request": ANSWER_USER_REQUEST,
                }
            ),
            "layout_sha256": stable_digest(
                {
                    "user_text": TEACHER_USER_TEXT_TEMPLATE,
                    "tool_history": TEACHER_TOOL_HISTORY_TEMPLATE,
                    "empty_history": NO_PREVIOUS_TOOL_INTERACTION,
                    "reference_answer": REFERENCE_ANSWER_TEMPLATE,
                }
            ),
            "query_schema": QUERY_STAGE_SCHEMA,
            "answer_schema": ANSWER_STAGE_SCHEMA,
        },
        "observation_tokenizer": tokenizer_fingerprint,
        "inputs": inputs,
    }
    return stable_digest(payload), inputs


def _sft_cache_reusable(
    *,
    output_dir: Path,
    cache_marker: Path,
    build_fingerprint: str,
) -> bool:
    if not cache_marker.is_file():
        return False
    try:
        marker = _load_json(cache_marker)
        report = _load_json(output_dir / "sft_build_report.json")
        if not isinstance(marker, Mapping) or not isinstance(report, Mapping):
            return False
        if (
            marker.get("schema_version") != SFT_SCHEMA_VERSION
            or report.get("schema_version") != SFT_SCHEMA_VERSION
            or marker.get("build_fingerprint") != build_fingerprint
            or report.get("build_fingerprint") != build_fingerprint
        ):
            return False
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return False
        for name in _SFT_OUTPUT_NAMES:
            record = artifacts.get(name)
            path = output_dir / name
            if (
                not isinstance(record, Mapping)
                or not path.is_file()
                or sha256_file(path) != str(record.get("sha256") or "")
            ):
                return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _write_sft_cache_marker(
    *,
    marker_path: Path,
    output_dir: Path,
    build_fingerprint: str,
    inputs: Mapping[str, Any],
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema_version": SFT_SCHEMA_VERSION,
        "stage": "sft",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_fingerprint": build_fingerprint,
        "inputs": dict(inputs),
        "artifacts": {
            name: {
                "path": str(output_dir / name),
                "sha256": sha256_file(output_dir / name),
            }
            for name in _SFT_OUTPUT_NAMES
        },
    }
    temporary = _stage_json(value, marker_path)
    try:
        os.replace(temporary, marker_path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verified DualSearch cold-start SFT Parquet files."
    )
    parser.add_argument("--config", required=True, help="Shared DualSearch JSON config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore a matching SFT cache manifest and rebuild.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    output_dir, seed, sft = _standalone_config(config_path)
    tokenizer_path = _resolve_observation_tokenizer_path(config_path, sft)
    build_fingerprint, inputs = _sft_build_fingerprint(
        output_dir=output_dir,
        seed=seed,
        sft=sft,
        tokenizer_path=tokenizer_path,
    )
    cache_marker = output_dir / ".build_cache" / _SFT_CACHE_MARKER
    if not args.force and _sft_cache_reusable(
        output_dir=output_dir,
        cache_marker=cache_marker,
        build_fingerprint=build_fingerprint,
    ):
        report = _load_json(output_dir / "sft_build_report.json")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    observation_tokenizer = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        observation_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
    api_key_env = str(sft.get("teacher_api_key_env") or "VLLM_API_KEY").strip()
    teacher = VLLMTeacherClient(
        TeacherConfig(
            base_url=str(sft["teacher_base_url"]).strip(),
            model=str(sft["teacher_model"]).strip(),
            api_key=os.getenv(api_key_env) if api_key_env else None,
            timeout_seconds=float(sft["timeout"]),
            max_retries=int(sft["retries"]),
        )
    )
    retriever = HTTPRetrieverClient(
        HTTPRetrieverConfig(
            vision_url=str(sft["vision_retriever_url"]),
            text_url=str(sft["text_retriever_url"]),
            timeout_seconds=float(sft["retriever_timeout"]),
            max_retries=int(sft["retriever_retries"]),
        )
    )
    result = build_sft_files(
        train_parquet=output_dir / "train.parquet",
        vision_corpus_path=output_dir / "vision_corpus.jsonl",
        text_corpus_path=output_dir / "text_corpus.jsonl",
        manifest_path=output_dir / "build_manifest.json",
        output_dir=output_dir,
        teacher=teacher,
        retriever=retriever,
        config=SFTBuilderConfig(
            sample_fraction=0.10,
            validation_fraction=0.10,
            seed=seed,
            retrieval_top_k=3,
            max_tool_response_tokens=DEFAULT_TOOL_RESPONSE_TOKENS,
        ),
        observation_tokenizer=observation_tokenizer,
        build_fingerprint=build_fingerprint,
    )
    _write_sft_cache_marker(
        marker_path=cache_marker,
        output_dir=output_dir,
        build_fingerprint=build_fingerprint,
        inputs=inputs,
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
