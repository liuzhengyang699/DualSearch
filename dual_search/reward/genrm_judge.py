import json
import re
from typing import Any

import aiohttp

from dual_search.reward.format_validator import extract_solution, is_valid_sequence


SYSTEM_PROMPT = (
    'Compare the candidate answer with the reference answer for the given question. '
    'If they are semantically equivalent, output {"score": 1.0}; otherwise, output {"score": 0.0}. '
    'Output only the JSON object.'
)

USER_PROMPT_TEMPLATE = """Question: {question}

Reference answer: {reference_answer}

Candidate answer: {candidate_answer}"""

QUESTION_MARKER_PATTERN = re.compile(r"(?:^|\n)\s*Question:\s*", re.IGNORECASE)

ANSWER_REWARD_WEIGHT = 0.8
FORMAT_REWARD_WEIGHT = 0.2
FREE_RETRIEVAL_CALLS = 2
RETRIEVAL_PENALTY_COEFFICIENT = 0.02


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _content_to_text(content: Any) -> str:
    content = _to_python(content)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        item_type = content.get("type")
        if item_type == "image":
            return ""
        if item_type == "text" and isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _content_to_text(content["content"])
        return ""
    if isinstance(content, (list, tuple)):
        return "\n".join(part for part in (_content_to_text(item) for item in content) if part)
    return ""


def extract_question(raw_prompt: Any) -> str:
    raw_prompt = _to_python(raw_prompt)
    if isinstance(raw_prompt, dict):
        messages = [raw_prompt]
    elif isinstance(raw_prompt, (list, tuple)):
        messages = list(raw_prompt)
    else:
        messages = []

    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _content_to_text(message.get("content", "")).strip()
        if not text:
            continue
        matches = list(QUESTION_MARKER_PATTERN.finditer(text))
        if matches:
            return text[matches[-1].end() :].strip()
        return text
    return ""


def format_reference_answer(ground_truth: Any) -> str:
    target = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
    target = _to_python(target)
    if isinstance(target, (list, tuple, set)):
        return " | ".join(str(item).strip() for item in target if str(item).strip())
    return str(target).strip() if target is not None else ""


def build_judge_messages(question: str, reference_answer: str, candidate_answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                question=question,
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
            ),
        },
    ]


def parse_judge_score(raw_output: str) -> float | None:
    try:
        parsed = json.loads(raw_output.strip())
    except Exception:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"score"}:
        return None
    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = float(score)
    return score if score in {0.0, 1.0} else None


def _nonnegative_int(value: Any) -> int:
    value = _to_python(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def get_retrieval_call_count(extra_info: dict | None) -> int:
    if not isinstance(extra_info, dict):
        return 0
    return _nonnegative_int(extra_info.get("valid_search_stats", 0)) + _nonnegative_int(
        extra_info.get("valid_vision_search_stats", 0)
    )


def compute_retrieval_penalty(retrieval_call_count: int) -> float:
    excess_calls = max(0, _nonnegative_int(retrieval_call_count) - FREE_RETRIEVAL_CALLS)
    return RETRIEVAL_PENALTY_COEFFICIENT * (excess_calls**2)


def compute_total_reward(judge_score: float, format_valid: bool, retrieval_call_count: int) -> float:
    return (
        ANSWER_REWARD_WEIGHT * float(judge_score)
        + FORMAT_REWARD_WEIGHT * float(format_valid)
        - compute_retrieval_penalty(retrieval_call_count)
    )


async def request_genrm(
    reward_router_address: str,
    genrm_model: str,
    messages: list[dict[str, str]],
    request_timeout: float,
) -> str:
    if not reward_router_address:
        raise ValueError("reward_router_address is required")
    if not genrm_model:
        raise ValueError("genrm_model is required")

    base_url = reward_router_address.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    payload = {
        "model": genrm_model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 32,
    }
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            result = await response.json(content_type=None)
    return result["choices"][0]["message"]["content"]


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    raw_prompt: Any = None,
    reward_router_address: str | None = None,
    reward_model_tokenizer: Any = None,
    genrm_model: str = "/path/to/GenRM",
    request_timeout: float = 120.0,
    **kwargs,
) -> dict[str, float]:
    del data_source, reward_model_tokenizer, kwargs

    format_valid, _ = is_valid_sequence(solution_str)
    candidate_answer = extract_solution(solution_str)
    question = extract_question(raw_prompt)
    reference_answer = format_reference_answer(ground_truth)
    retrieval_call_count = get_retrieval_call_count(extra_info)
    retrieval_penalty = compute_retrieval_penalty(retrieval_call_count)

    failure = {
        "score": compute_total_reward(0.0, format_valid, retrieval_call_count),
        "judge_score": 0.0,
        "format_score": float(format_valid),
        "judge_valid": 0.0,
        "retrieval_call_count": float(retrieval_call_count),
        "retrieval_penalty": retrieval_penalty,
    }
    if not question or not reference_answer or not candidate_answer:
        return failure

    messages = build_judge_messages(question, reference_answer, candidate_answer)
    try:
        raw_output = await request_genrm(
            reward_router_address=reward_router_address,
            genrm_model=genrm_model,
            messages=messages,
            request_timeout=float(request_timeout),
        )
        judge_score = parse_judge_score(raw_output)
    except Exception:
        return failure

    if judge_score is None:
        return failure

    return {
        "score": compute_total_reward(judge_score, format_valid, retrieval_call_count),
        "judge_score": judge_score,
        "format_score": float(format_valid),
        "judge_valid": 1.0,
        "retrieval_call_count": float(retrieval_call_count),
        "retrieval_penalty": retrieval_penalty,
    }
