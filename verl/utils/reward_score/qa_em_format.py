import random
import re
import string

from dual_search.protocol import extract_answer, validate_sequence
from dual_search.reward.genrm_judge import compute_retrieval_penalty, get_retrieval_call_count


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def is_valid_sequence(text):
    """Validate the same native-Qwen trajectory used by rollout and SFT.

    This compatibility entry point is still imported by veRL's default QA
    reward registry, so it must not silently accept DualSearch's retired XML
    action protocol.
    """

    return validate_sequence(text)


def extract_solution(solution_str):
    return extract_answer(solution_str)


def compute_score_em(
    solution_str,
    ground_truth,
    method="strict",
    structure_format_score=0.2,
    final_format_score=0.1,
    retrieval_score=0,
    format_score=0,
    score=1.0,
):
    is_valid_format, _ = is_valid_sequence(solution_str)
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return structure_format_score + retrieval_score if is_valid_format else 0

    if em_check(answer, ground_truth["target"]):
        return score if is_valid_format else score - structure_format_score

    if is_valid_format:
        return structure_format_score + retrieval_score

    return final_format_score


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    """Compute PPO's EM reward with the shared DualSearch call penalty."""

    base_score = compute_score_em(solution_str, ground_truth, **kwargs)
    retrieval_count = get_retrieval_call_count(extra_info)
    return base_score - compute_retrieval_penalty(retrieval_count)
