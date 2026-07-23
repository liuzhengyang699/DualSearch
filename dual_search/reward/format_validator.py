"""DualSearch-native trajectory validation used by reward functions."""

from dual_search.protocol import extract_answer, validate_sequence


def is_valid_sequence(text: str) -> tuple[bool, str]:
    return validate_sequence(text)


def extract_solution(solution_str: str) -> str | None:
    return extract_answer(solution_str)
