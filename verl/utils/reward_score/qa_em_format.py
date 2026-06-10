import random
import re
import string


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


def _assistant_content(text: str) -> str:
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    if assistant_match:
        return text[assistant_match.end() :]
    return text


def is_valid_sequence(text):
    content = _assistant_content(text)

    tags_to_check = ["think", "search", "information", "vision_search", "vision_information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|search|information|vision_search|vision_information|answer)>)"
    parts = re.split(split_pattern, content)
    state = "start"

    for part in parts:
        if not part.strip():
            continue

        if re.match(r"</?(?:think|search|information|vision_search|vision_information|answer)>", part):
            if part == "<think>" and state in ["start", "information", "vision_information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<vision_search>" and state == "after_think":
                state = "in_vision_search"
            elif part == "</vision_search>" and state == "in_vision_search":
                state = "after_vision_search"
            elif part == "<vision_information>" and state == "after_vision_search":
                state = "in_vision_information"
            elif part == "</vision_information>" and state == "in_vision_information":
                state = "vision_information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        elif state in [
            "in_think",
            "in_search",
            "in_information",
            "in_vision_search",
            "in_vision_information",
            "in_answer",
        ]:
            continue
        elif state in ["start", "after_think", "after_search", "information", "after_vision_search", "vision_information"]:
            return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
        else:
            return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


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


def compute_score(solution_str, ground_truth, **kwargs):
    return compute_score_em(solution_str, ground_truth, **kwargs)
