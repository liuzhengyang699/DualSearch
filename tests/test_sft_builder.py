import base64
import json
from pathlib import Path

import pandas as pd

from dual_search.data.sft_builder import (
    SFTBuilderConfig,
    TeacherConfig,
    TrajectoryBuildError,
    VLLMTeacherClient,
    _check_early_leak,
    _sft_child_id,
    _stable_key,
    build_sft_files,
    build_sft_records,
    is_multi_hop_record,
    truncate_tool_observation,
)
from dual_search.protocol import format_text_results, format_vision_results


class FakeTeacher:
    def __init__(
        self,
        fail_sample_ids=None,
        leak=False,
        final_answer="not checked against gold",
        search_query="Species Alpha habitat facts",
    ):
        self.fail_sample_ids = set(fail_sample_ids or [])
        self.leak = leak
        self.final_answer = final_answer
        self.search_query = search_query
        self.calls = []

    def generate(self, *, stage, messages, response_schema):
        text = messages[-1]["content"][-1]["text"]
        self.calls.append({"stage": stage, "messages": messages, "response_schema": response_schema})
        sample_id = next(
            sample_id for sample_id in self.fail_sample_ids if f'"sample_id": "{sample_id}"' in text
        ) if any(f'"sample_id": "{sample_id}"' in text for sample_id in self.fail_sample_ids) else None
        if sample_id is not None and stage == "vision":
            return {"unexpected": "schema failure"}
        if stage == "vision":
            think = "SecretAnswer is likely" if self.leak else "Compare visible markings and body shape"
            return {"think": think, "vision_search": {"query": "striped wings and narrow bill"}}
        if stage == "search":
            return {
                "think": "The visual result names the subject, so retrieve the requested fact",
                "search": {"query": self.search_query},
            }
        return {"think": "The retrieved evidence now supports a concise response", "answer": self.final_answer}


def _record(
    sample_id,
    image_key,
    *,
    question_type="single_hop",
    resolvable=True,
    image_paths=None,
):
    image_keys = [image_key] if isinstance(image_key, str) else list(image_key)
    image_paths = list(image_paths) if image_paths is not None else [__file__] * len(image_keys)
    assert len(image_paths) == len(image_keys)
    query_images = [
        {
            "image_index": index,
            "dataset_image_id": f"image-{index}-{key}",
            "image_key": key,
            "image": str(image_paths[index - 1]),
            "source_file_name": f"train/{index}.jpg",
            "source_split": "train",
        }
        for index, key in enumerate(image_keys, start=1)
    ]
    prompt_images = "\n".join(f"Image {index}:\n<image>" for index in range(1, len(image_keys) + 1))
    return {
        "schema_version": 2,
        "data_source": "dual_search",
        "sample_id": sample_id,
        "query_images": query_images,
        "image_keys": image_keys,
        "image_count": len(image_keys),
        "category_key": "inaturalist:cat-a",
        "dataset_category_id": "cat-a",
        "question": "What habitat does the pictured animal prefer?",
        "question_type": question_type,
        "retrieval_resolvable": resolvable,
        "answer": "SecretAnswer",
        "wikipedia_url": ["https://example.test/species-alpha"],
        "wikipedia_title": ["Species Alpha"],
        "evidence_section_id": ["alpha-habitat"],
        "evidence": "It lives in cloud forests.",
        "prompt": [{"role": "user", "content": f"{prompt_images}\nQuestion: What habitat does it prefer?"}],
        "images": [{"image": item["image"]} for item in query_images],
    }


def _record_image_keys(record):
    return set(record["image_keys"])


def _vision_corpus():
    return [
        {
            "id": "vision-positive",
            "image_key": "inaturalist:corpus-a",
            "category_key": "inaturalist:cat-a",
            "contents": "Species Alpha\nA narrow-billed animal with striped wings.",
        },
        {
            "id": "vision-negative-b",
            "image_key": "inaturalist:corpus-b",
            "category_key": "inaturalist:cat-b",
            "contents": "Species Beta\nA broad-winged animal.",
        },
        {
            "id": "vision-negative-c",
            "image_key": "inaturalist:corpus-c",
            "category_key": "inaturalist:cat-c",
            "contents": "Species Gamma\nA spotted animal.",
        },
    ]


def _text_corpus():
    return [
        {
            "id": "alpha-habitat",
            "section_id": "alpha-habitat",
            "url": "https://example.test/species-alpha",
            "title": "Species Alpha",
            "contents": "Species Alpha — Habitat\nIt lives in cloud forests.",
        },
        {
            "id": "beta-habitat",
            "section_id": "beta-habitat",
            "url": "https://example.test/species-beta",
            "title": "Species Beta",
            "contents": "Species Beta — Habitat\nIt lives in grasslands.",
        },
        {
            "id": "gamma-diet",
            "section_id": "gamma-diet",
            "url": "https://example.test/species-gamma",
            "title": "Species Gamma",
            "contents": "Species Gamma — Diet\nIt eats seeds.",
        },
    ]


def test_failed_teacher_sample_is_deterministically_supplemented():
    records = [_record("sample-a", "inaturalist:query-a"), _record("sample-b", "inaturalist:query-b")]
    ordered = sorted(
        records,
        key=lambda row: _stable_key(7, "sft_candidate_group", "single_hop", row["image_keys"][0]),
    )
    failed_child_id = _sft_child_id(ordered[0]["sample_id"], ordered[0]["image_keys"][0])
    teacher = FakeTeacher(fail_sample_ids={failed_child_id})
    result = build_sft_records(
        records,
        _vision_corpus(),
        _text_corpus(),
        set().union(*(_record_image_keys(row) for row in records)),
        teacher,
        SFTBuilderConfig(sample_fraction=0.05, validation_fraction=0.1, seed=7),
    )

    assert result.report["sampling"]["targets_by_question_type"] == {"single_hop": 1}
    assert result.report["sampling"]["attempted_by_question_type"] == {"single_hop": 2}
    assert result.report["sampling"]["successful_by_question_type"] == {"single_hop": 1}
    assert result.report["failures"]["by_stage"] == {"vision": 1}
    assert len(result.train_rows) == 1
    row = result.train_rows[0]
    assert json.loads(row["tools"])[0]["function"]["name"] == "search"
    arguments = row["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"image_index": 1, "query": "striped wings and narrow bill"}
    assert row["messages"][2]["role"] == "tool"
    assert "Caption " in row["messages"][2]["content"]
    assert row["messages"][-1]["content"].endswith("<answer>not checked against gold</answer>")


def test_teacher_requests_carry_image_planning_gold_and_structured_tool_history():
    record = _record("teacher-history", "inaturalist:query-history")
    teacher = FakeTeacher(search_query="pictured animal habitat facts")
    result = build_sft_records(
        [record],
        _vision_corpus(),
        _text_corpus(),
        _record_image_keys(record),
        teacher,
        SFTBuilderConfig(sample_fraction=1),
    )
    assert [call["stage"] for call in teacher.calls] == ["vision", "search", "answer"]

    for call in teacher.calls:
        messages = call["messages"]
        assert messages[0]["role"] == "system"
        assert "planning" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        image_items = [item for item in messages[1]["content"] if item.get("type") == "image_url"]
        assert len(image_items) == 1
        assert image_items[0]["image_url"]["url"].startswith("data:")
        planning_text = "\n".join(
            item.get("text", "") for item in messages[1]["content"] if item.get("type") == "text"
        )
        assert "Official answers for planning only" in planning_text
        assert "SecretAnswer" in planning_text
        # Gold is not fabricated into an assistant action or tool result.
        assert all(
            "SecretAnswer" not in json.dumps(message, ensure_ascii=False)
            for message in messages
            if message["role"] in {"assistant", "tool"}
        )

    vision_messages = teacher.calls[0]["messages"]
    assert [message["role"] for message in vision_messages] == ["system", "user"]

    search_messages = teacher.calls[1]["messages"]
    assert [message["role"] for message in search_messages] == ["system", "user", "assistant", "tool", "user"]
    vision_call = search_messages[2]["tool_calls"][0]
    assert vision_call["type"] == "function"
    assert vision_call["id"] == search_messages[3]["tool_call_id"]
    assert vision_call["function"]["name"] == "vision_search"
    assert json.loads(vision_call["function"]["arguments"]) == {
        "image_index": 1,
        "query": "striped wings and narrow bill",
    }
    assert search_messages[3]["role"] == "tool"
    assert "Caption " in search_messages[3]["content"]

    answer_messages = teacher.calls[2]["messages"]
    assert [message["role"] for message in answer_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "user",
    ]
    text_call = answer_messages[4]["tool_calls"][0]
    assert text_call["id"] == answer_messages[5]["tool_call_id"]
    assert text_call["function"]["name"] == "search"
    assert json.loads(text_call["function"]["arguments"]) == {
        "query": "pictured animal habitat facts"
    }
    assert "Doc " in answer_messages[5]["content"]

    # The student SFT trajectory never receives the planning-only gold field.
    student_messages = result.train_rows[0]["messages"]
    assert "SecretAnswer" not in json.dumps(student_messages, ensure_ascii=False)


def test_multi_image_parent_expands_to_stable_single_image_children(tmp_path):
    image_paths = [tmp_path / f"query-{index}.jpg" for index in range(1, 4)]
    for index, path in enumerate(image_paths, start=1):
        path.write_bytes(f"query-image-{index}".encode())
    parent = _record(
        "parent-multi",
        [
            "inaturalist:query-multi-a",
            "inaturalist:query-multi-b",
            "inaturalist:query-multi-c",
        ],
        question_type="visual_attribute",
        image_paths=image_paths,
    )
    heldout = _record_image_keys(parent)
    first_teacher = FakeTeacher(search_query="pictured animal habitat facts")
    first = build_sft_records(
        [parent],
        _vision_corpus(),
        _text_corpus(),
        heldout,
        first_teacher,
        SFTBuilderConfig(sample_fraction=1, validation_fraction=0, seed=3),
    )
    second = build_sft_records(
        [parent],
        _vision_corpus(),
        _text_corpus(),
        heldout,
        FakeTeacher(search_query="pictured animal habitat facts"),
        SFTBuilderConfig(sample_fraction=1, validation_fraction=0, seed=999),
    )

    assert len(first.train_rows) == 3
    assert len({row["sample_id"] for row in first.train_rows}) == 3
    assert {row["sample_id"] for row in first.train_rows} == {
        row["sample_id"] for row in second.train_rows
    }
    assert {row["parent_sample_id"] for row in first.train_rows} == {"parent-multi"}
    assert {row["source_image_index"] for row in first.train_rows} == {1, 2, 3}
    assert {row["image_key"] for row in first.train_rows} == heldout
    assert {
        (row["source_image_index"], row["images"][0]["image"])
        for row in first.train_rows
    } == {
        (index, str(path))
        for index, path in enumerate(image_paths, start=1)
    }
    teacher_image_payloads = []
    for call in first_teacher.calls:
        image_item = next(
            item
            for message in call["messages"]
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if item.get("type") == "image_url"
        )
        data_url = image_item["image_url"]["url"]
        teacher_image_payloads.append(base64.b64decode(data_url.split(",", 1)[1]))
    assert {
        payload for payload in teacher_image_payloads
    } == {
        path.read_bytes() for path in image_paths
    }
    assert len(teacher_image_payloads) == 3 * len(image_paths)
    for row in first.train_rows:
        assert row["schema_version"] == 2
        assert len(row["images"]) == 1
        assert row["messages"][0]["content"].count("<image>") == 1
        assert "Image 2:" not in row["messages"][0]["content"]
        arguments = json.loads(row["messages"][1]["tool_calls"][0]["function"]["arguments"])
        assert arguments["image_index"] == 1
        assert row["extra_info"]["parent_sample_id"] == "parent-multi"
        assert row["extra_info"]["source_image_index"] == row["source_image_index"]

    assert first.report["eligibility"]["eligible_parent_samples"] == 1
    assert first.report["eligibility"]["expanded_single_image_candidates"] == 3
    assert first.report["sampling"]["targets_by_question_type"] == {"visual_attribute": 3}
    assert "image_index" not in first.report["config"]


def test_sampling_fraction_uses_expanded_single_image_candidate_count():
    image_keys = [f"inaturalist:expanded-{index}" for index in range(20)]
    parent = _record("parent-expanded", image_keys, question_type="visual_attribute")
    result = build_sft_records(
        [parent],
        _vision_corpus(),
        _text_corpus(),
        set(image_keys),
        FakeTeacher(search_query="pictured animal habitat facts"),
        SFTBuilderConfig(sample_fraction=0.05, validation_fraction=0, seed=23),
    )

    assert result.report["eligibility"]["eligible_parent_samples"] == 1
    assert result.report["eligibility"]["expanded_single_image_candidates"] == 20
    assert result.report["sampling"]["targets_by_question_type"] == {
        "visual_attribute": 1
    }
    assert result.report["sampling"]["successful_by_question_type"] == {
        "visual_attribute": 1
    }
    assert len(result.train_rows) == 1


def test_multi_image_parent_rejects_misalignment_duplicates_and_missing_heldout():
    aligned = _record(
        "parent-aligned",
        ["inaturalist:aligned-a", "inaturalist:aligned-b"],
    )
    misaligned = dict(aligned)
    misaligned["image_keys"] = list(reversed(aligned["image_keys"]))
    try:
        build_sft_records(
            [misaligned],
            _vision_corpus(),
            _text_corpus(),
            _record_image_keys(aligned),
            FakeTeacher(),
        )
    except ValueError as exc:
        assert "inconsistent index, identity, path, or duplicate image_key" in str(exc)
    else:
        raise AssertionError("misaligned v2 image lists must be rejected")

    duplicate = _record(
        "parent-duplicate",
        ["inaturalist:duplicate", "inaturalist:duplicate"],
    )
    try:
        build_sft_records(
            [duplicate],
            _vision_corpus(),
            _text_corpus(),
            {"inaturalist:duplicate"},
            FakeTeacher(),
        )
    except ValueError as exc:
        assert "duplicate image_key" in str(exc)
    else:
        raise AssertionError("duplicate v2 image keys must be rejected")

    try:
        build_sft_records(
            [aligned],
            _vision_corpus(),
            _text_corpus(),
            {"inaturalist:aligned-a"},
            FakeTeacher(),
        )
    except ValueError as exc:
        assert "absent from the heldout manifest" in str(exc)
        assert "inaturalist:aligned-b" in str(exc)
    else:
        raise AssertionError("every SFT child image must be present in heldout")


def test_two_hop_and_unresolvable_rows_are_excluded_and_early_leak_is_rejected():
    records = [
        _record(
            "two-hop",
            ["inaturalist:query-two-a", "inaturalist:query-two-b"],
            question_type="2_hop",
        ),
        _record(
            "unresolvable",
            ["inaturalist:query-unresolvable-a", "inaturalist:query-unresolvable-b"],
            resolvable=False,
        ),
        _record("leaky", "inaturalist:query-leaky"),
    ]
    result = build_sft_records(
        records,
        _vision_corpus(),
        _text_corpus(),
        set().union(*(_record_image_keys(row) for row in records)),
        FakeTeacher(leak=True),
        SFTBuilderConfig(sample_fraction=1, seed=3),
    )

    assert result.train_rows == []
    assert result.report["eligibility"]["excluded"] == {
        "retrieval_unresolvable": 1,
        "two_hop": 1,
    }
    assert result.report["failures"]["by_stage"] == {"vision": 1}
    assert any("early leakage" in reason for reason in result.report["failures"]["by_reason"])


def test_multi_hop_detection_supports_hop_type_and_multiple_page_schemas():
    single = _record("single", "inaturalist:query-single", question_type="visual_attribute")
    assert not is_multi_hop_record(single)

    explicit = dict(single, sample_id="explicit", hop_type="two-hop")
    assert is_multi_hop_record(explicit)
    assert is_multi_hop_record(dict(single, sample_id="numeric-hop", hop_type=2.0))

    multiple_urls = dict(
        single,
        sample_id="urls",
        wikipedia_url=["https://example.test/page-a", "https://example.test/page-b"],
    )
    assert is_multi_hop_record(multiple_urls)

    multiple_pages = dict(
        single,
        sample_id="pages",
        wikipedia_url=[],
        wikipedia_pages=["Page A", "Page B"],
    )
    assert is_multi_hop_record(multiple_pages)

    result = build_sft_records(
        [explicit, multiple_urls, multiple_pages],
        _vision_corpus(),
        _text_corpus(),
        set().union(*(_record_image_keys(row) for row in [explicit, multiple_urls, multiple_pages])),
        FakeTeacher(search_query="pictured animal habitat facts"),
        SFTBuilderConfig(sample_fraction=1),
    )
    assert result.report["eligibility"]["excluded"] == {"two_hop": 3}
    assert result.report["eligibility"]["eligible_by_question_type"] == {}


def test_early_leak_matching_uses_token_boundaries_for_short_answers():
    _check_early_leak(
        stage="vision",
        generated=["Compare a redwood canopy", "redwood habitat"],
        protected_phrases=["red"],
        visible_context="What tree is shown?",
    )
    try:
        _check_early_leak(
            stage="vision",
            generated=["The answer is red", "color query"],
            protected_phrases=["red"],
            visible_context="What color is shown?",
        )
    except TrajectoryBuildError as exc:
        assert "early leakage" in exc.reason
    else:
        raise AssertionError("an exact protected token should be rejected")


class CharToolTokenizer:
    system = "<system>"
    prefix = "<tool_response>"
    suffix = "</tool_response>"
    generation = "<assistant>"
    user = "<user></user>"

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
    ):
        rendered = self.system
        for message in messages:
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    content = "".join(item.get("text", "") for item in content)
                rendered += "<user>" + str(content) + "</user>"
            elif message["role"] == "tool":
                rendered += self.prefix + message["content"] + self.suffix
        if add_generation_prompt:
            rendered += self.generation
        return self.encode(rendered) if tokenize else rendered


class UserRequiredCharToolTokenizer(CharToolTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if messages and messages[0]["role"] == "tool":
            raise ValueError("a user turn is required")
        return super().apply_chat_template(messages, **kwargs)


def test_observation_truncation_matches_native_wrapper_budget():
    tokenizer = CharToolTokenizer()
    content = "0123456789" * 20
    truncated = truncate_tool_observation(content, tokenizer=tokenizer, max_tokens=64)
    rendered = tokenizer.apply_chat_template(
        [{"role": "tool", "content": truncated}], add_generation_prompt=True
    )
    runtime_visible = rendered[len(tokenizer.system) :]
    assert len(runtime_visible) <= 64
    assert len(rendered) == 64 + len(tokenizer.system)
    assert len(truncated) == 64 - len(tokenizer.prefix) - len(tokenizer.suffix) - len(tokenizer.generation)

    # Qwen variants that reject a tool-only conversation take veRL's dummy
    # user fallback; the builder must mirror both fallback and subsequent
    # remove_system_prompt slicing.
    fallback_tokenizer = UserRequiredCharToolTokenizer()
    fallback = truncate_tool_observation(content, tokenizer=fallback_tokenizer, max_tokens=64)
    dummy = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    dummy_prefix = fallback_tokenizer.apply_chat_template(dummy, add_generation_prompt=False)
    with_tool = fallback_tokenizer.apply_chat_template(
        dummy + [{"role": "tool", "content": fallback}], add_generation_prompt=True
    )
    fallback_visible = with_tool[len(dummy_prefix) + len(fallback_tokenizer.system) :]
    assert len(fallback_visible) <= 64
    assert fallback_tokenizer.generation in fallback_tokenizer.decode(fallback_visible)

    conservative = truncate_tool_observation(
        "鸟" * 100,
        tokenizer=None,
        max_tokens=64,
        fallback_wrapper_token_reserve=16,
    )
    assert len(conservative.encode("utf-8")) <= 48
    assert conservative


def test_builder_observations_use_same_topk_format_then_token_truncation():
    tokenizer = CharToolTokenizer()
    record = _record("sample-format", "inaturalist:query-format")
    vision_corpus = _vision_corpus()
    text_corpus = _text_corpus()
    for document in vision_corpus + text_corpus:
        document["contents"] += " " + ("long evidence " * 50)
    result = build_sft_records(
        [record],
        vision_corpus,
        text_corpus,
        _record_image_keys(record),
        FakeTeacher(search_query="pictured animal habitat facts"),
        SFTBuilderConfig(sample_fraction=1, max_tool_response_tokens=120),
        observation_tokenizer=tokenizer,
    )
    row = result.train_rows[0]
    by_vision_id = {document["id"]: document for document in vision_corpus}
    by_text_id = {document["id"]: document for document in text_corpus}
    expected_vision = truncate_tool_observation(
        "image_index=1:\n"
        + format_vision_results(
            [{"document": by_vision_id[doc_id]} for doc_id in row["extra_info"]["oracle_vision_ids"]]
        ),
        tokenizer=tokenizer,
        max_tokens=120,
    )
    expected_text = truncate_tool_observation(
        format_text_results(
            [{"document": by_text_id[doc_id]} for doc_id in row["extra_info"]["oracle_text_ids"]]
        ),
        tokenizer=tokenizer,
        max_tokens=120,
    )
    assert len(row["extra_info"]["oracle_vision_ids"]) == 3
    assert len(row["extra_info"]["oracle_text_ids"]) == 3
    assert row["messages"][2]["content"] == expected_vision
    assert row["messages"][4]["content"] == expected_text
    assert result.report["observation_truncation"]["mode"] == "tokenizer_exact"


def test_successful_rows_are_split_by_image_group():
    records = [_record(f"sample-{index}", f"inaturalist:query-{index}") for index in range(12)]
    result = build_sft_records(
        records,
        _vision_corpus(),
        _text_corpus(),
        set().union(*(_record_image_keys(row) for row in records)),
        FakeTeacher(),
        SFTBuilderConfig(sample_fraction=1, validation_fraction=0.1, seed=11),
    )
    train_images = {row["image_key"] for row in result.train_rows}
    val_images = {row["image_key"] for row in result.val_rows}
    assert val_images
    assert train_images.isdisjoint(val_images)
    assert len(result.train_rows) + len(result.val_rows) == 12


def test_validation_split_uses_connected_image_and_parent_groups():
    records = [
        _record("parent-a", ["inaturalist:shared", "inaturalist:a"]),
        _record("parent-b", ["inaturalist:shared", "inaturalist:b"]),
        _record("parent-c", "inaturalist:c"),
        _record("parent-d", "inaturalist:d"),
    ]
    result = build_sft_records(
        records,
        _vision_corpus(),
        _text_corpus(),
        set().union(*(_record_image_keys(row) for row in records)),
        FakeTeacher(),
        SFTBuilderConfig(sample_fraction=1, validation_fraction=0.25, seed=17),
    )
    train_images = {row["image_key"] for row in result.train_rows}
    val_images = {row["image_key"] for row in result.val_rows}
    train_parents = {row["parent_sample_id"] for row in result.train_rows}
    val_parents = {row["parent_sample_id"] for row in result.val_rows}
    assert result.val_rows
    assert train_images.isdisjoint(val_images)
    assert train_parents.isdisjoint(val_parents)
    assert ({"parent-a", "parent-b"} <= train_parents) or ({"parent-a", "parent-b"} <= val_parents)
    assert result.report["output"]["image_group_overlap"] == 0
    assert result.report["output"]["parent_group_overlap"] == 0


def test_schema_v1_rl_rows_are_rejected_with_rebuild_instruction():
    record = _record("legacy", "inaturalist:legacy")
    record["schema_version"] = 1
    try:
        build_sft_records(
            [record],
            _vision_corpus(),
            _text_corpus(),
            _record_image_keys(record),
            FakeTeacher(),
        )
    except ValueError as exc:
        message = str(exc)
        assert "schema v1" in message
        assert "rerun split, corpus, and sft" in message
    else:
        raise AssertionError("schema v1 RL data must be rejected")


def test_heldout_image_in_visual_corpus_is_a_hard_error():
    records = [_record("sample-a", "inaturalist:query-a")]
    corpus = _vision_corpus() + [
        {
            "id": "leak",
            "image_key": "inaturalist:query-a",
            "category_key": "inaturalist:cat-a",
            "contents": "Query image\nDo not use.",
        }
    ]
    try:
        build_sft_records(
            records,
            corpus,
            _text_corpus(),
            {"inaturalist:query-a"},
            FakeTeacher(),
        )
    except ValueError as exc:
        assert "held-out query images" in str(exc)
    else:
        raise AssertionError("expected heldout corpus leak to fail")


def test_oracle_text_requires_the_exact_evidence_section():
    record = _record("sample-exact", "inaturalist:query-exact")
    wrong_section = {
        **_text_corpus()[0],
        "id": "alpha-diet",
        "section_id": "alpha-diet",
        "contents": "Species Alpha — Diet\nIt eats seeds.",
    }
    result = build_sft_records(
        [record],
        _vision_corpus(),
        [wrong_section, *_text_corpus()[1:]],
        _record_image_keys(record),
        FakeTeacher(),
        SFTBuilderConfig(sample_fraction=1),
    )
    assert result.train_rows == []
    assert result.report["failures"]["by_stage"] == {"oracle_text": 1}
    assert any("no matching evidence section" in reason for reason in result.report["failures"]["by_reason"])


def test_parquet_roundtrip_preserves_json_arguments_and_group_split(tmp_path):
    records = [_record("sample-a", "inaturalist:query-a"), _record("sample-b", "inaturalist:query-b")]
    train_path = tmp_path / "train.parquet"
    pd.DataFrame(records).to_parquet(train_path, index=False)
    vision_path = tmp_path / "vision_corpus.jsonl"
    vision_path.write_text("".join(json.dumps(row) + "\n" for row in _vision_corpus()), encoding="utf-8")
    text_path = tmp_path / "text_corpus.jsonl"
    text_path.write_text("".join(json.dumps(row) + "\n" for row in _text_corpus()), encoding="utf-8")
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "heldout": {
                    "images": [
                        {"image_key": image_key}
                        for row in records
                        for image_key in row["image_keys"]
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = build_sft_files(
        train_parquet=train_path,
        vision_corpus_path=vision_path,
        text_corpus_path=text_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "output",
        teacher=FakeTeacher(),
        config=SFTBuilderConfig(sample_fraction=1, validation_fraction=0.5, seed=19),
    )
    assert len(result.train_rows) == 1
    assert len(result.val_rows) == 1
    reloaded = pd.concat(
        [
            pd.read_parquet(tmp_path / "output" / "sft_train.parquet"),
            pd.read_parquet(tmp_path / "output" / "sft_val.parquet"),
        ]
    )
    assert len(reloaded) == 2
    physical_messages = reloaded.iloc[0]["messages"]
    argument_strings = []
    for message in physical_messages:
        for tool_call in message.get("tool_calls") or []:
            argument_strings.append(tool_call["function"]["arguments"])
    assert all(isinstance(value, str) and isinstance(json.loads(value), dict) for value in argument_strings)


def test_single_image_group_writes_a_loadable_empty_validation_schema(tmp_path):
    record = _record("sample-only", "inaturalist:query-only")
    train_path = tmp_path / "train.parquet"
    pd.DataFrame([record]).to_parquet(train_path, index=False)
    vision_path = tmp_path / "vision_corpus.jsonl"
    vision_path.write_text("".join(json.dumps(row) + "\n" for row in _vision_corpus()), encoding="utf-8")
    text_path = tmp_path / "text_corpus.jsonl"
    text_path.write_text("".join(json.dumps(row) + "\n" for row in _text_corpus()), encoding="utf-8")
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "heldout": {"heldout_image_keys": record["image_keys"]},
            }
        ),
        encoding="utf-8",
    )

    build_sft_files(
        train_parquet=train_path,
        vision_corpus_path=vision_path,
        text_corpus_path=text_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "output",
        teacher=FakeTeacher(),
        config=SFTBuilderConfig(sample_fraction=1, validation_fraction=0.1),
    )

    empty_val = pd.read_parquet(tmp_path / "output" / "sft_val.parquet")
    assert len(empty_val) == 0
    assert {
        "schema_version",
        "messages",
        "tools",
        "images",
        "sample_id",
        "parent_sample_id",
        "source_image_index",
        "image_key",
    }.issubset(empty_val.columns)


def test_schema_v1_manifest_is_rejected_before_sft_output(tmp_path):
    record = _record("sample-legacy-manifest", "inaturalist:query-legacy-manifest")
    train_path = tmp_path / "train.parquet"
    pd.DataFrame([record]).to_parquet(train_path, index=False)
    vision_path = tmp_path / "vision_corpus.jsonl"
    vision_path.write_text("".join(json.dumps(row) + "\n" for row in _vision_corpus()), encoding="utf-8")
    text_path = tmp_path / "text_corpus.jsonl"
    text_path.write_text("".join(json.dumps(row) + "\n" for row in _text_corpus()), encoding="utf-8")
    manifest_path = tmp_path / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "heldout": {"heldout_image_keys": record["image_keys"]},
            }
        ),
        encoding="utf-8",
    )

    try:
        build_sft_files(
            train_parquet=train_path,
            vision_corpus_path=vision_path,
            text_corpus_path=text_path,
            manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            teacher=FakeTeacher(),
        )
    except ValueError as exc:
        assert "schema v1" in str(exc)
        assert "rerun split, corpus, and sft" in str(exc)
    else:
        raise AssertionError("schema v1 manifest must be rejected")
    assert not (tmp_path / "output" / "sft_train.parquet").exists()


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {"message": {"content": json.dumps({"think": "x", "answer": "y"})}}
            ]
        }


class _Session:
    def __init__(self):
        self.urls = []

    def post(self, url, **kwargs):
        self.urls.append(url)
        return _Response()


def test_teacher_base_url_accepts_v1_suffix():
    session = _Session()
    client = VLLMTeacherClient(TeacherConfig(base_url="http://localhost:8000/v1", model="teacher"), session=session)
    output = client.generate(stage="answer", messages=[], response_schema={"type": "object"})
    assert output == {"think": "x", "answer": "y"}
    assert session.urls == ["http://localhost:8000/v1/chat/completions"]
    assert client.config.temperature == 0.0
