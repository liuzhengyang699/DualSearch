import json

import pandas as pd
import torch

from dual_search.protocol import canonical_tool_schemas_json
from verl.utils.dataset.multiturn_sft_dataset import (
    MultiTurnSFTDataset,
    _validate_dual_search_sft_schema,
    decode_message_tool_arguments,
    normalize_arrow_value,
    optional_bool,
)


class FakeQwenTokenizer:
    pad_token_id = 0

    def _render(self, messages, tools, add_generation_prompt):
        output = ""
        if tools:
            output += "<system-tools>" + json.dumps(tools, sort_keys=True) + "</system-tools>"
        for message in messages:
            role = message["role"]
            if role == "user":
                output += "<user>" + message["content"] + "</user>"
            elif role == "assistant":
                output += "<assistant>" + message.get("content", "")
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    arguments = function["arguments"]
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                    output += (
                        '<tool_call>{"name":"'
                        + function["name"]
                        + '","arguments":'
                        + arguments
                        + "}</tool_call>"
                    )
                output += "</assistant>"
            elif role == "tool":
                output += "<tool_response>" + message["content"] + "</tool_response>"
        if add_generation_prompt:
            output += "<assistant>"
        return output

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        rendered = self._render(messages, tools, add_generation_prompt)
        ids = [ord(character) for character in rendered]
        if return_dict:
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return ids if tokenize else rendered


def _messages():
    return [
        {"role": "user", "content": "Question"},
        {
            "role": "assistant",
            "content": "<think>visual reasoning</think>",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "vision_search",
                        "arguments": '{"image_index":1,"query":"striped wings"}',
                    },
                }
            ],
        },
        {"role": "tool", "name": "vision_search", "content": "Caption 1 hidden observation"},
        {
            "role": "assistant",
            "content": "<think>text reasoning</think>",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"species habitat"}'},
                }
            ],
        },
        {"role": "tool", "name": "search", "content": "Doc 1 private evidence"},
        {"role": "assistant", "content": "<think>final reasoning</think>\n<answer>forest</answer>"},
    ]


def _dual_search_sft_row(**overrides):
    row = {
        "schema_version": 2,
        "data_source": "dual_search_sft",
        "messages": _messages(),
        "tools": canonical_tool_schemas_json(),
        "images": [{"image": __file__}],
        "sample_id": "sft:child",
        "parent_sample_id": "evqa:parent",
        "source_image_index": 2,
        "image_key": "inaturalist:query-2",
    }
    row.update(overrides)
    return row


def test_arrow_null_members_are_removed_recursively():
    value = {
        "role": "user",
        "content": "hello",
        "tool_calls": None,
        "function": {"name": None, "arguments": None},
    }
    assert normalize_arrow_value(value) == {"role": "user", "content": "hello", "function": {}}


def test_physical_argument_strings_are_decoded_for_the_qwen_template():
    messages = decode_message_tool_arguments(_messages())
    arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert arguments == {"image_index": 1, "query": "striped wings"}


def test_optional_thinking_flag_does_not_treat_false_string_as_true():
    assert optional_bool(None, field="enable_thinking") is None
    assert optional_bool("false", field="enable_thinking") is False
    assert optional_bool("true", field="enable_thinking") is True


def test_loader_passes_tools_and_masks_only_assistant_tokens(tmp_path):
    path = tmp_path / "sft.parquet"
    pd.DataFrame([{"messages": _messages(), "tools": canonical_tool_schemas_json()}]).to_parquet(path, index=False)
    tokenizer = FakeQwenTokenizer()
    dataset = MultiTurnSFTDataset(
        parquet_files=str(path),
        tokenizer=tokenizer,
        processor=None,
        config={"pad_mode": "no_padding", "max_length": 8192, "truncation": "error"},
    )
    item = dataset[0]
    decoded = "".join(chr(token) for token in item["input_ids"].tolist())
    supervised = "".join(
        chr(token)
        for token, mask in zip(item["input_ids"].tolist(), item["loss_mask"].tolist(), strict=False)
        if mask
    )

    assert "<system-tools>" in decoded
    assert "<tool_call>" in supervised
    assert "visual reasoning" in supervised
    assert "text reasoning" in supervised
    assert "<answer>forest</answer>" in supervised
    assert "Question" not in supervised
    assert "hidden observation" not in supervised
    assert "private evidence" not in supervised
    assert item["input_ids"].shape == item["position_ids"].shape == item["loss_mask"].shape


def test_loader_accepts_an_empty_validation_file_with_fixed_columns(tmp_path):
    path = tmp_path / "empty_val.parquet"
    pd.DataFrame(columns=["messages", "tools", "images", "sample_id", "image_key"]).to_parquet(
        path, index=False
    )
    dataset = MultiTurnSFTDataset(
        parquet_files=str(path),
        tokenizer=FakeQwenTokenizer(),
        processor=None,
        config={"pad_mode": "no_padding", "max_length": 8192, "truncation": "error"},
    )
    assert len(dataset) == 0


def test_loader_rejects_legacy_dual_search_sft_schema():
    legacy = pd.DataFrame([_dual_search_sft_row(schema_version=1)])

    try:
        _validate_dual_search_sft_schema(legacy)
    except ValueError as exc:
        assert "schema_version must be 2" in str(exc)
        assert "Rerun the sft stage" in str(exc)
    else:
        raise AssertionError("legacy DualSearch SFT rows must be rejected")


def test_loader_accepts_valid_v2_after_mixed_dataframe_float_promotion():
    dual_search = pd.DataFrame([_dual_search_sft_row()])
    ordinary = pd.DataFrame(
        [
            {
                "data_source": "other_sft",
                "messages": [],
                "tools": "[]",
                "images": [],
            }
        ]
    )
    mixed = pd.concat([dual_search, ordinary], ignore_index=True)

    assert float(mixed.loc[0, "schema_version"]) == 2.0
    _validate_dual_search_sft_schema(mixed)


def test_dataset_init_accepts_valid_v2_single_image_parquet(tmp_path):
    path = tmp_path / "valid_sft_v2.parquet"
    pd.DataFrame([_dual_search_sft_row()]).to_parquet(path, index=False)

    dataset = MultiTurnSFTDataset(
        parquet_files=str(path),
        tokenizer=FakeQwenTokenizer(),
        processor=None,
        config={"pad_mode": "no_padding", "max_length": 8192, "truncation": "error"},
    )

    assert len(dataset) == 1


def test_loader_rejects_malformed_v2_single_image_child():
    malformed = pd.DataFrame(
        [
            _dual_search_sft_row(
                parent_sample_id="",
                images=[
                    {"image": "/tmp/first.jpg"},
                    {"image": "/tmp/second.jpg"},
                ],
            )
        ]
    )

    try:
        _validate_dual_search_sft_schema(malformed)
    except ValueError as exc:
        assert "Malformed DualSearch SFT schema v2" in str(exc)
        assert "Rerun the sft stage" in str(exc)
    else:
        raise AssertionError("malformed schema v2 SFT rows must be rejected")


def test_dataset_init_rejects_v1_before_rendering(tmp_path):
    path = tmp_path / "legacy_sft.parquet"
    pd.DataFrame([_dual_search_sft_row(schema_version=1)]).to_parquet(path, index=False)

    try:
        MultiTurnSFTDataset(
            parquet_files=str(path),
            tokenizer=FakeQwenTokenizer(),
            processor=None,
            config={"pad_mode": "no_padding", "max_length": 8192, "truncation": "error"},
        )
    except ValueError as exc:
        assert "schema_version must be 2" in str(exc)
        assert "Rerun the sft stage" in str(exc)
    else:
        raise AssertionError("MultiTurnSFTDataset must reject v1 before rendering")
