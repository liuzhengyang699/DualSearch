# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Multi-turn multimodal SFT dataset with native tool-call supervision."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.chat_template import apply_chat_template
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer import hf_tokenizer, normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DUAL_SEARCH_SFT_DATA_SOURCE = "dual_search_sft"
DUAL_SEARCH_SFT_SCHEMA_VERSION = 2


def _integer_value(value: Any) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    return None


def _validate_dual_search_sft_schema(
    dataframe: pd.DataFrame,
    *,
    messages_column: str = "messages",
    image_column: str = "images",
    tools_column: str = "tools",
) -> None:
    """Reject legacy DualSearch SFT rows before rendering their conversations."""

    if dataframe.empty or "data_source" not in dataframe.columns:
        return
    dual_search_mask = dataframe["data_source"] == DUAL_SEARCH_SFT_DATA_SOURCE
    if not bool(dual_search_mask.any()):
        return
    required_columns = {
        "schema_version",
        "sample_id",
        "parent_sample_id",
        "source_image_index",
        "image_key",
        image_column,
        messages_column,
        tools_column,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Legacy DualSearch SFT data is incompatible with schema v2; "
            f"missing columns {missing_columns}. Rerun the sft stage."
        )

    for index, row in dataframe.loc[dual_search_mask].iterrows():
        if _integer_value(row["schema_version"]) != DUAL_SEARCH_SFT_SCHEMA_VERSION:
            raise ValueError(
                "Legacy DualSearch SFT data is incompatible with schema v2; "
                f"schema_version must be 2 (invalid row index: {index}). "
                "Rerun the sft stage."
            )
        sample_id = row["sample_id"]
        parent_sample_id = row["parent_sample_id"]
        image_key = row["image_key"]
        source_image_index = _integer_value(row["source_image_index"])
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or not isinstance(parent_sample_id, str)
            or not parent_sample_id.strip()
            or not isinstance(image_key, str)
            or not image_key.strip()
            or source_image_index is None
            or source_image_index <= 0
        ):
            raise ValueError(
                f"Malformed DualSearch SFT schema v2 row {index}: sample_id, "
                "parent_sample_id, image_key, and positive source_image_index are required. "
                "Rerun the sft stage."
            )

        images = normalize_arrow_value(row[image_column])
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(
                f"Malformed DualSearch SFT schema v2 row {index}: exactly one image is required. "
                "Rerun the sft stage."
            )
        physical_image = images[0]
        if isinstance(physical_image, dict):
            physical_image = physical_image.get("image")
        if not isinstance(physical_image, str) or not physical_image.strip():
            raise ValueError(
                f"Malformed DualSearch SFT schema v2 row {index}: the child image path is empty. "
                "Rerun the sft stage."
            )

        try:
            messages = decode_message_tool_arguments(row[messages_column])
            tools = decode_tools_column(row[tools_column])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed DualSearch SFT schema v2 row {index}: {exc}. "
                "Rerun the sft stage."
            ) from exc
        if not messages or not tools:
            raise ValueError(
                f"Malformed DualSearch SFT schema v2 row {index}: messages and tools "
                "must be non-empty. Rerun the sft stage."
            )


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def normalize_arrow_value(value: Any) -> Any:
    """Remove Arrow-injected null struct members recursively.

    PyArrow unifies structs across messages in a list.  A ``tool_calls`` field
    that only belongs to assistant messages therefore reappears as ``None`` on
    user/tool messages; Qwen's Jinja template treats presence and truthiness
    differently in a few releases, so omit those members before rendering.
    """

    value = convert_nested_value_to_list_recursive(value)
    if _is_null(value):
        return None
    if isinstance(value, dict):
        return {
            str(key): normalized
            for key, item in value.items()
            if (normalized := normalize_arrow_value(item)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [normalize_arrow_value(item) for item in value if not _is_null(item)]
    return value


def decode_tools_column(value: Any) -> list[dict[str, Any]] | None:
    value = normalize_arrow_value(value)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("SFT tools column must be valid JSON") from exc
    if not isinstance(value, list) or not all(isinstance(tool, dict) for tool in value):
        raise ValueError("SFT tools column must decode to a list of schema objects")
    return value


def decode_message_tool_arguments(messages: Any) -> list[dict[str, Any]]:
    """Decode physical JSON strings before invoking the chat template.

    ``function.arguments`` is deliberately stored as a canonical JSON string
    in Parquet so Arrow never has to merge different argument-object schemas.
    The in-memory Qwen message representation should still be the structured
    object expected by the native template.
    """

    messages = normalize_arrow_value(messages)
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be an object")
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                raise TypeError("tool_calls entries must be objects")
            function = tool_call.get("function")
            if not isinstance(function, dict):
                raise TypeError("tool call function must be an object")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("SFT function.arguments must be valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError("SFT function.arguments must decode to a JSON object")
            function["arguments"] = arguments
    return messages


def optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None or _is_null(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field} must be a boolean or null, got {value!r}")


def _token_ids(rendered: Any) -> list[int]:
    return normalize_token_ids(rendered)


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    index = 0
    for left_id, right_id in zip(left, right, strict=False):
        if left_id != right_id:
            break
        index += 1
    return index


class MultiTurnSFTDataset(Dataset):
    """Render full Qwen conversations and supervise assistant-only tokens.

    ``messages`` contains structured assistant ``tool_calls`` and role=tool
    observations.  ``tools`` may be a JSON string in Parquet; it is decoded
    immediately before being passed to the processor's chat template.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer | str,
        config: DictConfig | dict,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        config = config or {}
        self.pad_mode = DatasetPadMode(config.get("pad_mode", "right"))
        if self.pad_mode not in {DatasetPadMode.RIGHT, DatasetPadMode.NO_PADDING}:
            raise ValueError("MultiTurnSFTDataset supports pad_mode=right or no_padding")
        self.truncation = config.get("truncation", "error")
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError("truncation must be error, left, or right")
        self.max_length = int(config.get("max_length", 8192))
        self.messages_key = config.get("messages_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = dict(config.get("apply_chat_template_kwargs", {}))
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed", None)
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = bool(config.get("ignore_input_ids_mismatch", False))
        self.image_patch_size = config.get(
            "image_patch_size", getattr(getattr(processor, "image_processor", None), "patch_size", 14)
        )

        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = [str(path) for path in parquet_files]
        self.tokenizer = hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.processor = processor
        self._download()
        self._read_files_and_process()

    def _download(self) -> None:
        for index, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[index] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self) -> None:
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        if not frames:
            raise ValueError("at least one SFT parquet file is required")
        self.dataframe = pd.concat(frames, ignore_index=True)
        _validate_dual_search_sft_schema(
            self.dataframe,
            messages_column=self.messages_key,
            image_column=self.image_key,
            tools_column=self.tools_key,
        )
        if self.messages_key not in self.dataframe.columns:
            raise KeyError(f"SFT dataset is missing {self.messages_key!r}")
        total = len(self.dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
        print(f"dataset len: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages = copy.deepcopy(decode_message_tool_arguments(example[self.messages_key]))
        images = normalize_arrow_value(example.get(self.image_key)) or []
        videos = normalize_arrow_value(example.get(self.video_key)) or []
        if not isinstance(images, list):
            images = [images]
        if not isinstance(videos, list):
            videos = [videos]

        image_offset = 0
        video_offset = 0
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("each SFT message must contain role and content")
            content = message["content"]
            if isinstance(content, list):
                # Already structured content still needs local media materialized.
                rebuilt = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        payload = item.get("image") or item.get("image_url") or item
                        rebuilt.append({"type": "image", "image": process_image(payload, self.image_patch_size)})
                    elif isinstance(item, dict) and item.get("type") == "video":
                        rebuilt.append({"type": "video", "video": process_video(item, self.image_patch_size)})
                    else:
                        rebuilt.append(item)
                message["content"] = rebuilt
                continue
            if not isinstance(content, str):
                raise TypeError("message content must be text or a multimodal content list")

            segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
            if not any(segment in {"<image>", "<video>"} for segment in segments):
                if self.processor is not None:
                    message["content"] = [{"type": "text", "text": content}]
                continue
            if self.processor is None:
                raise ValueError("a multimodal processor is required for image/video SFT samples")
            content_items: list[dict[str, Any]] = []
            for segment in segments:
                if segment == "<image>":
                    if image_offset >= len(images):
                        raise ValueError("more <image> placeholders than images")
                    image = process_image(images[image_offset], self.image_patch_size)
                    content_items.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    if video_offset >= len(videos):
                        raise ValueError("more <video> placeholders than videos")
                    video = process_video(videos[video_offset], self.image_patch_size)
                    content_items.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    content_items.append({"type": "text", "text": segment})
            message["content"] = content_items

        if image_offset != len(images):
            raise ValueError(f"used {image_offset} images but row contains {len(images)}")
        if video_offset != len(videos):
            raise ValueError(f"used {video_offset} videos but row contains {len(videos)}")
        return messages

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
        return_dict: bool,
    ) -> Any:
        processor = self.processor if self.processor is not None else self.tokenizer
        kwargs = dict(self.apply_chat_template_kwargs)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return apply_chat_template(
            processor,
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_tensors="pt" if return_dict else None,
            **kwargs,
        )

    def _render_ids(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        add_generation_prompt: bool,
        enable_thinking: bool | None,
    ) -> list[int]:
        rendered = self._render(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
        return _token_ids(rendered)

    def _assistant_loss_mask(
        self,
        full_ids: list[int],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        enable_thinking: bool | None,
    ) -> torch.Tensor:
        loss_mask = torch.zeros(len(full_ids), dtype=torch.long)
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prefix = messages[:index]
            completed = messages[: index + 1]
            before_ids = self._render_ids(
                prefix,
                tools,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            generation_ids = self._render_ids(
                prefix,
                tools,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            completed_ids = self._render_ids(
                completed,
                tools,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            prefix_matches = full_ids[: len(completed_ids)] == completed_ids
            if not prefix_matches and not self.ignore_input_ids_mismatch:
                raise AssertionError(
                    "chat template rendering is not prefix-stable; set ignore_input_ids_mismatch=True only after review"
                )
            if not prefix_matches:
                logger.warning("Ignoring non-prefix-stable chat template while constructing assistant loss mask")
                end = _longest_common_prefix(full_ids, completed_ids)
            else:
                end = len(completed_ids)
            start = max(len(before_ids), _longest_common_prefix(generation_ids, completed_ids))
            start = min(start, end)
            loss_mask[start:end] = 1
        if not torch.any(loss_mask):
            raise ValueError("SFT conversation contains no supervised assistant tokens")
        return loss_mask

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, Any],
    ) -> torch.Tensor:
        if self.processor is None:
            return torch.arange(input_ids.shape[0], dtype=torch.long)
        processor_name = self.processor.__class__.__name__
        image_processor_name = getattr(getattr(self.processor, "image_processor", None), "__class__", type(None)).__name__
        class_name = f"{processor_name} {image_processor_name}"
        rope_index = None
        if "Qwen3VL" in class_name:
            from verl.models.transformers.qwen3_vl import get_rope_index

            rope_index = get_rope_index
        elif "Qwen2VL" in class_name or "Qwen2_5_VL" in class_name:
            from verl.models.transformers.qwen2_vl import get_rope_index

            rope_index = get_rope_index
        if rope_index is None:
            return torch.arange(input_ids.shape[0], dtype=torch.long)

        vision_position_ids = rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
            video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
            second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask,
        )
        text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
        return torch.cat((text_position_ids, vision_position_ids), dim=0)

    def __getitem__(self, item: int) -> dict[str, Any]:
        example = normalize_arrow_value(self.dataframe.iloc[item].to_dict())
        messages = self._build_messages(example)
        tools = decode_tools_column(example.get(self.tools_key))
        enable_thinking = optional_bool(
            example.get(self.enable_thinking_key, self.enable_thinking_default),
            field=self.enable_thinking_key,
        )

        rendered = self._render(
            messages,
            tools,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_dict=True,
        )
        rendered = dict(rendered)
        input_ids = rendered.pop("input_ids").squeeze(0)
        attention_mask = rendered.pop("attention_mask", torch.ones_like(input_ids).unsqueeze(0)).squeeze(0)
        full_ids = input_ids.tolist()
        loss_mask = self._assistant_loss_mask(full_ids, messages, tools, enable_thinking)
        multi_modal_inputs = {
            key: value for key, value in rendered.items() if value is not None and key != "mm_token_type_ids"
        }
        position_ids = self._position_ids(input_ids, attention_mask, multi_modal_inputs)

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"sequence_length={sequence_length} is larger than max_length={self.max_length}")
            selection = slice(-self.max_length, None) if self.truncation == "left" else slice(0, self.max_length)
            input_ids = input_ids[selection]
            attention_mask = attention_mask[selection]
            loss_mask = loss_mask[selection]
            position_ids = position_ids[..., selection]
            sequence_length = self.max_length

        if self.pad_mode == DatasetPadMode.RIGHT and sequence_length < self.max_length:
            pad_size = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = F.pad(input_ids, (0, pad_size), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, pad_size), value=0)
            loss_mask = F.pad(loss_mask, (0, pad_size), value=0)
            position_ids = F.pad(position_ids, (0, pad_size), value=0)

        result = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        if self.pad_mode == DatasetPadMode.RIGHT:
            result["attention_mask"] = attention_mask
        if multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs
        return result
