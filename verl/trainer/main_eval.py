# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline evaluation with overall and retrieval-resolvable subset metrics."""

import asyncio
import inspect
import json
from collections.abc import Mapping
from collections import defaultdict
from typing import Any

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


SUBSET_NAME = "retrieval_resolvable=true"


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _as_response_list(value: Any) -> list[str]:
    value = _to_python(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _is_explicit_true(extra_info: Any) -> bool:
    extra_info = _to_python(extra_info)
    if not isinstance(extra_info, Mapping):
        return False
    value = extra_info.get("retrieval_resolvable")
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _merge_resolvability_flag(extra_info: Any, fallback: Any) -> dict[str, Any]:
    """Return a plain extra-info mapping with a top-level flag fallback.

    New DualSearch parquet files intentionally store resolvability both as a
    physical column and inside ``extra_info``.  Keeping the fallback here makes
    offline evaluation robust to producers that only kept the physical column,
    while legacy files with neither field remain outside the subset.
    """

    extra_info = _to_python(extra_info)
    merged = dict(extra_info) if isinstance(extra_info, Mapping) else {}
    if "retrieval_resolvable" not in merged and isinstance(fallback, (bool, np.bool_)):
        merged["retrieval_resolvable"] = bool(fallback)
    return merged


def _numeric_metrics(value: Any) -> dict[str, float]:
    value = _to_python(value)
    if isinstance(value, dict):
        metrics = {}
        for key, item in value.items():
            if isinstance(item, bool) or not isinstance(item, (int, float, np.number)):
                continue
            metrics[str(key)] = float(item)
        return metrics
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"Reward function returned unsupported value: {type(value).__name__}")
    return {"score": float(value)}


def _resolve_reward(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _process_item_impl(
    config,
    data_source,
    response_value,
    reward_data,
    raw_prompt,
    extra_info,
    *,
    reward_fn=None,
):
    """Evaluate every rollout for one prompt and average within the prompt."""

    reward_fn = reward_fn or get_custom_reward_fn(config)
    if reward_fn is None:
        raise ValueError("Offline evaluation requires reward.custom_reward_function.path")

    reward_data = _to_python(reward_data)
    ground_truth = reward_data.get("ground_truth") if isinstance(reward_data, dict) else reward_data
    extra_info = _to_python(extra_info)
    raw_prompt = _to_python(raw_prompt)
    metric_values: dict[str, list[float]] = defaultdict(list)
    responses = _as_response_list(response_value)
    for response in responses:
        result = reward_fn(
            data_source,
            response,
            ground_truth,
            extra_info=extra_info,
            raw_prompt=raw_prompt,
        )
        for key, value in _numeric_metrics(_resolve_reward(result)).items():
            metric_values[key].append(value)

    means = {key: float(np.mean(values)) for key, values in metric_values.items() if values}
    return str(data_source), means, _is_explicit_true(extra_info), len(responses)


@ray.remote
def process_item(config, data_source, response_value, reward_data, raw_prompt, extra_info):
    return _process_item_impl(config, data_source, response_value, reward_data, raw_prompt, extra_info)


def _column_or_default(dataset: pd.DataFrame, key: str, default: Any) -> list[Any]:
    if key in dataset.columns:
        return dataset[key].tolist()
    return [default for _ in range(len(dataset))]


def _aggregate_results(results: list[tuple[str, dict[str, float], bool, int]]) -> dict[str, float | int]:
    overall: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    subset: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prompt_counts: dict[str, int] = defaultdict(int)
    response_counts: dict[str, int] = defaultdict(int)
    subset_prompt_counts: dict[str, int] = defaultdict(int)
    subset_response_counts: dict[str, int] = defaultdict(int)

    for data_source, metrics, is_resolvable, response_count in results:
        prompt_counts[data_source] += 1
        response_counts[data_source] += response_count
        for key, value in metrics.items():
            overall[data_source][key].append(value)
        if is_resolvable:
            subset_prompt_counts[data_source] += 1
            subset_response_counts[data_source] += response_count
            for key, value in metrics.items():
                subset[data_source][key].append(value)

    metric_dict: dict[str, float | int] = {}
    for data_source in sorted(prompt_counts):
        # Keep the compact count keys emitted by the first implementation and
        # additionally expose explicit prompt/response names for dashboards.
        metric_dict[f"test_count/{data_source}"] = prompt_counts[data_source]
        metric_dict[f"test_count/{data_source}/responses"] = response_counts[data_source]
        metric_dict[f"test_count/{data_source}/subset/{SUBSET_NAME}"] = subset_prompt_counts[data_source]
        metric_dict[f"test_count/{data_source}/subset/{SUBSET_NAME}/responses"] = subset_response_counts[data_source]
        metric_dict[f"test_count/{data_source}/num_prompts"] = prompt_counts[data_source]
        metric_dict[f"test_count/{data_source}/num_responses"] = response_counts[data_source]
        metric_dict[f"test_count/{data_source}/subset/{SUBSET_NAME}/num_prompts"] = subset_prompt_counts[
            data_source
        ]
        metric_dict[f"test_count/{data_source}/subset/{SUBSET_NAME}/num_responses"] = subset_response_counts[
            data_source
        ]

        for metric_name, values in overall[data_source].items():
            mean_value = float(np.mean(values))
            if metric_name == "score":
                metric_dict[f"test_score/{data_source}"] = mean_value
            else:
                metric_dict[f"test_metric/{data_source}/{metric_name}"] = mean_value
        for accuracy_name in ("acc", "accuracy", "judge_score"):
            if overall[data_source].get(accuracy_name):
                metric_dict[f"test_accuracy/{data_source}"] = float(np.mean(overall[data_source][accuracy_name]))
                break

        for metric_name, values in subset[data_source].items():
            if not values:
                continue
            mean_value = float(np.mean(values))
            if metric_name == "score":
                metric_dict[f"test_score/{data_source}/subset/{SUBSET_NAME}"] = mean_value
            else:
                metric_dict[f"test_metric/{data_source}/subset/{SUBSET_NAME}/{metric_name}"] = mean_value
        for accuracy_name in ("acc", "accuracy", "judge_score"):
            if subset[data_source].get(accuracy_name):
                metric_dict[f"test_accuracy/{data_source}/subset/{SUBSET_NAME}"] = float(
                    np.mean(subset[data_source][accuracy_name])
                )
                break
    return metric_dict


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key].tolist()
    data_sources = dataset[config.data.data_source_key].tolist()
    reward_model_data = dataset[config.data.reward_model_key].tolist()
    raw_prompts = _column_or_default(dataset, config.data.prompt_key, None)
    stored_extra_infos = _column_or_default(dataset, config.data.extra_info_key, {})
    resolvability_key = config.data.get("retrieval_resolvable_key", "retrieval_resolvable")
    top_level_resolvability = _column_or_default(dataset, resolvability_key, None)
    extra_infos = [
        _merge_resolvability_flag(extra_info, fallback)
        for extra_info, fallback in zip(stored_extra_infos, top_level_resolvability, strict=True)
    ]

    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    remote_tasks = [
        process_item.remote(
            config,
            data_sources[index],
            responses[index],
            reward_model_data[index],
            raw_prompts[index],
            extra_infos[index],
        )
        for index in range(len(dataset))
    ]
    results = []
    with tqdm(total=len(remote_tasks)) as pbar:
        while remote_tasks:
            done_ids, remote_tasks = ray.wait(remote_tasks)
            results.extend(ray.get(done_ids))
            pbar.update(len(done_ids))

    print(json.dumps(_aggregate_results(results), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
