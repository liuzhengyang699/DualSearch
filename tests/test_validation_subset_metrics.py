import ast
import importlib.util
import sys
import types
import unittest
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_main_eval():
    """Load the offline helpers without importing the optional VERL stack."""

    fake_ray = types.ModuleType("ray")

    class _RemoteFunction:
        def __init__(self, function):
            self.function = function

        def remote(self, *args, **kwargs):  # pragma: no cover - main() owns this path
            return self.function(*args, **kwargs)

    fake_ray.remote = lambda function: _RemoteFunction(function)
    fake_ray.is_initialized = lambda: False
    fake_ray.init = lambda **kwargs: None

    fake_hydra = types.ModuleType("hydra")
    fake_hydra.main = lambda **kwargs: (lambda function: function)

    fake_omegaconf = types.ModuleType("omegaconf")
    fake_omegaconf.OmegaConf = types.SimpleNamespace(to_container=lambda value: value)

    fake_reward = types.ModuleType("verl.trainer.ppo.reward")
    fake_reward.get_custom_reward_fn = lambda config: None
    fake_fs = types.ModuleType("verl.utils.fs")
    fake_fs.copy_to_local = lambda path, use_shm=False: path

    fake_modules = {
        "ray": fake_ray,
        "hydra": fake_hydra,
        "omegaconf": fake_omegaconf,
        "verl": types.ModuleType("verl"),
        "verl.trainer": types.ModuleType("verl.trainer"),
        "verl.trainer.ppo": types.ModuleType("verl.trainer.ppo"),
        "verl.trainer.ppo.reward": fake_reward,
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.fs": fake_fs,
    }
    module_name = "_dual_search_main_eval_test_module"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "verl/trainer/main_eval.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, fake_modules):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def _fake_process_validation_metrics(data_sources, sample_uids, infos_dict):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, (source, uid) in enumerate(zip(data_sources, sample_uids, strict=True)):
        for metric_name, values in infos_dict.items():
            grouped[source][uid][metric_name].append(values[index])

    result = defaultdict(lambda: defaultdict(dict))
    for source, uid_metrics in grouped.items():
        metric_names = {name for metrics in uid_metrics.values() for name in metrics}
        for metric_name in metric_names:
            prompt_means = []
            response_count = 0
            for metrics in uid_metrics.values():
                values = metrics.get(metric_name, [])
                if values:
                    response_count = max(response_count, len(values))
                    prompt_means.append(float(np.mean(values)))
            if prompt_means:
                result[source][metric_name][f"mean@{response_count}"] = float(np.mean(prompt_means))
    return result


def _load_online_metric_methods():
    """Execute only the dependency-free validation helpers from ray_trainer."""

    source_path = ROOT / "verl/trainer/ppo/ray_trainer.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    wanted_functions = {
        "_is_explicit_true",
        "_extract_retrieval_resolvable_mask",
        "_filter_validation_values",
    }
    body = []
    class_methods = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer":
            class_methods = {
                item.name: item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and item.name in {"_val_metrics_update", "_merge_validation_results"}
            }
    body.extend(class_methods.values())
    namespace = {
        "Any": Any,
        "Mapping": Mapping,
        "np": np,
        "process_validation_metrics": _fake_process_validation_metrics,
        "RETRIEVAL_RESOLVABLE_SUBSET": "retrieval_resolvable=true",
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)

    class _Harness:
        pass

    _Harness._val_metrics_update = namespace["_val_metrics_update"]
    _Harness._merge_validation_results = namespace["_merge_validation_results"]
    return namespace, _Harness()


class OfflineEvaluationMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_main_eval()

    def test_async_reward_is_awaited_and_receives_prompt_metadata(self):
        calls = []

        async def reward_fn(data_source, response, ground_truth, *, extra_info, raw_prompt):
            calls.append((data_source, response, ground_truth, extra_info, raw_prompt))
            score = 1.0 if response == "right" else 0.0
            return {"score": score, "judge_score": score}

        result = self.module._process_item_impl(
            None,
            "dual_search",
            ["right", "wrong"],
            {"ground_truth": {"target": ["answer"]}},
            [{"role": "user", "content": "question"}],
            {"retrieval_resolvable": True, "sample_id": "sample-1"},
            reward_fn=reward_fn,
        )

        self.assertEqual(result, ("dual_search", {"score": 0.5, "judge_score": 0.5}, True, 2))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][3]["sample_id"], "sample-1")
        self.assertEqual(calls[0][4][0]["role"], "user")

    def test_multiple_rollouts_are_prompt_averaged_and_counted(self):
        metrics = self.module._aggregate_results(
            [
                ("dual_search", {"score": 0.5, "judge_score": 0.5}, True, 2),
                ("dual_search", {"score": 1.0, "judge_score": 1.0}, False, 1),
            ]
        )

        self.assertEqual(metrics["test_score/dual_search"], 0.75)
        self.assertEqual(metrics["test_accuracy/dual_search"], 0.75)
        self.assertEqual(metrics["test_count/dual_search/num_prompts"], 2)
        self.assertEqual(metrics["test_count/dual_search/num_responses"], 3)
        subset = "retrieval_resolvable=true"
        self.assertEqual(metrics[f"test_score/dual_search/subset/{subset}"], 0.5)
        self.assertEqual(metrics[f"test_accuracy/dual_search/subset/{subset}"], 0.5)
        self.assertEqual(metrics[f"test_count/dual_search/subset/{subset}/num_prompts"], 1)
        self.assertEqual(metrics[f"test_count/dual_search/subset/{subset}/num_responses"], 2)

    def test_empty_and_legacy_subsets_emit_zero_counts_without_nan_metrics(self):
        metrics = self.module._aggregate_results(
            [("dual_search", {"score": 0.25, "judge_score": 0.0}, False, 4)]
        )
        subset = "retrieval_resolvable=true"
        self.assertEqual(metrics[f"test_count/dual_search/subset/{subset}/num_prompts"], 0)
        self.assertEqual(metrics[f"test_count/dual_search/subset/{subset}/num_responses"], 0)
        self.assertNotIn(f"test_score/dual_search/subset/{subset}", metrics)
        self.assertNotIn(f"test_accuracy/dual_search/subset/{subset}", metrics)

        self.assertFalse(self.module._is_explicit_true({}))
        self.assertFalse(self.module._is_explicit_true(None))
        merged = self.module._merge_resolvability_flag({}, np.bool_(True))
        self.assertTrue(self.module._is_explicit_true(merged))


class OnlineValidationMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.harness = _load_online_metric_methods()

    def test_nested_flag_precedence_and_top_level_fallback(self):
        extract = self.namespace["_extract_retrieval_resolvable_mask"]
        mask = extract(
            [{"retrieval_resolvable": False}, {}, None],
            [True, np.bool_(True), False],
        )
        self.assertEqual(mask, [False, True, False])
        self.assertEqual(extract(None, [True, False]), [True, False])
        self.assertEqual(extract([{}, {}]), [False, False])

    def test_overall_names_remain_and_subset_handles_multiple_rollouts(self):
        subset = "retrieval_resolvable=true"
        metrics = self.harness._val_metrics_update(
            ["dual_search"] * 4,
            ["prompt-1", "prompt-1", "prompt-2", "prompt-2"],
            {"reward": [0.0, 1.0, 1.0, 1.0], "acc": [0.0, 1.0, 1.0, 1.0]},
            [],
            subset_masks={subset: [True, True, False, False]},
        )

        self.assertEqual(metrics["val-core/dual_search/acc/mean@2"], 0.75)
        self.assertEqual(metrics[f"val-core/dual_search/subset/{subset}/acc/mean@2"], 0.5)
        self.assertEqual(metrics[f"val-aux/dual_search/subset/{subset}/num_prompts"], 1)
        self.assertEqual(metrics[f"val-aux/dual_search/subset/{subset}/num_responses"], 2)

    def test_empty_subset_emits_only_zero_counts(self):
        subset = "retrieval_resolvable=true"
        metrics = self.harness._val_metrics_update(
            ["dual_search", "dual_search"],
            ["prompt-1", "prompt-1"],
            {"reward": [0.2, 0.4]},
            [],
            subset_masks={subset: [False, False]},
        )
        self.assertEqual(metrics[f"val-aux/dual_search/subset/{subset}/num_prompts"], 0)
        self.assertEqual(metrics[f"val-aux/dual_search/subset/{subset}/num_responses"], 0)
        self.assertFalse(any(f"/subset/{subset}/reward/" in key for key in metrics))

    def test_merge_accepts_legacy_payload_without_subset_field(self):
        subset = "retrieval_resolvable=true"
        legacy = {
            "data_sources": [np.array(["dual_search", "dual_search"], dtype=object)],
            "sample_uids": ["prompt-1", "prompt-1"],
            "sample_turns": [],
            "reward_extra_infos_dict": {"reward": [0.0, 1.0]},
        }
        metrics = self.harness._merge_validation_results(legacy, None)
        self.assertEqual(metrics["val-core/dual_search/reward/mean@2"], 0.5)
        self.assertEqual(metrics[f"val-aux/dual_search/subset/{subset}/num_prompts"], 0)
        self.assertEqual(metrics[f"val-aux/dual_search/subset/{subset}/num_responses"], 0)


if __name__ == "__main__":
    unittest.main()
