import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from dual_search.reward import genrm_judge
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


VALID_SOLUTION = "<think>identify the entity</think><answer>Bronze Copper</answer>"
RAW_PROMPT = [
    {
        "role": "user",
        "content": "<image>\nAnswer the question about the image.\nQuestion: What is the common name?",
    }
]
GROUND_TRUTH = {"target": ["Bronze Copper"]}


class GenRMJudgeTest(unittest.IsolatedAsyncioTestCase):
    def test_extract_question_from_multimodal_prompt(self):
        raw_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/tmp/image.png"},
                    {"type": "text", "text": "Use the images.\nQuestion: Which species is shown?"},
                ],
            }
        ]
        self.assertEqual(genrm_judge.extract_question(raw_prompt), "Which species is shown?")

    def test_first_two_retrieval_calls_are_free(self):
        self.assertEqual(genrm_judge.compute_retrieval_penalty(0), 0.0)
        self.assertEqual(genrm_judge.compute_retrieval_penalty(1), 0.0)
        self.assertEqual(genrm_judge.compute_retrieval_penalty(2), 0.0)

    def test_quadratic_retrieval_penalty(self):
        self.assertAlmostEqual(genrm_judge.compute_retrieval_penalty(4), 0.08)
        self.assertAlmostEqual(genrm_judge.compute_retrieval_penalty(6), 0.32)
        self.assertAlmostEqual(genrm_judge.compute_retrieval_penalty(8), 0.72)

    def test_reward_is_not_clipped_at_zero(self):
        self.assertAlmostEqual(genrm_judge.compute_total_reward(0.0, True, 8), -0.52)
        self.assertAlmostEqual(genrm_judge.compute_total_reward(1.0, True, 8), 0.28)
        self.assertAlmostEqual(genrm_judge.compute_total_reward(0.0, False, 8), -0.72)

    async def test_correct_judge_output_uses_requested_prompt(self):
        with patch.object(
            genrm_judge,
            "request_genrm",
            new=AsyncMock(return_value='{"score": 1.0}'),
        ) as request_mock:
            result = await genrm_judge.compute_score(
                data_source="dual_search",
                solution_str=VALID_SOLUTION,
                ground_truth=GROUND_TRUTH,
                raw_prompt=RAW_PROMPT,
                reward_router_address="127.0.0.1:9000",
                genrm_model="/models/small-genrm",
            )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["judge_score"], 1.0)
        self.assertEqual(result["judge_valid"], 1.0)
        self.assertEqual(result["retrieval_call_count"], 0.0)
        self.assertEqual(result["retrieval_penalty"], 0.0)
        messages = request_mock.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["content"], genrm_judge.SYSTEM_PROMPT)
        self.assertEqual(
            messages[1]["content"],
            "Question: What is the common name?\n\n"
            "Reference answer: Bronze Copper\n\n"
            "Candidate answer: Bronze Copper",
        )

    async def test_noncompliant_output_returns_zero(self):
        with patch.object(
            genrm_judge,
            "request_genrm",
            new=AsyncMock(return_value='```json\n{"score": 1.0}\n```'),
        ):
            result = await genrm_judge.compute_score(
                data_source="dual_search",
                solution_str=VALID_SOLUTION,
                ground_truth=GROUND_TRUTH,
                raw_prompt=RAW_PROMPT,
                reward_router_address="127.0.0.1:9000",
                genrm_model="/models/small-genrm",
                extra_info={"valid_search_stats": 4, "valid_vision_search_stats": 4},
            )

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["judge_valid"], 0.0)
        self.assertEqual(result["retrieval_call_count"], 8.0)
        self.assertAlmostEqual(result["retrieval_penalty"], 0.72)

    async def test_request_failure_returns_zero(self):
        with patch.object(
            genrm_judge,
            "request_genrm",
            new=AsyncMock(side_effect=RuntimeError("server unavailable")),
        ):
            result = await genrm_judge.compute_score(
                data_source="dual_search",
                solution_str=VALID_SOLUTION,
                ground_truth=GROUND_TRUTH,
                raw_prompt=RAW_PROMPT,
                reward_router_address="127.0.0.1:9000",
                genrm_model="/models/small-genrm",
            )

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["judge_valid"], 0.0)

    async def test_valid_incorrect_judgement_keeps_format_reward(self):
        with patch.object(
            genrm_judge,
            "request_genrm",
            new=AsyncMock(return_value='{"score": 0.0}'),
        ):
            result = await genrm_judge.compute_score(
                data_source="dual_search",
                solution_str=VALID_SOLUTION,
                ground_truth=GROUND_TRUTH,
                raw_prompt=RAW_PROMPT,
                reward_router_address="127.0.0.1:9000",
                genrm_model="/models/small-genrm",
            )

        self.assertEqual(result["score"], 0.2)
        self.assertEqual(result["judge_score"], 0.0)
        self.assertEqual(result["format_score"], 1.0)
        self.assertEqual(result["judge_valid"], 1.0)

    async def test_eight_retrieval_calls_can_produce_negative_reward(self):
        with patch.object(
            genrm_judge,
            "request_genrm",
            new=AsyncMock(return_value='{"score": 0.0}'),
        ):
            result = await genrm_judge.compute_score(
                data_source="dual_search",
                solution_str=VALID_SOLUTION,
                ground_truth=GROUND_TRUTH,
                raw_prompt=RAW_PROMPT,
                reward_router_address="127.0.0.1:9000",
                genrm_model="/models/small-genrm",
                extra_info={"valid_search_stats": 3, "valid_vision_search_stats": 5},
            )

        self.assertAlmostEqual(result["score"], -0.52)
        self.assertEqual(result["retrieval_call_count"], 8.0)
        self.assertAlmostEqual(result["retrieval_penalty"], 0.72)


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return VALID_SOLUTION


class _FakeData:
    def __init__(self, item):
        self.item = item

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self
        if key == 0:
            return self.item
        raise IndexError(key)


class NaiveRewardManagerRawPromptTest(unittest.IsolatedAsyncioTestCase):
    async def test_raw_prompt_is_forwarded_to_custom_reward(self):
        captured = {}

        async def compute_score(**kwargs):
            captured.update(kwargs)
            return {"score": 1.0}

        manager = object.__new__(NaiveRewardManager)
        manager.compute_score = compute_score
        manager.is_async_reward_score = True
        manager.reward_router_address = None
        manager.reward_model_tokenizer = None
        manager.loop = asyncio.get_running_loop()
        manager.tokenizer = _FakeTokenizer()

        item = SimpleNamespace(
            batch={
                "responses": torch.tensor([1, 2, 3], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            },
            non_tensor_batch={
                "data_source": "dual_search",
                "reward_model": {"ground_truth": GROUND_TRUTH},
                "raw_prompt": RAW_PROMPT,
                "extra_info": {},
            },
        )

        result = await manager.run_single(_FakeData(item))

        self.assertEqual(result["reward_score"], 1.0)
        self.assertEqual(captured["raw_prompt"], RAW_PROMPT)


if __name__ == "__main__":
    unittest.main()
