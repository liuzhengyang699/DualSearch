import asyncio
import unittest
from types import SimpleNamespace

from dual_search.llm_agent.dual_search_agent_loop import DualSearchAgentLoop


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeServerManager:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def generate(self, **kwargs):
        text = self.outputs[self.calls]
        self.calls += 1
        token_ids = [ord(char) for char in text]
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=[0.0] * len(token_ids),
            num_preempted=None,
            routed_experts=None,
        )


class DualSearchAgentLoopBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def _make_agent(self, outputs, max_turns=1):
        agent = object.__new__(DualSearchAgentLoop)
        agent.max_turns = max_turns
        agent.response_length = 4096
        agent.max_obs_length = 512
        agent.search_url = "http://text-retriever"
        agent.vision_search_url = "http://vision-retriever"
        agent.topk = 3
        agent.request_timeout = 1
        agent.tokenizer = CharTokenizer()
        agent.server_manager = FakeServerManager(outputs)
        agent.loop = asyncio.get_running_loop()

        async def process_multi_modal_info(messages):
            return {}

        async def apply_chat_template(messages, **kwargs):
            return [1, 2, 3]

        agent.process_multi_modal_info = process_multi_modal_info
        agent.apply_chat_template = apply_chat_template
        agent._get_mm_processor_kwargs = lambda audios: {}
        return agent

    async def test_budget_exhaustion_returns_observation_then_allows_final_answer(self):
        agent = await self._make_agent(
            [
                "<think>first</think><search>first query</search>",
                "<think>again</think><search>second query</search>",
                "<think>enough</think><answer>final</answer>",
            ],
            max_turns=1,
        )
        search_calls = []

        async def batch_search(queries):
            search_calls.append(queries)
            return [[{"document": {"contents": '"Doc"\nEvidence'}}]]

        agent._batch_search = batch_search

        output = await agent.run(
            sampling_params={},
            raw_prompt=[{"role": "user", "content": "question"}],
        )
        response_text = agent.tokenizer.decode(output.response_ids)

        self.assertEqual(search_calls, [["first query"]])
        self.assertEqual(agent.server_manager.calls, 3)
        self.assertIn(
            "<information>Tool-call budget exhausted. Answer using the information already obtained.",
            response_text,
        )
        self.assertTrue(response_text.endswith("<think>enough</think><answer>final</answer>"))
        self.assertEqual(output.extra_fields["tool_budget_exhausted_stats"], 1)

        budget_start = response_text.index("<information>Tool-call budget exhausted.")
        budget_end = response_text.index("</information>", budget_start) + len("</information>")
        self.assertTrue(all(mask == 0 for mask in output.response_mask[budget_start:budget_end]))

    async def test_answer_only_turn_does_not_execute_another_tool(self):
        agent = await self._make_agent(
            [
                "<think>first</think><search>first query</search>",
                "<think>again</think><search>second query</search>",
                "<think>ignore instruction</think><search>third query</search>",
            ],
            max_turns=1,
        )
        search_calls = []

        async def batch_search(queries):
            search_calls.append(queries)
            return [[{"document": {"contents": '"Doc"\nEvidence'}}]]

        agent._batch_search = batch_search

        output = await agent.run(
            sampling_params={},
            raw_prompt=[{"role": "user", "content": "question"}],
        )
        response_text = agent.tokenizer.decode(output.response_ids)

        self.assertEqual(search_calls, [["first query"]])
        self.assertEqual(agent.server_manager.calls, 3)
        self.assertTrue(response_text.endswith("<search>third query</search>"))
        self.assertEqual(output.extra_fields["tool_budget_exhausted_stats"], 1)

    async def test_vision_budget_observation_preserves_protocol_tag(self):
        agent = await self._make_agent([], max_turns=0)
        observation = agent._tool_budget_exhausted_observation("vision_search")

        self.assertIn("<vision_information>", observation)
        self.assertIn("</vision_information>", observation)
        self.assertNotIn("<information>", observation)


if __name__ == "__main__":
    unittest.main()
