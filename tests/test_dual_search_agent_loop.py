import asyncio
import unittest
from types import SimpleNamespace

from dual_search.llm_agent.dual_search_agent_loop import DualSearchAgentLoop
from dual_search.protocol import DUAL_SEARCH_TOOL_SCHEMAS, parse_assistant_action


def tool_call(name, arguments, think="plan"):
    import json

    return (
        f"<think>{think}</think><tool_call>"
        + json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
        + "</tool_call>"
    )


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class SpecialBoundaryTokenizer(CharTokenizer):
    SPECIAL_ID = 0

    def decode(self, token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids if token_id != self.SPECIAL_ID]
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens)


class FakeServerManager:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.seen_prompt_ids = []

    async def generate(self, **kwargs):
        text = self.outputs[self.calls]
        self.calls += 1
        self.seen_prompt_ids.append(kwargs["prompt_ids"])
        token_ids = [ord(char) for char in text]
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=[float(index) for index in range(len(token_ids))],
            num_preempted=None,
            routed_experts=None,
        )


class DualSearchAgentLoopTest(unittest.IsolatedAsyncioTestCase):
    async def _make_agent(self, outputs, max_turns=1, raw_images=None):
        agent = object.__new__(DualSearchAgentLoop)
        agent.max_turns = max_turns
        agent.response_length = 8192
        agent.max_obs_length = 500
        agent.search_url = "http://text-retriever"
        agent.vision_search_url = "http://vision-retriever"
        agent.topk = 3
        agent.request_timeout = 1
        agent.tokenizer = CharTokenizer()
        agent.server_manager = FakeServerManager(outputs)
        agent.loop = asyncio.get_running_loop()
        agent.template_calls = []

        async def process_multi_modal_info(messages):
            return {"images": list(raw_images)} if raw_images else {}

        async def apply_chat_template(messages, tools=None, remove_system_prompt=False, **kwargs):
            agent.template_calls.append(
                {"messages": messages, "tools": tools, "remove_system_prompt": remove_system_prompt}
            )
            if messages and messages[0].get("role") == "tool":
                rendered = f"<tool_response>{messages[0]['content']}</tool_response>"
                return agent.tokenizer.encode(rendered)
            return [1, 2, 3]

        agent.process_multi_modal_info = process_multi_modal_info
        agent.apply_chat_template = apply_chat_template
        agent._get_mm_processor_kwargs = lambda audios: {}
        return agent

    async def test_schema_is_passed_to_initial_chat_template(self):
        agent = await self._make_agent(["<think>done</think><answer>final</answer>"])
        await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])

        self.assertEqual(agent.template_calls[0]["tools"], DUAL_SEARCH_TOOL_SCHEMAS)

    async def test_last_allowed_tool_result_is_followed_by_answer_only_turn(self):
        agent = await self._make_agent(
            [
                tool_call("search", {"query": "first query"}, think="first"),
                "<think>enough</think><answer>final</answer>",
            ],
            max_turns=1,
        )
        search_calls = []

        async def batch_search(queries):
            search_calls.append(queries)
            return [[{"document": {"contents": '"Doc"\nEvidence'}}]]

        agent._batch_search = batch_search
        output = await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])
        response_text = agent.tokenizer.decode(output.response_ids)

        self.assertEqual(search_calls, [["first query"]])
        self.assertEqual(agent.server_manager.calls, 2)
        self.assertIn("<tool_response>Doc 1(Title: Doc) Evidence", response_text)
        self.assertIn("Tool-call budget exhausted.", response_text)
        self.assertNotIn("<information>", response_text)
        self.assertNotIn("<vision_information>", response_text)
        self.assertTrue(response_text.endswith("<think>enough</think><answer>final</answer>"))
        self.assertEqual(output.extra_fields["tool_attempt_stats"], 1)
        self.assertEqual(output.extra_fields["tool_budget_exhausted_stats"], 1)

        observation_start = response_text.index("<tool_response>")
        observation_end = response_text.index("</tool_response>", observation_start) + len("</tool_response>")
        self.assertTrue(all(mask == 0 for mask in output.response_mask[observation_start:observation_end]))

    async def test_answer_only_turn_never_executes_another_tool(self):
        agent = await self._make_agent(
            [
                tool_call("search", {"query": "first query"}),
                tool_call("search", {"query": "ignored query"}),
            ],
            max_turns=1,
        )
        search_calls = []

        async def batch_search(queries):
            search_calls.append(queries)
            return [[]]

        agent._batch_search = batch_search
        output = await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])

        self.assertEqual(search_calls, [["first query"]])
        self.assertEqual(agent.server_manager.calls, 2)
        self.assertEqual(output.extra_fields["valid_search_stats"], 1)
        self.assertEqual(output.extra_fields["tool_attempt_stats"], 1)

    async def test_exactly_eight_attempts_then_one_answer_only_generation(self):
        calls = [tool_call("search", {"query": f"query-{index}"}) for index in range(8)]
        agent = await self._make_agent(
            [*calls, "<think>done</think><answer>answer</answer>"],
            max_turns=8,
        )
        search_calls = []

        async def batch_search(queries):
            search_calls.append(queries)
            return [[]]

        agent._batch_search = batch_search
        output = await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])

        self.assertEqual(search_calls, [[f"query-{index}"] for index in range(8)])
        self.assertEqual(agent.server_manager.calls, 9)
        self.assertEqual(output.extra_fields["tool_attempt_stats"], 8)
        self.assertEqual(output.extra_fields["valid_search_stats"], 8)
        self.assertEqual(output.extra_fields["tool_budget_exhausted_stats"], 1)
        self.assertEqual(len(output.response_ids), len(output.response_mask))
        self.assertEqual(len(output.response_ids), len(output.response_logprobs))

    async def test_invalid_legacy_call_consumes_budget_without_execution(self):
        agent = await self._make_agent(
            [
                "<think>old</think><search>legacy query</search>",
                tool_call("search", {"query": "native query"}),
                "<think>done</think><answer>answer</answer>",
            ],
            max_turns=2,
        )
        search_calls = []

        async def batch_search(queries):
            search_calls.append(queries)
            return [[]]

        agent._batch_search = batch_search
        output = await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])

        self.assertEqual(search_calls, [["native query"]])
        self.assertEqual(output.extra_fields["tool_attempt_stats"], 2)
        self.assertEqual(output.extra_fields["invalid_tool_call_stats"], 1)
        self.assertEqual(output.extra_fields["valid_search_stats"], 1)

    async def test_backend_failure_consumes_budget_but_is_not_valid(self):
        agent = await self._make_agent(
            [tool_call("search", {"query": "query"}), "<think>done</think><answer>answer</answer>"],
            max_turns=1,
        )

        async def batch_search(queries):
            raise RuntimeError("offline")

        agent._batch_search = batch_search
        output = await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])
        response_text = agent.tokenizer.decode(output.response_ids)

        self.assertIn("Tool error (search): offline", response_text)
        self.assertEqual(output.extra_fields["tool_attempt_stats"], 1)
        self.assertEqual(output.extra_fields["tool_execution_failure_stats"], 1)
        self.assertEqual(output.extra_fields["valid_search_stats"], 0)

    async def test_vision_search_sends_separate_index_and_query(self):
        agent = await self._make_agent(
            [
                tool_call("vision_search", {"image_index": 1, "query": "butterfly wing pattern"}),
                "<think>done</think><answer>answer</answer>",
            ],
            max_turns=1,
            raw_images=["/tmp/query.jpg"],
        )
        requests = []

        async def batch_vision_search(items):
            requests.extend(items)
            return [[]]

        agent._batch_vision_search = batch_vision_search
        await agent.run(
            sampling_params={},
            raw_prompt=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/tmp/query.jpg"},
                        {"type": "text", "text": "question"},
                    ],
                }
            ],
        )

        self.assertEqual(
            requests,
            [
                {
                    "query": "butterfly wing pattern",
                    "image_index": 1,
                    "image": "/tmp/query.jpg",
                }
            ],
        )

    async def test_action_truncation_keeps_original_token_logprob_alignment(self):
        action = "<think>done</think><answer>final</answer>"
        agent = await self._make_agent([action])
        output = await agent.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "question"}])

        self.assertEqual(output.response_ids, agent.tokenizer.encode(action))
        self.assertEqual(output.response_logprobs, [float(index) for index in range(len(action))])
        self.assertEqual(len(output.response_ids), len(output.response_mask))
        self.assertEqual(len(output.response_ids), len(output.response_logprobs))

    async def test_token_boundary_keeps_qwen_assistant_end_special_token(self):
        agent = await self._make_agent([])
        agent.tokenizer = SpecialBoundaryTokenizer()
        action = "<think>done</think><answer>final</answer>"
        token_ids = agent.tokenizer.encode(action) + [SpecialBoundaryTokenizer.SPECIAL_ID]
        decoded = agent.tokenizer.decode(token_ids, skip_special_tokens=True)

        selected = await agent._token_prefix_at_character_boundary(token_ids, decoded, len(decoded))

        self.assertEqual(selected, token_ids)

    async def test_trailing_whitespace_keeps_qwen_assistant_end_special_token(self):
        agent = await self._make_agent([])
        agent.tokenizer = SpecialBoundaryTokenizer()
        action = "<think>done</think><answer>final</answer>\n"
        parsed = parse_assistant_action(action)
        token_ids = agent.tokenizer.encode(action) + [SpecialBoundaryTokenizer.SPECIAL_ID]
        decoded = agent.tokenizer.decode(token_ids, skip_special_tokens=True)

        selected = await agent._token_prefix_at_character_boundary(token_ids, decoded, parsed.end_offset)

        self.assertEqual(parsed.end_offset, len(decoded))
        self.assertEqual(selected, token_ids)


if __name__ == "__main__":
    unittest.main()
