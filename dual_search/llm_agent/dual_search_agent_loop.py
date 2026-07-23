import base64
import io
import logging
import os
from typing import Any
from uuid import uuid4

import requests
from PIL import Image

from dual_search.protocol import (
    ParsedAssistantAction,
    ToolCall,
    format_text_results,
    format_vision_results,
    get_tool_schemas,
    parse_assistant_action,
    sanitize_tool_response,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_TOOL_ATTEMPTS = 8


class DualSearchAgentLoop(AgentLoopBase):
    """DualSearch loop using Qwen's native ``<tool_call>`` JSON protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        retriever_config = self.config.get("retriever", {})
        self.search_url = retriever_config.get("url")
        self.vision_search_url = retriever_config.get("vision_search_url")
        self.topk = retriever_config.get("topk", 3)
        self.request_timeout = retriever_config.get("timeout", 60)
        # Configurations may deliberately choose a smaller budget for ablations,
        # but the project protocol has a hard upper bound of eight attempts.
        configured_max_turns = int(self.config.get("max_turns", MAX_TOOL_ATTEMPTS))
        self.max_turns = min(MAX_TOOL_ATTEMPTS, max(0, configured_max_turns))
        self.max_obs_length = retriever_config.get(
            "max_obs_length",
            self.config.data.get("max_obs_length", self.rollout_config.multi_turn.max_tool_response_length),
        )
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)
        # Prefer processor-materialized images (normally RGB PIL objects). Raw
        # OpenAI/Qwen message payloads can contain nested ``image_url`` objects
        # that are not directly consumable by the retrieval service; retain
        # them only as a fallback when no multimodal processor is configured.
        image_refs = self._normalize_images(images) or self._extract_image_references(messages)

        # This is the authoritative schema injection point.  The exact same
        # objects are serialized into SFT examples by dual_search.protocol.
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=get_tool_schemas(),
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        has_logprobs = False
        routed_experts = None
        valid_action_count = 0
        valid_search_count = 0
        valid_vision_search_count = 0
        tool_attempt_count = 0
        invalid_tool_call_count = 0
        tool_execution_failure_count = 0
        tool_budget_exhausted_count = 0
        num_preempted = 0
        assistant_turns = 0
        answer_only_turn = self.max_turns == 0

        # Up to ``max_turns`` attempted tool interactions, then exactly one
        # answer-only generation.  Malformed calls and backend failures consume
        # the same budget as successful calls.
        while assistant_turns < self.max_turns + 1:
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break

            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = remaining
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params=turn_sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            assistant_turns += 1

            num_preempted += output.num_preempted if output.num_preempted is not None else 0
            if routed_experts is None and output.routed_experts is not None:
                routed_experts = output.routed_experts

            generated_ids = list(output.token_ids)
            action_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            parsed = parse_assistant_action(action_text, image_count=len(image_refs))
            if parsed.end_offset is not None:
                generated_ids = await self._token_prefix_at_character_boundary(
                    generated_ids, action_text, parsed.end_offset
                )
            generated_ids = generated_ids[:remaining]
            if not generated_ids:
                break

            prior_length = len(response_ids)
            output_logprobs = output.log_probs
            if output_logprobs is not None:
                if not has_logprobs:
                    response_logprobs.extend([0.0] * prior_length)
                    has_logprobs = True
                selected_logprobs = list(output_logprobs)[: len(generated_ids)]
                response_logprobs.extend(selected_logprobs)
                if len(selected_logprobs) < len(generated_ids):
                    response_logprobs.extend([0.0] * (len(generated_ids) - len(selected_logprobs)))
            elif has_logprobs:
                response_logprobs.extend([0.0] * len(generated_ids))

            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))

            if parsed.kind == "answer":
                valid_action_count += 1
                break

            # A generation made after the budget is exhausted is answer-only.
            # Even if the model ignores the instruction, no tool is executed and
            # no additional observation is appended.
            if answer_only_turn:
                break

            tool_attempt_count += 1
            if parsed.kind == "tool" and parsed.tool_call is not None:
                observation_text, succeeded = await self._execute_tool(parsed.tool_call, image_refs)
                if succeeded:
                    valid_action_count += 1
                    if parsed.tool_call.name == "search":
                        valid_search_count += 1
                    else:
                        valid_vision_search_count += 1
                else:
                    tool_execution_failure_count += 1
            else:
                invalid_tool_call_count += 1
                observation_text = self._invalid_observation(parsed)

            if tool_attempt_count >= self.max_turns:
                observation_text = self._append_budget_exhausted_instruction(observation_text)
                tool_budget_exhausted_count = 1
                answer_only_turn = True

            observation_ids = await self._encode_tool_response(observation_text)
            remaining = self.response_length - len(response_ids)
            observation_ids = observation_ids[:remaining]
            if not observation_ids:
                break
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if has_logprobs:
                response_logprobs.extend([0.0] * len(observation_ids))

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = num_preempted if num_preempted else -1

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if has_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=max(1, assistant_turns),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "valid_action_stats": valid_action_count,
                "valid_search_stats": valid_search_count,
                "valid_vision_search_stats": valid_vision_search_count,
                "tool_attempt_stats": tool_attempt_count,
                "invalid_tool_call_stats": invalid_tool_call_count,
                "tool_execution_failure_stats": tool_execution_failure_count,
                "tool_budget_exhausted_stats": tool_budget_exhausted_count,
            },
        )
        return output

    async def _token_prefix_at_character_boundary(
        self, token_ids: list[int], decoded_text: str, end_offset: int
    ) -> list[int]:
        """Select an original-token prefix ending at a parsed action boundary.

        The previous implementation decoded, sliced text and re-encoded it.  A
        non-invertible tokenizer can then change token ids while the retained
        logprobs still refer to the old ids.  Binary searching decoded prefixes
        keeps ids and logprobs sourced from exactly the same generation.
        """

        if end_offset >= len(decoded_text):
            # Qwen/vLLM may retain an EOS / assistant-end id even though
            # ``skip_special_tokens=True`` removes it from ``decoded_text``.
            # Keep the complete original token sequence in this common case so
            # the assistant role boundary is never dropped.
            return token_ids
        target = decoded_text[:end_offset]

        def find_prefix() -> list[int]:
            low, high = 1, len(token_ids)
            candidate = len(token_ids)
            while low <= high:
                middle = (low + high) // 2
                prefix_text = self.tokenizer.decode(token_ids[:middle], skip_special_tokens=True)
                if len(prefix_text) >= end_offset:
                    candidate = middle
                    high = middle - 1
                else:
                    low = middle + 1
            prefix_text = self.tokenizer.decode(token_ids[:candidate], skip_special_tokens=True)
            if prefix_text[:end_offset] != target:
                # Decoding can be non-monotonic around byte-fallback tokens.  In
                # that rare case retaining all original ids is safer than pairing
                # re-tokenized ids with unrelated logprobs.
                return token_ids
            return token_ids[:candidate]

        return await self.loop.run_in_executor(None, find_prefix)

    async def _execute_tool(self, call: ToolCall, images: list[Any]) -> tuple[str, bool]:
        try:
            if call.name == "search":
                if not self.search_url:
                    raise RuntimeError("text retrieval endpoint is not configured")
                result = await self._batch_search([call.arguments["query"]])
                documents = result[0] if result else []
                return format_text_results(documents), True

            if not self.vision_search_url:
                raise RuntimeError("vision retrieval endpoint is not configured")
            request = self._build_vision_search_request(call, images)
            result = await self._batch_vision_search([request])
            documents = result[0] if result else []
            captions = format_vision_results(documents)
            prefix = f"image_index={request['image_index']}"
            return f"{prefix}:\n{captions}".rstrip(), True
        except Exception as exc:
            logger.warning("DualSearch tool %s failed: %s", call.name, exc)
            return f"Tool error ({call.name}): {sanitize_tool_response(exc)}", False

    def _invalid_observation(self, parsed: ParsedAssistantAction) -> str:
        detail = sanitize_tool_response(parsed.error or "invalid assistant action")
        return (
            f"Invalid tool call: {detail}. Use exactly one Qwen-native call such as "
            '<tool_call>{"name":"search","arguments":{"query":"example"}}</tool_call> '
            "or return the final answer inside <answer>...</answer>."
        )

    def _append_budget_exhausted_instruction(self, content: str) -> str:
        return (
            f"{content}\n\nTool-call budget exhausted. Do not call any more tools. "
            "Answer using the information already obtained and return the final answer "
            "inside <answer>...</answer>."
        ).strip()

    async def _encode_tool_response(self, content: str) -> list[int]:
        """Render a native role=tool response within the observation budget."""

        content = sanitize_tool_response(content)
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)
        content_ids = content_ids[: self.max_obs_length]

        async def render(candidate_ids: list[int]) -> list[int]:
            candidate = self.tokenizer.decode(candidate_ids, skip_special_tokens=True)
            return await self.apply_chat_template(
                [{"role": "tool", "content": candidate}],
                remove_system_prompt=True,
            )

        rendered = await render(content_ids)
        if len(rendered) <= self.max_obs_length:
            return rendered

        low, high = 0, len(content_ids)
        best: list[int] | None = None
        while low <= high:
            middle = (low + high) // 2
            candidate = await render(content_ids[:middle])
            if len(candidate) <= self.max_obs_length:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best
        # A pathological template whose empty tool wrapper alone exceeds the
        # configured budget cannot be safely truncated without losing closing
        # protocol tokens, so fail loudly instead of emitting malformed context.
        raise ValueError("max_obs_length is smaller than the native tool-response wrapper")

    async def _batch_search(self, queries: list[str]) -> list[list[dict[str, Any]]]:
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        return await self._post_retrieval_json(self.search_url, payload)

    async def _batch_vision_search(self, vision_requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        payload = {"queries": vision_requests, "topk": self.topk, "return_scores": True}
        return await self._post_retrieval_json(self.vision_search_url, payload)

    async def _post_retrieval_json(self, url: str, payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
        def post() -> Any:
            response = requests.post(url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or not isinstance(body.get("result"), list):
                raise RuntimeError("retrieval service returned an invalid response")
            return body["result"]

        return await self.loop.run_in_executor(None, post)

    def _build_vision_search_request(self, call: ToolCall, images: list[Any]) -> dict[str, Any]:
        image_index = call.arguments["image_index"]
        if image_index < 1 or image_index > len(images):
            raise ValueError(f"image_index {image_index} is out of range")
        return {
            "query": call.arguments["query"],
            "image_index": image_index,
            "image": self._jsonable_image(images[image_index - 1]),
        }

    def _extract_image_references(self, messages: list[dict[str, Any]]) -> list[Any]:
        image_refs = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, (list, tuple)):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                reference = item.get("image") or item.get("image_url") or item
                if isinstance(reference, dict) and isinstance(reference.get("url"), str):
                    reference = reference["url"]
                image_refs.append(reference)
        return image_refs

    def _normalize_images(self, images: Any) -> list[Any]:
        if images is None:
            return []
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    def _jsonable_image(self, image: Any) -> Any:
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        return image

    # Compatibility helpers retained for callers that format mocked retrieval
    # output directly.
    def _passages2string(self, retrieval_result: list[dict[str, Any]]) -> str:
        return format_text_results(retrieval_result)

    def _captions2string(self, retrieval_result: list[dict[str, Any]]) -> str:
        return format_vision_results(retrieval_result)
