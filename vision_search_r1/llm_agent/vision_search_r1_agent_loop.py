import base64
import io
import logging
import os
import re
from typing import Any
from uuid import uuid4

import requests
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VisionSearchR1AgentLoop(AgentLoopBase):
    """Vision-Search-R1 XML-tag agent loop for text and image retrieval."""

    ACTION_PATTERN = re.compile(r"<(search|vision_search|answer)>(.*?)</\1>", re.DOTALL)
    IMAGE_ASSIGNMENT_PATTERN = re.compile(r"(?<!\w)image\s*=")
    IMAGE_VALUE_PATTERN = re.compile(r"(?<!\w)image\s*=\s*(\d+)(?=$|[^\w])")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        retriever_config = self.config.get("retriever", {})
        self.search_url = retriever_config.get("url")
        self.vision_search_url = retriever_config.get("vision_search_url")
        self.topk = retriever_config.get("topk", 3)
        self.request_timeout = retriever_config.get("timeout", 60)
        self.max_turns = self.config.get("max_turns", 2)
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
        image_refs = self._extract_image_references(messages) or self._normalize_images(images)

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        metrics = {}
        request_id = uuid4().hex
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        routed_experts = None
        valid_action_count = 0
        valid_search_count = 0
        valid_vision_search_count = 0
        num_preempted = 0

        for turn_idx in range(self.max_turns + 1):
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

            num_preempted += output.num_preempted if output.num_preempted is not None else 0
            if routed_experts is None and output.routed_experts is not None:
                routed_experts = output.routed_experts

            action_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            )
            truncated_text, action, content = self._truncate_to_first_action(action_text)
            current_response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(truncated_text, add_special_tokens=False)
            )
            current_response_ids = current_response_ids[:remaining]
            if not current_response_ids:
                break

            response_ids.extend(current_response_ids)
            response_mask.extend([1] * len(current_response_ids))
            if output.log_probs and len(current_response_ids) <= len(output.log_probs):
                response_logprobs.extend(output.log_probs[: len(current_response_ids)])
            elif response_logprobs:
                response_logprobs.extend([0.0] * len(current_response_ids))

            if action == "answer":
                valid_action_count += 1
                break

            if turn_idx >= self.max_turns:
                break

            observation_text = await self._build_observation(action, content, image_refs)
            if action in {"search", "vision_search"} and not observation_text.startswith("\nMy previous action is invalid."):
                valid_action_count += 1
                if action == "search":
                    valid_search_count += 1
                else:
                    valid_vision_search_count += 1

            observation_ids = self._encode_observation(observation_text, tag=None)
            remaining = self.response_length - len(response_ids)
            observation_ids = observation_ids[:remaining]
            if not observation_ids:
                break
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if response_logprobs:
                response_logprobs.extend([0.0] * len(observation_ids))

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = num_preempted if num_preempted else -1

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length] if routed_experts is not None else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=max(1, valid_action_count + 1),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "valid_action_stats": valid_action_count,
                "valid_search_stats": valid_search_count,
                "valid_vision_search_stats": valid_vision_search_count,
            },
        )
        return output

    def _truncate_to_first_action(self, response: str) -> tuple[str, str | None, str]:
        match = self.ACTION_PATTERN.search(response)
        if not match:
            return response, None, ""
        return response[: match.end()], match.group(1), match.group(2).strip()

    async def _build_observation(self, action: str | None, content: str, images: list[Any]) -> str:
        if action == "search":
            if not self.search_url:
                return self._invalid_observation()
            result = await self._batch_search([content])
            information = self._passages2string(result[0]) if result else ""
            return self._wrap_tagged_observation("information", information)

        if action == "vision_search":
            if not self.vision_search_url:
                return self._invalid_observation()
            request = self._build_vision_search_request(content, images)
            if request is None:
                return self._invalid_observation()
            result = await self._batch_vision_search([request])
            captions = self._captions2string(result[0]) if result else ""
            return self._wrap_tagged_observation(
                "vision_information",
                f"image={request['image_index']}: {captions}".strip(),
            )

        return self._invalid_observation()

    def _invalid_observation(self) -> str:
        return (
            "\nMy previous action is invalid. "
            "If I want to search, I should put the query between <search> and </search>. "
            "If I want to search an input image, I should put exactly one image=N reference between "
            "<vision_search> and </vision_search>. "
            "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
            "Let me try again.\n"
        )

    def _wrap_tagged_observation(self, tag: str, content: str) -> str:
        prefix = f"\n\n<{tag}>"
        suffix = f"</{tag}>\n\n"
        return prefix + self._truncate_content_to_observation_budget(prefix, content, suffix) + suffix

    def _truncate_content_to_observation_budget(self, prefix: str, content: str, suffix: str) -> str:
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        budget = max(0, self.max_obs_length - len(prefix_ids) - len(suffix_ids))
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)
        if len(content_ids) <= budget:
            return content
        return self.tokenizer.decode(content_ids[:budget], skip_special_tokens=True)

    def _encode_observation(self, observation: str, tag: str | None) -> list[int]:
        if tag is not None:
            observation = self._wrap_tagged_observation(tag, observation)
        ids = self.tokenizer.encode(observation, add_special_tokens=False)
        if len(ids) > self.max_obs_length and tag is None:
            ids = ids[: self.max_obs_length]
        return ids

    async def _batch_search(self, queries: list[str]) -> list[list[dict[str, Any]]]:
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        response = await self.loop.run_in_executor(
            None,
            lambda: requests.post(self.search_url, json=payload, timeout=self.request_timeout).json(),
        )
        return response["result"]

    async def _batch_vision_search(self, vision_requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        payload = {"queries": vision_requests, "topk": self.topk, "return_scores": True}
        response = await self.loop.run_in_executor(
            None,
            lambda: requests.post(self.vision_search_url, json=payload, timeout=self.request_timeout).json(),
        )
        return response["result"]

    def _build_vision_search_request(self, content: str, images: list[Any]) -> dict[str, Any] | None:
        assignments = self.IMAGE_ASSIGNMENT_PATTERN.findall(content)
        matches = self.IMAGE_VALUE_PATTERN.findall(content)
        if len(assignments) != 1 or len(matches) != 1:
            return None

        image_index = int(matches[0])
        if image_index < 1 or image_index > len(images):
            return None

        return {
            "query": content,
            "image_index": image_index,
            "image": self._jsonable_image(images[image_index - 1]),
        }

    def _extract_image_references(self, messages: list[dict[str, Any]]) -> list[Any]:
        image_refs = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                image_refs.append(item.get("image") or item.get("image_url") or item)
        return image_refs

    def _normalize_images(self, images: Any) -> list[Any]:
        if images is None:
            return []
        if isinstance(images, list | tuple):
            return list(images)
        return [images]

    def _jsonable_image(self, image: Any) -> Any:
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
        return image

    def _passages2string(self, retrieval_result: list[dict[str, Any]]) -> str:
        lines = []
        for idx, doc_item in enumerate(retrieval_result):
            document = doc_item.get("document", doc_item)
            content = document.get("contents", "")
            title, text = self._split_title_text(content)
            lines.append(f"Doc {idx + 1}(Title: {title}) {text}")
        return "\n".join(lines)

    def _captions2string(self, retrieval_result: list[dict[str, Any]]) -> str:
        lines = []
        for idx, doc_item in enumerate(retrieval_result):
            document = doc_item.get("document", doc_item)
            content = document.get("contents", "")
            title, text = self._split_title_text(content)
            lines.append(f"Caption {idx + 1}(Title: {title}) {text}")
        return "\n".join(lines)

    def _split_title_text(self, content: str) -> tuple[str, str]:
        if not content:
            return "", ""
        parts = content.split("\n")
        title = parts[0].strip().strip('"')
        text = "\n".join(parts[1:]).strip()
        return title, text
