import torch
import re
from collections import defaultdict
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    vision_search_url: str = None
    topk: int = 3
    image_key: str = "images"

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        self.sequence_keys = {'input_ids', 'attention_mask', 'position_ids'}

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _truncate_to_first_action(self, response: str) -> str:
        """Keep text through the first complete tool/answer tag."""
        pattern = r'<(search|vision_search|answer)>(.*?)</\1>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return response[:match.end()]
        return response

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [self._truncate_to_first_action(resp) for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        preserved_tensors = {k: v for k, v in rollings.batch.items() if k not in self.sequence_keys}
        rolling_tensors = {
                'input_ids': new_input_ids[:, -max_len:],
                'position_ids': new_position_ids[:, -max_len:],
                'attention_mask': new_attention_mask[:, -max_len:]
        }
        rolling_tensors.update(preserved_tensors)
        new_rollings = DataProto.from_dict(
            rolling_tensors,
            non_tensors=rollings.non_tensor_batch,
        )
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _slice_active_batch(self, batch: DataProto, active_mask: torch.Tensor) -> DataProto:
        """Slice tensor and non-tensor fields to active examples."""
        mask_np = active_mask.detach().cpu().numpy()
        active_batch = DataProto.from_dict(
            tensors={k: v[active_mask] for k, v in batch.batch.items()},
            non_tensors={k: v[mask_np] for k, v in batch.non_tensor_batch.items()},
        )
        active_batch.meta_info.update(batch.meta_info)
        return active_batch

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            if not torch.is_floating_point(active_batch.batch[key]):
                active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_non_tensors = {}
        for k, v in active_batch.non_tensor_batch.items():
            pad_values = np.repeat(v[0:1], padding_size, axis=0)
            padded_non_tensors[k] = np.concatenate([v, pad_values], axis=0)

        padded_active_batch = DataProto.from_dict(padded_batch, non_tensors=padded_non_tensors)
        for key in padded_active_batch.batch.keys():
            if not torch.is_floating_point(padded_active_batch.batch[key]):
                padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        trimmed_non_tensors = {k: v[:-padding_size] for k, v in padded_output.non_tensor_batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        padded_output.non_tensor_batch = trimmed_non_tensors
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_vision_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = self._slice_active_batch(rollings, active_mask)
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, is_vision_search = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                image_batches=rollings.non_tensor_batch.get(self.config.image_key),
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            valid_vision_search_stats += torch.tensor(is_vision_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = self._slice_active_batch(rollings, active_mask)
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search, is_vision_search = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                do_search=False,
                image_batches=rollings.non_tensor_batch.get(self.config.image_key),
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            valid_vision_search_stats += torch.tensor(is_vision_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['valid_vision_search_stats'] = valid_vision_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            gen_batch.non_tensor_batch,
            {k: v for k, v in gen_batch.batch.items() if k not in self.sequence_keys},
        )

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            non_tensor_batch: Dict = None,
                            preserved_tensors: Dict[str, torch.Tensor] = None) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        if preserved_tensors:
            final_output.update(preserved_tensors)
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search=True,
        image_batches=None,
    ) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        if active_mask is None:
            active_mask = [True] * len(predictions)
        active_flags = [bool(active) for active in active_mask]

        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search, is_vision_search = [], [], [], [], []
        
        search_queries = [
            content for action, content, active in zip(cur_actions, contents, active_flags)
            if action == 'search' and active
        ]
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == len(search_queries)
        else:
            search_results = [''] * len(search_queries)

        vision_requests_by_index = {}
        vision_requests = []
        for i, (action, content, active) in enumerate(zip(cur_actions, contents, active_flags)):
            if action != 'vision_search' or not active:
                continue
            images = image_batches[i] if image_batches is not None else None
            request = self._build_vision_search_request(content, images, i)
            if request is None:
                continue
            vision_requests_by_index[i] = request
            vision_requests.append(request)

        if do_search:
            vision_search_results = self.batch_vision_search(vision_requests)
            assert len(vision_search_results) == len(vision_requests)
        else:
            vision_search_results = [''] * len(vision_requests)

        for i, (action, active) in enumerate(zip(cur_actions, active_flags)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                is_vision_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    is_vision_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    is_vision_search.append(0)
                elif action == 'vision_search' and i in vision_requests_by_index:
                    result = vision_search_results.pop(0).strip()
                    next_obs.append(
                        f'\n\n<vision_information>{result}</vision_information>\n\n'
                    )
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(0)
                    is_vision_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to search an input image, I should put exactly one image=N reference between <vision_search> and </vision_search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    is_vision_search.append(0)
            
        assert len(search_results) == 0
        assert len(vision_search_results) == 0
            
        return next_obs, dones, valid_action, is_search, is_vision_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|vision_search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def _normalize_images(self, images: Any) -> List[Any]:
        if images is None:
            return []
        if isinstance(images, np.ndarray):
            images = images.tolist()
        if isinstance(images, (list, tuple)):
            return list(images)
        return [images]

    def _build_vision_search_request(self, content: str, images: Any, sample_index: int) -> Dict[str, Any]:
        assignment_pattern = r'(?<!\w)image\s*='
        value_pattern = r'(?<!\w)image\s*=\s*(\d+)(?=$|[^\w])'
        assignments = re.findall(assignment_pattern, content)
        matches = re.findall(value_pattern, content)
        if len(assignments) != 1 or len(matches) != 1:
            return None

        image_index = int(matches[0])
        image_list = self._normalize_images(images)
        if image_index < 1 or image_index > len(image_list):
            return None

        return {
            'sample_index': sample_index,
            'query': content,
            'image_index': image_index,
            'image': image_list[image_index - 1],
            'images': image_list,
        }

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        if not queries:
            return []
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def batch_vision_search(self, vision_requests: List[Dict[str, Any]] = None) -> List[str]:
        """
        Batchified image retrieval hook.
        Each request contains:
        sample_index, query, image_index (1-based), image, and images.
        """
        if not vision_requests:
            return []
        results = self._batch_vision_search(vision_requests)['result']

        return [self._captions2string(result) for result in results]

    def _batch_vision_search(self, vision_requests: List[Dict[str, Any]]):

        payload = {
            "queries": vision_requests,
            "topk": self.config.topk,
            "return_scores": True
        }

        return requests.post(self.config.vision_search_url, json=payload).json()

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def _captions2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):

            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Caption {idx+1}(Title: {title}) {text}\n"

        return format_reference
