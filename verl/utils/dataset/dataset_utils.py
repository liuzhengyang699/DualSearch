# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from enum import Enum
from typing import Any

import torch
from tensordict.tensorclass import NonTensorData

from verl.utils.tensordict_utils import nested_tensor_from_tensor_list


class DatasetPadMode(str, Enum):
    """Padding mode for dataset."""

    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    NO_PADDING = "no_padding"


class SFTTensorCollator:
    """Collate SFT samples, using NestedTensors for variable-length batches."""

    def __init__(self, pad_mode: DatasetPadMode | str = DatasetPadMode.LEFT_RIGHT):
        self.pad_mode = DatasetPadMode(pad_mode)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return self.collate_variable_batch(batch)
        if self.pad_mode in (DatasetPadMode.RIGHT, DatasetPadMode.LEFT_RIGHT):
            from torch.utils.data import default_collate

            return default_collate(batch)
        raise NotImplementedError(f"pad_mode {self.pad_mode} not implemented")

    def collate_variable_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        final_batch = {}
        tensor_keys = set().union(*(sample.keys() for sample in batch))

        for key in tensor_keys:
            if isinstance(batch[0][key], torch.Tensor):
                tensors = [sample[key] for sample in batch]
                if tensors[0].dim() >= 2:
                    final_batch[key] = nested_tensor_from_tensor_list(tensors)
                else:
                    final_batch[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            else:
                values = [NonTensorData(sample.get(key)) for sample in batch]
                final_batch[key] = torch.stack(values, dim=0)

        return final_batch
