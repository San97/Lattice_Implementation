# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_base import PaddingStrategy


class CustomT5TokenizerFast(T5TokenizerFast):
    def _pad(self, encoded_inputs, max_length=None, padding_strategy=PaddingStrategy.DO_NOT_PAD,
             pad_to_multiple_of=None, return_attention_mask=None) -> dict:
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        input_ids = encoded_inputs[self.model_input_names[0]]
        input_length = len(input_ids)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = input_length

        if max_length is not None and pad_to_multiple_of is not None:
            max_length = ((max_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        needs_padding = padding_strategy != PaddingStrategy.DO_NOT_PAD and input_length != max_length

        if needs_padding:
            padding_length = max_length - input_length
            if self.padding_side == "right":
                padding_id = self.pad_token_id
                input_ids += [padding_id] * padding_length
                if return_attention_mask:
                    attention_mask = [1] * input_length + [0] * padding_length
                    encoded_inputs["attention_mask"] = attention_mask
                if "token_type_ids" in encoded_inputs:
                    padding_type_id = self.pad_token_type_id
                    token_type_ids = encoded_inputs["token_type_ids"] + [padding_type_id] * padding_length
                    encoded_inputs["token_type_ids"] = token_type_ids
                if "special_tokens_mask" in encoded_inputs:
                    special_tokens_mask = encoded_inputs["special_tokens_mask"] + [1] * padding_length
                    encoded_inputs["special_tokens_mask"] = special_tokens_mask
                for name in ["type_ids", "row_ids", "col_ids"]:
                    if name in encoded_inputs:
                        encoded_inputs[name] += [0] * padding_length
            else:
                raise ValueError(f"Invalid padding strategy: {self.padding_side}")
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * input_length

        encoded_inputs[self.model_input_names[0]] = input_ids

        return encoded_inputs
