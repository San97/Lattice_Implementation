# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data.dataset import Dataset

from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.utils import logging

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", max_length=128, num_beams=None):
        self._max_length = max_length if max_length is not None else self._max_length
        self._num_beams = num_beams if num_beams is not None else self._num_beams
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only,
                                           ignore_keys=ignore_keys)

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        generation_args = {}
        if self._max_length is not None:
            generation_args["max_length"] = self._max_length
        elif hasattr(self.model.config, "max_length"):
            generation_args["max_length"] = self.model.config.max_length

        if self._num_beams is not None:
            generation_args["num_beams"] = self._num_beams
        elif hasattr(self.model.config, "num_beams"):
            generation_args["num_beams"] = self.model.config.num_beams

        for input_name in ["type_ids", "row_ids", "col_ids"]:
            if input_name in inputs:
                generation_args[input_name] = inputs[input_name]

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_args,
        )

        if generated_tokens.shape[-1] < generation_args.get("max_length", 128):
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, generation_args["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < generation_args.get("max_length", 128):
            labels = self._pad_tensors_to_max_len(labels, generation_args["max_length"])

        return (loss, generated_tokens, labels)
