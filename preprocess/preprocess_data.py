# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
# Lint as: python3
"""Augments json files with table linearization used by baselines.

Note that this code is merely meant to be starting point for research and
there may be much better table representations for this task.
"""
import copy
import json
from tqdm import tqdm

import six

from table_linearization import table_do_linearize

input_path = None
output_path = None
examples_to_visualize = 100


def set_flags(input_path_value, output_path_value, examples_to_visualize_value):
    global input_path, output_path, examples_to_visualize
    input_path = input_path_value
    output_path = output_path_value
    examples_to_visualize = examples_to_visualize_value
set_flags("<input_path>", "<output_path>", 100)


def main(input_path: str, output_path: str):
    processed_examples = examples_fetch(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in processed_examples:
            f.write(json.dumps(example) + '\n')

def examples_fetch(input_path):
    processed = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            line = six.ensure_text(line, "utf-8")
    example = json.loads(line)
    table = example["table"]
    page_title = example["table_page_title"]
    section_title = example["table_section_title"]
    cell_indices = example["highlighted_cells"]

    table_metadata, type_ids, row_ids, col_ids = table_do_linearize(
        input_table=table,
        indices=cell_indices,
        name=page_title,
        section_name=section_title,
        ordered=True
    )

    processed_example = copy.deepcopy(example)
    processed_example["subtable_metadata"] = table_metadata
    processed_example["type_ids"] = " ".join([str(x) for x in type_ids])
    processed_example["row_ids"] = " ".join([str(x) for x in row_ids])
    processed_example["col_ids"] = " ".join([str(x) for x in col_ids])

    processed.append(processed_example)
    print(f"Processed {len(processed)} examples")

    return processed


