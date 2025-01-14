#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import math

from djl_converter.huggingface_converter import HuggingfaceConverter


class ZeroShotClassificationConverter(HuggingfaceConverter):

    def __init__(self):
        super().__init__()
        self.task = "zero-shot-classification"
        self.application = "nlp/zero_shot_classification"
        self.translator = "ai.djl.huggingface.translator.ZeroShotClassificationTranslatorFactory"
        self.inputs = "one day I will see the world"
        self.labels = ['travel']

    def encode_inputs(self, tokenizer):
        return tokenizer(self.inputs,
                         f"This example is {self.labels[0]}.",
                         return_tensors='pt')

    def verify_jit_output(self, hf_pipeline, encoding, out):
        logits = out['logits']
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        score = probs[:, 1].item()

        pipeline_output = hf_pipeline(self.inputs, self.labels)
        expected = pipeline_output["scores"][0]

        if math.isclose(expected, score, abs_tol=1e-3):
            return True, None

        return False, f"Unexpected inference result"

    def get_extra_arguments(self, hf_pipeline, model_id: str,
                            temp_dir: str) -> dict:
        return {
            "padding": "true",
            "truncation": "only_first",
        }
