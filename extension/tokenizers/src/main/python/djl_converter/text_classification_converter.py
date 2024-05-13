#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import logging
import math

import torch

from huggingface_converter import HuggingfaceConverter


class TextClassificationConverter(HuggingfaceConverter):

    def __init__(self):
        super().__init__()
        self.task = "text-classification"
        self.application = "nlp/text_classification"
        self.translator = "ai.djl.huggingface.translator.TextClassificationTranslatorFactory"
        self.inputs = "DJL is the best."
        self.outputs = None

    def verify_jit_output(self, hf_pipeline, encoding, out):
        config = hf_pipeline.model.config
        logits = out['logits'][0]

        if config.problem_type == "multi_label_classification" or config.num_labels == 1:
            logits = torch.sigmoid(logits)
        elif config.problem_type == "single_label_classification" or config.num_labels > 1:
            logits = torch.softmax(logits, dim=0)
        elif hasattr(config, "function_to_apply"):
            logging.error(
                f"Customized function not supported: {config.function_to_apply}"
            )
            return False, "Customized function not supported"

        index = logits.argmax().item()
        label = config.id2label[index]
        score = logits[index]
        pipeline_output = hf_pipeline(self.inputs)

        for item in pipeline_output:
            if item["label"] == label:
                if math.isclose(item["score"], score, abs_tol=1e-3):
                    return True, None
                break

        return False, f"Unexpected inference result"
