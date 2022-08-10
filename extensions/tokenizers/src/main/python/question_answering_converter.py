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

import torch

from huggingface_converter import HuggingfaceConverter


class QuestionAnsweringConverter(HuggingfaceConverter):

    def __init__(self):
        super().__init__()
        self.task = "question-answering"
        self.application = "nlp/question_answer"
        self.translator = "ai.djl.huggingface.translator.QuestionAnsweringTranslatorFactory"
        self.inputs = {
            "question":
            "When did BBC Japan start broadcasting?",
            "context":
            "BBC Japan was a general entertainment Channel. Which operated between December 2004 and April 2006."
            " It ceased operations after its Japanese distributor folded."
        }
        self.outputs = "december 2004"

    def verify_jit_output(self, hf_pipeline, encoding, out):
        tokenizer = hf_pipeline.tokenizer
        input_ids = encoding["input_ids"]

        start_ = out["start_logits"]
        end_ = out["end_logits"]
        start_[0, 0] = 0
        end_[0, 0] = 0
        answer_start = torch.argmax(start_)
        answer_end = torch.argmax(end_) + 1

        out_ids = input_ids[0].tolist()[answer_start:answer_end]
        tokens = tokenizer.convert_ids_to_tokens(out_ids)
        prediction = tokenizer.convert_tokens_to_string(tokens).strip()

        if prediction.lower() != self.outputs:
            pipeline_output = hf_pipeline(self.inputs)
            if pipeline_output != prediction:
                return False, f"Unexpected inference result: {prediction}"
            else:
                logging.warning(
                    f"pipeline output differs from expected: {pipeline_output}"
                )

        return True, None

    def encode_inputs(self, tokenizer):
        text = self.inputs["question"]
        text_pair = self.inputs["context"]
        return tokenizer.encode_plus(text,
                                     text_pair=text_pair,
                                     return_tensors='pt')
