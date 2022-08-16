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
from transformers import AutoTokenizer, AutoModel

from huggingface_converter import HuggingfaceConverter


class PipelineHolder(object):

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model


class SentenceSimilarityConverter(HuggingfaceConverter):

    def __init__(self):
        super().__init__()
        self.task = "sentence-similarity"
        self.application = "nlp/text_embedding"
        self.translator = "ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory"
        self.inputs = "This is an example sentence"
        self.outputs = 0

    def load_model(self, model_id: str):
        logging.info(f"Loading model: {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        return PipelineHolder(tokenizer, model)

    def verify_jit_output(self, hf_pipeline, encoding, out):
        last_hidden_state = out["last_hidden_state"]

        pipeline_output = hf_pipeline.model(**encoding)
        if not torch.allclose(pipeline_output.last_hidden_state,
                              last_hidden_state):
            return False, f"Unexpected inference result: {last_hidden_state}"

        return True, None
