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
import json
import logging
import os

import requests
import torch
from transformers import AutoTokenizer, AutoModel

from huggingface_converter import HuggingfaceConverter
from huggingface_hub import hf_hub_download


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
        last_hidden_state = out["last_hidden_state"].to("cpu")

        pipeline_output = hf_pipeline.model(**encoding)
        expected = pipeline_output.last_hidden_state
        if not torch.allclose(
                expected, last_hidden_state, atol=1e-05, rtol=1e-03):
            return False, "Unexpected inference result"

        return True, None

    def get_extra_arguments(self, hf_pipeline, model_id: str) -> dict:
        args = {"padding": "true"}
        try:
            file = hf_hub_download(repo_id=model_id,
                                   filename="1_Pooling/config.json")
            if os.path.exists(file):
                with open(file, "r") as f:
                    pooling = json.load(f)
                    if pooling.get("pooling_mode_cls_token"):
                        args["pooling"] = "cls_token"
                    elif pooling.get("pooling_mode_max_tokens"):
                        args["pooling"] = "max_tokens"
                    elif pooling.get("pooling_mode_mean_sqrt_len_tokens"):
                        args["pooling"] = "mean_sqrt_len_tokens"
                    elif pooling.get("pooling_mode_weightedmean_tokens"):
                        args["pooling"] = "weightedmean_tokens"
                    elif pooling.get("pooling_mode_lasttoken"):
                        args["pooling"] = "lasttoken"
        except requests.exceptions.HTTPError:
            logging.warning(f"{model_id}: 1_Pooling/config.json not found.")

        return args
