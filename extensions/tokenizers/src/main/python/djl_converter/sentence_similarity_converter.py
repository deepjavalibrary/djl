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
import shutil

import requests
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from djl_converter.huggingface_converter import HuggingfaceConverter, PipelineHolder
from huggingface_hub import hf_hub_download


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

    def get_extra_arguments(self, hf_pipeline, model_id: str,
                            temp_dir: str) -> dict:
        args = {"padding": "true"}
        for config_name in [
                'sentence_bert_config.json', 'sentence_roberta_config.json',
                'sentence_distilbert_config.json',
                'sentence_camembert_config.json',
                'sentence_albert_config.json',
                'sentence_xlm-roberta_config.json',
                'sentence_xlnet_config.json'
        ]:
            try:
                file = hf_hub_download(repo_id=model_id, filename=config_name)
                with open(file) as f:
                    config = json.load(f)
                    if config.get("max_seq_length"):
                        args["maxLength"] = config.get("max_seq_length")
                    if config.get("do_lower_case"):
                        args["doLowerCase"] = config.get("do_lower_case")

                break
            except requests.exceptions.HTTPError:
                pass

        if not "maxLength" in args:
            if hasattr(hf_pipeline.model, "config"):
                config = hf_pipeline.model.config
            else:
                config = AutoConfig.from_pretrained(model_id)
            tokenizer = hf_pipeline.tokenizer
            if hasattr(config, "max_position_embeddings") and hasattr(
                    tokenizer, "model_max_length"):
                max_seq_length = min(config.max_position_embeddings,
                                     tokenizer.model_max_length)
                args["maxLength"] = str(max_seq_length)

        pooling_path = None
        dense_path = None
        layer_norm_path = None
        normalize = False
        try:
            file = hf_hub_download(repo_id=model_id, filename="modules.json")
            with open(file, "r") as f:
                modules = json.load(f)

            for module in modules:
                module_type = module.get("type")
                if module_type == "sentence_transformers.models.Pooling":
                    pooling_path = module["path"]
                elif module_type == "sentence_transformers.models.Dense":
                    dense_path = module["path"]
                elif module_type == "sentence_transformers.models.LayerNorm":
                    layer_norm_path = module["path"]
                elif module_type == "sentence_transformers.models.Normalize":
                    normalize = "true"
                elif module_type != "sentence_transformers.models.Transformer":
                    logging.warning(f"Unexpected module: {module_type}.")
        except requests.exceptions.HTTPError:
            logging.warning(f"{model_id}: modules.json not found.")

        if pooling_path:
            try:
                file = hf_hub_download(repo_id=model_id,
                                       filename=f"{pooling_path}/config.json")
                if os.path.exists(file):
                    with open(file, "r") as f:
                        pooling = json.load(f)
                        if pooling.get("pooling_mode_cls_token"):
                            args["pooling"] = "cls"
                        elif pooling.get("pooling_mode_max_tokens"):
                            args["pooling"] = "max"
                        elif pooling.get("pooling_mode_mean_sqrt_len_tokens"):
                            args["pooling"] = "mean_sqrt_len"
                        elif pooling.get("pooling_mode_weightedmean_tokens"):
                            args["pooling"] = "weightedmean"
                        elif pooling.get("pooling_mode_lasttoken"):
                            args["pooling"] = "lasttoken"
            except requests.exceptions.HTTPError:
                logging.warning(
                    f"{model_id}: {pooling_path}/config.json not found.")

        if dense_path:
            try:
                file = hf_hub_download(repo_id=model_id,
                                       filename=f"{dense_path}/config.json")
                with open(file, "r") as f:
                    dense = json.load(f)
                    activation = dense.get("activation_function")
                    if activation == "torch.nn.modules.activation.Tanh":
                        args["denseActivation"] = "Tanh"
                    elif activation != "torch.nn.modules.linear.Identity":
                        logging.warning(
                            f"Unexpected activation function: {activation}.")
                self.save_module_weight(model_id, temp_dir, dense_path,
                                        "linear")
                args["dense"] = "linear.safetensors"
            except requests.exceptions.HTTPError:
                logging.debug(f"{model_id}: {dense_path} not found.")

        if layer_norm_path:
            try:
                self.save_module_weight(model_id, temp_dir, layer_norm_path,
                                        "norm")
                args["layerNorm"] = "norm.safetensors"
            except requests.exceptions.HTTPError:
                logging.warning(f"{model_id}: {layer_norm_path} not found.")

        if not normalize:
            args["normalize"] = "false"

        return args

    @staticmethod
    def save_module_weight(model_id: str, temp_dir: str, layer: str,
                           name: str):
        file = hf_hub_download(repo_id=model_id,
                               filename=f"{layer}/model.safetensors")
        shutil.copyfile(file, os.path.join(temp_dir, f"{name}.safetensors"))
