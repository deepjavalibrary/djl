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
import os.path
import shutil

import torch
from huggingface_hub import hf_hub_download
from transformers import pipeline

from metadata import HuggingfaceMetadata
from shasum import sha1_sum
from zip_utils import zip_dir


class HuggingfaceConverter:

    def __init__(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.task = None
        self.application = None
        self.translator = None
        self.inputs = None
        self.outputs = None

    def save_model(self, model_id: str, output_dir: str, temp_dir: str):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        hf_pipeline = self.load_model(model_id)
        # Save tokenizer.json to temp dir
        self.save_tokenizer(hf_pipeline, temp_dir)

        # Save config.json just for reference
        config = hf_hub_download(repo_id=model_id, filename="config.json")
        shutil.copyfile(config, os.path.join(temp_dir, "config.json"))

        # Save jit traced .pt file to temp dir
        model_file = self.jit_trace_model(hf_pipeline, model_id, temp_dir)
        if not model_file:
            return False, "Failed to trace model", -1

        result, reason = self.verify_jit_model(hf_pipeline, model_file)
        if not result:
            return False, reason, -1

        size = self.save_to_model_zoo(model_id, output_dir, temp_dir,
                                      hf_pipeline.tokenizer.mask_token)

        return True, None, size

    @staticmethod
    def save_tokenizer(hf_pipeline, temp_dir: str):
        hf_pipeline.tokenizer.save_pretrained(temp_dir)
        # only keep tokenizer.json file
        for path in os.listdir(temp_dir):
            if path != "tokenizer.json":
                os.remove(os.path.join(temp_dir, path))

    def jit_trace_model(self, hf_pipeline, model_id: str, temp_dir: str):
        logging.info(f"Tracing model: {model_id} ...")
        encoding = self.encode_inputs(hf_pipeline.tokenizer)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # noinspection PyBroadException
        try:
            script_module = torch.jit.trace(hf_pipeline.model,
                                            (input_ids, attention_mask),
                                            strict=False)
        except Exception as ex:
            logging.info(
                f"Trace with attention_mask failed, model: {model_id}.")
            logging.debug(ex, exc_info=False)
            try:
                script_module = torch.jit.trace(hf_pipeline.model,
                                                input_ids,
                                                strict=False)
            except Exception as e:
                logging.warning(f"Failed to trace model: {model_id}.")
                logging.warning(e, exc_info=True)
                return None

        model_name = model_id.split("/")[-1]
        logging.info(f"Saving torchscript model: {model_name}.pt ...")
        model_file = os.path.join(temp_dir, f"{model_name}.pt")
        script_module.save(model_file)
        return model_file

    def save_to_model_zoo(self, model_id: str, output_dir: str, temp_dir: str,
                          mask_token: str):
        artifact_ids = model_id.split("/")
        model_name = artifact_ids[-1]

        repo_dir = f"{output_dir}/model/{self.application}/ai/djl/huggingface/{model_id}"
        model_dir = f"{repo_dir}/0.0.1"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save serving.properties
        serving_file = os.path.join(temp_dir, "serving.properties")
        with open(serving_file, 'w') as f:
            f.write(f"engine=PyTorch\n"
                    f"option.modelName={model_name}\n"
                    f"option.mapLocation=true\n"
                    f"translatorFactory={self.translator}\n")
            if mask_token:
                f.write(f"maskToken={mask_token}\n")

        # Save model as .zip file
        logging.info(f"Saving DJL model as zip: {model_name}.zip ...")
        zip_file = os.path.join(model_dir, f"{model_name}.zip")
        zip_dir(temp_dir, zip_file)

        # Save metadata.json
        sha1 = sha1_sum(zip_file)
        file_size = os.path.getsize(zip_file)
        metadata = HuggingfaceMetadata(artifact_ids, self.application,
                                       self.translator, sha1, file_size)
        metadata_file = os.path.join(repo_dir, "metadata.json")
        metadata.save_metadata(metadata_file)

        return file_size

    def verify_jit_model(self, hf_pipeline, model_file: str):
        logging.info(f"Verifying torchscript model: {model_file} ...")

        tokenizer = hf_pipeline.tokenizer
        encoding = self.encode_inputs(tokenizer)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        if torch.cuda.is_available():
            traced_model = torch.jit.load(model_file, map_location='cuda:0')
            traced_model.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
        else:
            traced_model = torch.jit.load(model_file)

        traced_model.eval()

        # test traced model
        out = traced_model(input_ids, attention_mask)

        return self.verify_jit_output(hf_pipeline, encoding, out)

    def verify_jit_output(self, hf_pipeline, encoding, out):
        if not hasattr(out, "last_hidden_layer"):
            return False, f"Unexpected inference result: {out}"

        return True, None

    def load_model(self, model_id: str):
        logging.info(f"Loading model: {model_id}.")
        kwargs = {
            "tokenizer": model_id,
            "device": -1  # always use CPU to trace the model
        }
        return pipeline(task=self.task,
                        model=model_id,
                        framework="pt",
                        **kwargs)

    def encode_inputs(self, tokenizer):
        return tokenizer.encode_plus(self.inputs, return_tensors='pt')
