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
import sys
from argparse import Namespace

import onnx
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

from djl_converter.safetensors_convert import convert_file
import torch
from huggingface_hub import hf_hub_download, HfApi
from transformers import pipeline, AutoTokenizer, AutoConfig

from djl_converter.metadata import HuggingfaceMetadata
from djl_converter.shasum import sha1_sum
from djl_converter.zip_utils import zip_dir


class PipelineHolder(object):

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model


class ModelHolder(object):

    def __init__(self, config):
        self.config = config


class ModelWrapper(nn.Module):

    def __init__(self, model, include_types: bool) -> None:
        super().__init__()
        self.model = model
        self.include_types = include_types

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None):
        if self.include_types:
            output = self.model(input_ids, attention_mask)
        else:
            output = self.model(input_ids, attention_mask, token_type_ids)
        if isinstance(output, TokenClassifierOutput):
            # TokenClassifierOutput may contains mix of Tensor and Tuple(Tensor)
            return {"logits": output["logits"]}

        return output


class HuggingfaceConverter:

    def __init__(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.task = None
        self.application = None
        self.translator = None
        self.inputs = None
        self.outputs = None
        self.max_model_size = int(os.getenv("MAX_MODEL_SIZE", "2_000_000_000"))
        self.api = HfApi()

    def save_model(self, model_info, task: str, args: Namespace, temp_dir: str,
                   model_zoo: bool):
        if args.output_format == "OnnxRuntime":
            return self.save_onnx_model(model_info, task, args, temp_dir,
                                        model_zoo)
        elif args.output_format == "Rust":
            return self.save_rust_model(model_info, args, temp_dir, model_zoo)
        else:
            return self.save_pytorch_model(model_info, args, temp_dir,
                                           model_zoo)

    def save_onnx_model(self, model_info, task: str, args: Namespace,
                        temp_dir: str, model_zoo: bool):
        model_id = model_info.modelId

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        logging.info(f"Saving onnxruntime model: {model_id} ...")

        from optimum.commands.optimum_cli import main

        sys.argv = ["model_zoo_importer.py", "export", "onnx", "-m", model_id]
        if args.optimize:
            sys.argv.extend(["--optimize", args.optimize])
        if args.device:
            sys.argv.extend(["--device", args.device])
        if args.dtype:
            sys.argv.extend(["--dtype", args.dtype])
        if args.trust_remote_code:
            sys.argv.append("--trust-remote-code")
        if task:
            sys.argv.extend(["--task", task])
        sys.argv.append(temp_dir)

        main()

        model = onnx.load_model(os.path.join(temp_dir, "model.onnx"),
                                load_external_data=False)
        inputs = repr(model.graph.input)
        include_types = "token_type_id" in inputs

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code)
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code)
        hf_pipeline = PipelineHolder(tokenizer, ModelHolder(config))
        arguments = self.save_serving_properties(model_info, "OnnxRuntime",
                                                 temp_dir, hf_pipeline,
                                                 include_types)
        if model_zoo:
            model_size = self.get_dir_size(temp_dir)
            if model_size > self.max_model_size:
                return False, f"Model size too large: {model_size}", -1
            size = self.save_to_model_zoo(model_info, args.output_dir,
                                          "OnnxRuntime", temp_dir, arguments)
        else:
            size = -1

        return True, None, size

    def save_rust_model(self, model_info, args: Namespace, temp_dir: str,
                        model_zoo: bool):
        model_id = model_info.modelId

        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=args.trust_remote_code)
        if hasattr(config, "model_type"):
            # TODO: Monitor if "new" model_type will change or if there is conflict
            if config.model_type not in [
                    "bert", "camembert", "distilbert", "xlm-roberta",
                    "roberta", "nomic_bert", "mistral", "qwen2", "new",
                    "gemma2"
            ]:
                return False, f"Unsupported model_type: {config.model_type}", -1
        else:
            return False, f"Unknown model_type: {model_id}", -1

        logging.info(f"Saving rust model: {model_id} ...")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        include_types = config.model_type not in [
            "distilbert", "mistral", "qwen2", "gemma2"
        ]
        hf_pipeline = PipelineHolder(tokenizer, ModelHolder(config))
        try:
            # Save tokenizer.json to temp dir
            self.save_tokenizer(hf_pipeline, temp_dir)
        except Exception as e:
            logging.warning(f"Failed to save tokenizer: {model_id}.")
            logging.warning(e, exc_info=True)
            return False, "Failed to save tokenizer", -1

        # Save config.json
        config_file = self.get_file(model_id, "config.json")
        shutil.copyfile(config_file, os.path.join(temp_dir, "config.json"))

        target = os.path.join(temp_dir, "model.safetensors")
        if os.path.exists(model_id):
            file = os.path.join(model_id, "model.safetensors")
            if os.path.exists(file):
                shutil.copyfile(file, target)
            else:
                file = os.path.join(model_id, "pytorch_model.bin")
                if os.path.exists(file):
                    convert_file(file, target)
                else:
                    return False, f"No model file found for: {model_id}", -1
        else:
            model = self.api.model_info(model_id, files_metadata=True)
            sf_files = []
            pt_files = []
            for sibling in model.siblings:
                if sibling.rfilename.endswith(".safetensors"):
                    sf_files.append(sibling.rfilename)
                elif sibling.rfilename == "pytorch_model.bin":
                    pt_files.append(sibling.rfilename)

            if sf_files:
                for f in sf_files:
                    file = hf_hub_download(repo_id=model_id, filename=f)
                    shutil.copyfile(file, os.path.join(temp_dir, f))
            elif pt_files:
                for f in pt_files:
                    # Change file name from pytorch_model*.bin to model*.safetensors
                    target = f.replace("pytorch_model", "model").replace(
                        ".bin", ".safetensors")
                    file = hf_hub_download(repo_id=model_id, filename=f)
                    convert_file(file, os.path.join(temp_dir, target))
            else:
                return False, f"No model file found for: {model_id}", -1

        arguments = self.save_serving_properties(model_info, "Rust", temp_dir,
                                                 hf_pipeline, include_types)
        if model_zoo:
            model_size = self.get_dir_size(temp_dir)
            if model_size > self.max_model_size:
                return False, f"Model size too large: {model_size}", -1
            size = self.save_to_model_zoo(model_info, args.output_dir, "Rust",
                                          temp_dir, arguments)
        else:
            size = -1

        return True, None, size

    def save_pytorch_model(self, model_info, args: Namespace, temp_dir: str,
                           model_zoo: bool):
        model_id = model_info.modelId
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        try:
            hf_pipeline = self.load_model(model_id)
        except Exception as e:
            logging.warning(f"Failed to load model: {model_id}.")
            logging.warning(e, exc_info=True)
            return False, "Failed to load model", -1

        try:
            # Save tokenizer.json to temp dir
            self.save_tokenizer(hf_pipeline, temp_dir)
        except Exception as e:
            logging.warning(f"Failed to save tokenizer: {model_id}.")
            logging.warning(e, exc_info=True)
            return False, "Failed to save tokenizer", -1

        # Save config.json just for reference
        config_file = self.get_file(model_id, "config.json")
        shutil.copyfile(config_file, os.path.join(temp_dir, "config.json"))

        # Save jit traced .pt file to temp dir
        include_types = "token_type_ids" in hf_pipeline.tokenizer.model_input_names
        model_file = self.jit_trace_model(hf_pipeline, model_id, temp_dir,
                                          include_types)
        if not model_file:
            return False, "Failed to trace model", -1

        result, reason = self.verify_jit_model(hf_pipeline, model_file,
                                               include_types, args.cpu_only)
        if not result:
            return False, reason, -1

        arguments = self.save_serving_properties(model_info, "PyTorch",
                                                 temp_dir, hf_pipeline,
                                                 include_types)
        if model_zoo:
            model_size = self.get_dir_size(temp_dir)
            if model_size > self.max_model_size:
                return False, f"Model size too large: {model_size}", -1
            size = self.save_to_model_zoo(model_info, args.output_dir,
                                          "PyTorch", temp_dir, arguments)
        else:
            size = -1

        return True, None, size

    @staticmethod
    def save_tokenizer(hf_pipeline, temp_dir: str):
        hf_pipeline.tokenizer.save_pretrained(temp_dir)
        if not os.path.exists(os.path.join(temp_dir, "tokenizer.json")):
            raise ValueError("no fast tokenizer found.")

        # only keep tokenizer.json file
        for path in os.listdir(temp_dir):
            if path != "tokenizer.json" and path != "tokenizer_config.json":
                os.remove(os.path.join(temp_dir, path))

    def jit_trace_model(self, hf_pipeline, model_id: str, temp_dir: str,
                        include_types: bool):
        logging.info(
            f"Tracing model: {model_id} include_token_types={include_types} ..."
        )
        encoding = self.encode_inputs(hf_pipeline.tokenizer)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids")
        if include_types and token_type_ids is None:
            return None

        # noinspection PyBroadException
        try:
            if include_types:
                script_module = torch.jit.trace(
                    ModelWrapper(hf_pipeline.model, include_types),
                    (input_ids, attention_mask, token_type_ids),
                    strict=False)
            else:
                script_module = torch.jit.trace(ModelWrapper(
                    hf_pipeline.model, include_types),
                                                (input_ids, attention_mask),
                                                strict=False)

            model_name = model_id.split("/")[-1]
            logging.info(f"Saving torchscript model: {model_name}.pt ...")
            model_file = os.path.join(temp_dir, f"{model_name}.pt")
            script_module.save(model_file)
        except Exception as e:
            logging.warning(f"Failed to trace model: {model_id}.")
            logging.warning(e, exc_info=True)
            return None

        return model_file

    def save_serving_properties(self, model_info, engine: str, temp_dir: str,
                                hf_pipeline, include_types: bool) -> dict:
        model_id = model_info.modelId
        model_name = model_id.split("/")[-1]

        serving_file = os.path.join(temp_dir, "serving.properties")
        arguments = self.get_extra_arguments(hf_pipeline, model_id, temp_dir)
        if include_types:
            arguments["includeTokenTypes"] = "true"
        arguments["translatorFactory"] = self.translator

        with open(serving_file, 'w') as f:
            f.write(f"engine={engine}\n"
                    f"option.modelName={model_name}\n")
            if engine == "PyTorch":
                f.write(f"option.mapLocation=true\n")

            for k, v in arguments.items():
                f.write(f"{k}={v}\n")

        arguments["engine"] = engine
        return arguments

    def save_to_model_zoo(self, model_info, output_dir: str, engine: str,
                          temp_dir: str, arguments: dict):
        model_id = model_info.modelId
        model_name = model_id.split("/")[-1]
        group_id = f"ai/djl/huggingface/{engine.lower()}"
        repo_dir = f"{output_dir}/model/{self.application}/{group_id}/{model_id}"
        model_dir = f"{repo_dir}/0.0.1"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save model as .zip file
        logging.info(f"Saving DJL model as zip: {model_name}.zip ...")
        zip_file = os.path.join(model_dir, f"{model_name}.zip")
        zip_dir(temp_dir, zip_file)

        # Save metadata.json
        sha1 = sha1_sum(zip_file)
        file_size = os.path.getsize(zip_file)
        metadata = HuggingfaceMetadata(model_info, engine, self.application,
                                       sha1, file_size, arguments)
        metadata_file = os.path.join(repo_dir, "metadata.json")
        metadata.save_metadata(metadata_file)

        return file_size

    @staticmethod
    def get_dir_size(path: str) -> int:
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += HuggingfaceConverter.get_dir_size(entry.path)
        return total

    def verify_jit_model(self, hf_pipeline, model_file: str,
                         include_types: bool, cpu_only: bool):
        logging.info(
            f"Verifying torchscript model(include_token_types={include_types}): {model_file} ..."
        )

        tokenizer = hf_pipeline.tokenizer
        encoding = self.encode_inputs(tokenizer)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids")
        if torch.cuda.is_available() and not cpu_only:
            traced_model = torch.jit.load(model_file, map_location='cuda:0')
            traced_model.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
        else:
            traced_model = torch.jit.load(model_file)

        traced_model.eval()

        try:
            # test traced model
            if include_types:
                out = traced_model(input_ids, attention_mask, token_type_ids)
            else:
                out = traced_model(input_ids, attention_mask)
        except RuntimeError as e:
            logging.warning(e, exc_info=True)
            return False, "Failed to run inference on jit model"

        return self.verify_jit_output(hf_pipeline, encoding, out)

    @staticmethod
    def get_file(model_id: str, file_name: str) -> str:
        if os.path.exists(model_id):
            return os.path.join(model_id, file_name)
        else:
            return hf_hub_download(repo_id=model_id, filename=file_name)

    def get_extra_arguments(self, hf_pipeline, model_id: str,
                            temp_dir: str) -> dict:
        return {}

    def verify_jit_output(self, hf_pipeline, encoding, out):
        if not hasattr(out, "last_hidden_layer"):
            return False, f"Unexpected inference result: {out}"

        return True, None

    def load_model(self, model_id: str):
        logging.info(f"Loading model: {model_id} ...")
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
